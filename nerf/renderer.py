import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import queue

import raymarching
from .utils import custom_meshgrid

def ComputeSH(dirs):
    '''
        dirs: b*3
    '''
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    xx = dirs[..., 0]**2
    yy = dirs[..., 1]**2
    zz = dirs[..., 2]**2

    xy = dirs[..., 0]*dirs[..., 1]
    yz = dirs[..., 1]*dirs[..., 2]
    xz = dirs[..., 0]*dirs[..., 2]


    sh = torch.zeros((dirs.shape[0], 25)).to(dirs.device)

    sh[:, 0] = 0.282095
    
    sh[:, 1] = -0.4886025119029199 * y
    sh[:, 2] = 0.4886025119029199 * z
    sh[:, 3] = -0.4886025119029199 * x

    sh[:, 4] = 1.0925484305920792 * xy
    sh[:, 5] = -1.0925484305920792 * yz
    sh[:, 6] = 0.31539156525252005 * (2.0 * zz - xx - yy)
    sh[:, 7] = -1.0925484305920792 * xz
    # sh2p2
    sh[:, 8] = 0.5462742152960396 * (xx - yy)


    sh[:, 9] = -0.5900435899266435 * y * (3 * xx - yy)
    sh[:, 10] = 2.890611442640554 * xy * z
    sh[:, 11] = -0.4570457994644658 * y * (4 * zz - xx - yy)
    sh[:, 12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy)
    sh[:, 13] = -0.4570457994644658 * x * (4 * zz - xx - yy)
    sh[:, 14] = 1.445305721320277 * z * (xx - yy)
    sh[:, 15] = -0.5900435899266435 * x * (xx - 3 * yy)


    sh[:, 16] = 2.5033429417967046 * xy * (xx - yy)
    sh[:, 17] = -1.7701307697799304 * yz * (3 * xx - yy)
    sh[:, 18] = 0.9461746957575601 * xy * (7 * zz - 1.0)
    sh[:, 19] = -0.6690465435572892 * yz * (7 * zz - 3.0)
    sh[:, 20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3)
    sh[:, 21] = -0.6690465435572892 * xz * (7 * zz - 3)
    sh[:, 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1.0)
    sh[:, 23] = -1.7701307697799304 * xz * (xx - 3 * yy)
    sh[:, 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return sh

def computeRGB(dirs, coeff, sh_dim):
    '''
    dirs: n*3
    coeff: n*25*3
    '''
    
    return ((ComputeSH(dirs)[...,:sh_dim].unsqueeze(-1) * coeff).sum(1))

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

class TreeNode:
    def __init__(self, flag=0, xyz = (0,0,0)):
        self.node_list = None
        self.xyz = xyz
        self.flag = flag

@torch.no_grad()
def line_interp(a,b,t):
    return a+(b-a)*t

class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4 # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        #print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        return {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
        }


    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            
            sigmas, rgbs = self(xyzs, dirs)
            # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            # sigmas = density_outputs['sigma']
            # rgbs = self.color(xyzs, dirs, **density_outputs)
            sigmas = self.density_scale * sigmas

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # special case for CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays, T_thresh)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
            
                depth = torch.stack(depths, axis=0) # [K, B, N]
                image = torch.stack(images, axis=0) # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)
            
            results['weights_sum'] = weights_sum

        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            total_ray_counter = 0
            loop_counter = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                total_ray_counter+=n_alive
                loop_counter = loop_counter + 1


                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                '''test space
                test_n_alive = int(1)
                test_n_step = int(8)
                test_num = int(52296)
                test_rays_alive = torch.arange(test_n_alive, dtype=torch.int32, device=device)
                test_rays_t = rays_t[test_num:test_num+1]
                test_rays_o = rays_o[test_num:test_num+1,:]
                test_rays_d = rays_d[test_num:test_num+1,:]
                test_nears = nears[test_num:test_num+1]
                test_fars = fars[test_num:test_num+1]
                test_xyzs, test_dirs, test_deltas = raymarching.march_rays(test_n_alive, test_n_step, test_rays_alive, test_rays_t, test_rays_o, test_rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, test_nears, test_fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                sigmas, rgbs = self(test_xyzs, test_dirs)
                #test_xyzs = test_xyzs[0:8,:]
                #test_dirs = test_dirs[0:8,:]
                #test_deltas = test_deltas[0:8,:]
                '''
                '''test space
                temp_numpy = self.density_bitfield.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\data\\density_bitfield.txt",temp_numpy)
                '''    
                '''
                temp_numpy = test_rays_t.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\input\\rays_t.txt",temp_numpy)
                
                temp_numpy = test_rays_o.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\input\\rays_o.txt",temp_numpy)

                temp_numpy = test_rays_d.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\input\\rays_d.txt",temp_numpy)

                temp_numpy = test_nears.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\input\\nears.txt",temp_numpy)

                temp_numpy = test_fars.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\input\\fars.txt",temp_numpy)


                temp_numpy = test_xyzs.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\output\\xyzs.txt",temp_numpy)

                temp_numpy = test_dirs.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\output\\dirs.txt",temp_numpy)

                temp_numpy = test_deltas.cpu().numpy()
                np.savetxt("D:\\workwork\\path_nerf\\ngp\\output\\deltas.txt",temp_numpy)

                sigmas, rgbs = self(test_xyzs, test_dirs)
                '''
                

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs = self(xyzs, dirs)
                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image

        return results

#my part--------------
    def run_dummy_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            
            sigmas, rgbs_15 = self(xyzs)
            #rgbs = self.forward_color(rgbs_15,dirs)
            # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            # sigmas = density_outputs['sigma']
            # rgbs = self.color(xyzs, dirs, **density_outputs)
            sigmas = self.density_scale * sigmas

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # len(sigmas.shape) should be 1
            weights_sum, depth, image = raymarching.dummpy_composite_rays_train(sigmas, rgbs_15, deltas, rays, T_thresh)

            image = self.forward_color(image,rays_d)
            #'''one point
            bg_color = 0
            #'''
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            
            results['weights_sum'] = weights_sum

        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            rgb_feature_num = (int)(15)
            image = torch.zeros(N, rgb_feature_num, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            #total_ray_counter = 0
            #loop_counter = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                #total_ray_counter+=n_alive
                #loop_counter = loop_counter + 1
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)          

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs_15 = self(xyzs)
                #rgbs = self.forward_color(rgbs_15,dirs)
                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                sigmas = self.density_scale * sigmas

                raymarching.composite_dummy_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs_15, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step
            
            #'''one point
            bg_color = 0
            #'''
            image = self.forward_color(image,rays_d)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image

        return results
#-------------------

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)
        
        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign 
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        #self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')


    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        use_one_point = True
        if(use_one_point):
            _run = self.run_dummy_cuda
        elif self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results
    
    @torch.no_grad()
    def vis_density(self, mins, maxs, L= 32):
    
        x = torch.linspace(mins[0],maxs[0],steps=L).cuda()
        y = torch.linspace(mins[1],maxs[1],steps=L).cuda()
        z = torch.linspace(mins[2],maxs[2],steps=L).cuda()
        grid_x ,grid_y,grid_z = torch.meshgrid(x, y,z)
        xyz = torch.stack([grid_x ,grid_y,grid_z], dim = -1)  #(L,L,L,3)

        xyz = xyz.reshape((-1,3)) #(L*L*L,3)

        xyzs = xyz.split(5000, dim=0)

        sigmas = []
        coeffs = []
        for i in tqdm(xyzs):
            with torch.no_grad():
                density, coeff = self.octree_forward(i) #(L*L*L,1)
                sigmas.append(density.detach().cpu())
                coeffs.append(coeff.detach().cpu())
                
        sigmas = torch.cat(sigmas, dim=0)
        coeffs = torch.cat(coeffs, dim=0)

        return sigmas, coeffs
    
    @torch.no_grad()
    def gen_octree(self,x, y, z, size):
        
        print("\r{0}/{1}".format(self.vis_count,self.L*self.L*self.L),end="")
        self.vis_count = self.vis_count+1
        if(size<=2):
            node_list = []
            flag=False
            for i in range(x, x+2):
                for j in range(y, y+2):
                    for k in range(z, z+2):
                        node_list.append(TreeNode(flag=0, xyz = (i,j,k)))
                        if(self.grid[i,j,k]):
                            flag = True
            
            node = TreeNode(1, (x+size//2, y+size//2, z+size//2))
            if(flag==True):
                node.node_list = node_list
            else:
                node.node_list = None
                node.flag=0
            return flag, node

        flag = False
        node_list = []
        
        cnt=1
        for i in range(x, x+size, size//2):
            for j in range(y, y+size, size//2):
                for k in range(z, z+size, size//2):
                    _flag, node = self.gen_octree(i, j, k, size//2)
                    
                    node_list.append(node)
                    if(_flag==True):
                        flag = _flag
                    cnt+=1
        
        node = TreeNode(1, (x+size//2, y+size//2, z+size//2))
        if(flag):
            node.node_list = node_list
        else:
            node.flag=0
        
        return flag, node
    
    @torch.no_grad()
    def BFS(self, node, coeffs):
        L = self.L
        x_val = self.x_val
        y_val = self.y_val
        z_val = self.z_val
        q = queue.Queue()
        cnt_q = queue.Queue()
        q.put(node)
        cnt_q.put(0)
        cnt=1
        total_cnt=0
        child = []
        density_coeff = []
        sample_count = 4
        vis_count = 0
        while(not q.empty()):
            cnt_tmp = cnt
            cnt = 0
            for i in range(cnt_tmp):
                node = q.get()
                cnt_num = cnt_q.get()
                
                for j in range(len(node.node_list)):
                    vis_count = vis_count+1
                    print("\r{0}/{1}".format(vis_count,L*L*L),end="")
                    xyz = node.node_list[j].xyz
                    xx = xyz[0]
                    yy = xyz[1]
                    zz = xyz[2]
                    now_coeff = coeffs[xx:xx+1, yy:yy+1, zz:zz+1]
                    '''sample points
                    xx_left = xx
                    xx_right = xx
                    if(xx-1>=0):
                        xx_left = xx-1
                    if(xx+1<L):
                        xx_right = xx+1
                    
                    yy_left = yy
                    yy_right = yy
                    if(yy-1>=0):
                        yy_left = yy-1
                    if(yy+1<L):
                        yy_right = yy+1

                    zz_left = zz
                    zz_right = zz
                    if(zz-1>=0):
                        zz_left = zz-1
                    if(zz+1<L):
                        zz_right = zz+1
                    
                    x_list = torch.linspace(line_interp(x_val[xx_left],x_val[xx],0.8),line_interp(x_val[xx],x_val[xx_right],0.2),steps=sample_count).cuda()
                    y_list = torch.linspace(line_interp(y_val[yy_left],y_val[yy],0.8),line_interp(y_val[yy],y_val[yy_right],0.2),steps=sample_count).cuda()
                    z_list = torch.linspace(line_interp(z_val[zz_left],z_val[zz],0.8),line_interp(z_val[zz],z_val[zz_right],0.2),steps=sample_count).cuda()
                    grid_x ,grid_y,grid_z = torch.meshgrid(x_list, y_list, z_list)
                    xyz_list = torch.stack([grid_x ,grid_y,grid_z], dim = -1)  #(L,L,L,3)

                    xyz_list = xyz_list.reshape((-1,3)) #(L*L*L,3)

                    coeff_box = []
                    coeff_box.append(now_coeff.reshape(1,now_coeff.shape[3]))
                    with torch.no_grad():
                        _, coeff = self.octree_forward(xyz_list)
                        coeff_box.append(coeff.detach().cpu().numpy())

                    coeff_box = np.concatenate(coeff_box, axis=0)
                    '''
                    
                    density_coeff.append(coeffs[xx, yy, zz]) #coeffs[xx, yy, zz] coeff_box.mean(0)
                    if(node.node_list[j].flag==1):
                        total_cnt+=1
                        q.put(node.node_list[j])
                        cnt_q.put(total_cnt)
                        child.append(total_cnt - cnt_num)
                        cnt += 1
                    else:
                        child.append(0)
                        
        child = np.array(child).reshape((-1,2,2,2))
        density_coeff = np.array(density_coeff)
        density_coeff = density_coeff.reshape((-1,2,2,2,28))
        return child, density_coeff

    @torch.no_grad()
    def render_octree(self, staged=False, max_ray_batch=4096, **kwargs):
        L = 128
        self.L = L
        iv0 = 0.5
        iv1 = 0.5
        iv2 = 0.5
        offset0 = 0.5
        offset1 = 0.5
        offset2 = 0.5
        #should be self.aabb_infer one day
        maxs = (np.array([1.0, 1.0, 1.0]) - np.array([offset0, offset1, offset2]))/np.array([iv0,iv1,iv2])
        mins = (np.array([0.0, 0.0, 0.0]) - np.array([offset0, offset1, offset2]))/np.array([iv0,iv1,iv2])
        print("start visiting density....")
        sigma, coeffs = self.vis_density(mins,maxs, L)

        self.grid = sigma.reshape((L, L, L))>0.01
        print("start building octree....")
        self.vis_count = 0
        flag, node = self.gen_octree(0,0,0,L)

        coeffs = coeffs.reshape((L, L, L, 28)).cpu().numpy()

        self.x_val = torch.linspace(mins[0],maxs[0],steps=L).cuda()
        self.y_val = torch.linspace(mins[1],maxs[1],steps=L).cuda()
        self.z_val = torch.linspace(mins[2],maxs[2],steps=L).cuda()

        print("start BFS travel....")
        child, density_coeff = self.BFS(node, coeffs)

        self.iv0 = iv0
        self.iv1 = iv1
        self.iv2 = iv2
        self.offset0 = offset0
        self.offset1 = offset1
        self.offset2 = offset2

        return child, density_coeff