import torch

from nerf.network import NeRFNetwork
from nerf.provider import NeRFDataset
from nerf.utils import *

import argparse

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=12800)
    
    parser.add_argument('--radius', type=float, default=2, help="assume the camera is located on sphere(0, radius))")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in sphere(0, size)")

    opt = parser.parse_args()

    print(opt)

    seed_everything(opt.seed)

    train_dataset = NeRFDataset(opt.path, 'train', radius=opt.radius)
    valid_dataset = NeRFDataset(opt.path, 'valid', downscale=4, radius=opt.radius)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
    
    model = NeRFNetwork(encoding="hashgrid", encoding_dir="sphere_harmonics", num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64)
    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)

    print(model)

    criterion = torch.nn.SmoothL1Loss()

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)

    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=False, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1)

    trainer.train(train_loader, valid_loader, 200)

    # test dataset
    test_dataset = NeRFDataset(opt.path, 'test', downscale=4, radius=opt.radius)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
