# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
import sys

from vq_model_qbridge import VQ_models
from qbridge import QBridge_models

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # Setup a feature folder:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)

   
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if 'args' in checkpoint:
        old_args = checkpoint['args']
        args.codebook_size = old_args.codebook_size
        args.codebook_embed_dim = old_args.codebook_embed_dim
        args.QB_type = old_args.QB_type
        
    
     # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        QB_type=args.QB_type,
        codebook_l2_norm=False,
        )
    vq_model.load_state_dict(checkpoint["model"])
    print(f'vq_model config: {vq_model.config}')
    vq_model.to(device)
    vq_model.eval()
    del checkpoint

    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = build_dataset(args, start_index=args.data_start_index, transform=transform)
    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    code_save_path = f'{args.code_path}/{args.dataset}{args.image_size}_codes/2'
    label_save_path = f'{args.code_path}/{args.dataset}{args.image_size}_labels/2'
    os.makedirs(code_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)

    pre_code_save_path = f'{args.code_path}/{args.dataset}{args.image_size}_codes/1'
    pre_label_save_path = f'{args.code_path}/{args.dataset}{args.image_size}_labels/1'
    os.makedirs(pre_code_save_path, exist_ok=True)
    os.makedirs(pre_label_save_path, exist_ok=True)

    total = args.data_start_index
    print("Total nums:", len(loader))
    for x, y in loader:
        train_steps = rank + total
        
        file_name = f'{train_steps}.npy'
        if (os.path.isfile(os.path.join(pre_code_save_path, file_name)) and os.path.isfile(os.path.join(pre_label_save_path, file_name))) or \
            (os.path.isfile(os.path.join(code_save_path, file_name)) and os.path.isfile(os.path.join(label_save_path, file_name))):
            if not args.debug:
                total += dist.get_world_size()
            else:
                total += 1
            print("No process:", total)
            continue
        
        # import pdb;pdb.set_trace()
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
        y = y.to(device)
        # fake_y = torch.full_like(y, 1000).to('cuda', non_blocking=True)
        fake_y = y.clone().to('cuda', non_blocking=True)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(x_all, fake_y) # x_all [B, C, H, W]
        codes = indices.reshape(x.shape[0], num_aug, -1)

        x = codes.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        # train_steps = rank + total
        # np.save(f'{code_save_path}/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        # np.save(f'{label_save_path}/{train_steps}.npy', y)

        if train_steps < 999999:
            np.save(f'{pre_code_save_path}/{train_steps}.npy', x)
            np.save(f'{pre_label_save_path}/{train_steps}.npy', y)
        else:
            np.save(f'{code_save_path}/{train_steps}.npy', x)
            np.save(f'{label_save_path}/{train_steps}.npy', y)


        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
        print(total)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--data-start-index", type=int, default=0)
    parser.add_argument("--debug", action='store_true')
    
    
    parser.add_argument("--QB_type", type=str, choices=list(QBridge_models.keys()), default="QBridge-L/4-cdp-02")
    args = parser.parse_args()
    main(args)
