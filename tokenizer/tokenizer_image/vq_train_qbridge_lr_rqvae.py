# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import re
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from copy import deepcopy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
# from tokenizer.tokenizer_image.vq_model_qbridge import VQ_models
from tokenizer.tokenizer_image.vq_model_qbridge_rqvae import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss
from torch.utils.tensorboard import SummaryWriter
from qbridge import QBridge_models
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR

import warnings
warnings.filterwarnings('ignore')

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory created at {args.results_dir}")
        checkpoint_dir = args.results_dir
        writer = SummaryWriter(log_dir=os.path.join(args.results_dir, "logs"))
    else:
        logger = create_logger(None)
        writer = None

    # training args
    logger.info(f"==============================================")
    logger.info(f"======= THIS IS LLAMAGEN's QBridge TRAINING =======")
    logger.info(f"==============================================")
    
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        QB_type=args.QB_type,
        codebook_l2_norm=args.codebook_l2_norm,
        is_uncondition=args.is_uncondition,
        emb_nograd=args.emb_nograd,
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    model_parts = ["encoder", "decoder", "quantize", "quant_conv", "post_quant_conv"]
    for part in model_parts:
        logger.info(f"VQ Model {part}: {sum(p.numel() for p in getattr(vq_model, part).parameters()):,}")
        logger.info(f"VQ Model {part} (requires_grad): {sum(p.numel() for p in getattr(vq_model, part).parameters() if p.requires_grad):,}")
        
    quantize_parts = ["embedding", "qbridge"]
    for part in quantize_parts:
        logger.info(f"Quantize {part}: {sum(p.numel() for p in getattr(vq_model.quantize, part).parameters()):,}")
        logger.info(f"Quantize {part} (requires_grad): {sum(p.numel() for p in getattr(vq_model.quantize, part).parameters() if p.requires_grad):,}")
        

    
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
        
    
    vq_model = vq_model.to(device)

    
    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,  
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    tlr = args.lr * (args.global_batch_size / 128)
    
    if args.finetune_from_llmgen:
       base_params = [p for name, p in vq_model.named_parameters() if "qbridge" not in name]
       optimizer = torch.optim.Adam(base_params, lr=tlr, betas=(args.beta1, args.beta2))
    else:
        optimizer = torch.optim.Adam(vq_model.parameters(), lr=tlr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=tlr, betas=(args.beta1, args.beta2))
    
    # Setup data:
    transform = transforms.Compose([
        # transforms.Resize(args.image_size),  # 短边缩放到 args.image_size
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    def combined_lr_lambda(current_step: int):
        warmup_steps = args.warmup_ep * len(loader)  # 每个 epoch 的 step 数
        total_steps = args.epochs * len(loader)
        wp0 = 0.005
        wpe = 0.01
        T=args.lrT; max_rest=1-T
        if current_step < warmup_steps:
            return wp0 + (1-wp0) * (float(current_step) / float(max(1, warmup_steps)))  # 从 0.1 开始
        else:
            pasd = (current_step - warmup_steps) / (total_steps-1 - warmup_steps)
            rest = 1 - pasd
            if pasd < T: return 1.0
            else: return wpe + (1 - wpe) * rest / max_rest
    
    scheduler = LambdaLR(optimizer, lr_lambda=combined_lr_lambda)
    scheduler_disc = LambdaLR(optimizer_disc, lr_lambda=combined_lr_lambda)
    
    # Prepare models for training:
    if args.resume:
        if args.vq_ckpt:
            checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        else:
            pattern = '*.pt'
            file = os.path.join(args.results_dir, pattern)
            all_ckpt = sorted(
                [f for f in glob(file, recursive=False) if int(re.search(r'\d+', os.path.basename(f)).group()) % 10000 == 0],
                key=os.path.getmtime,
                reverse=True
            )
            # all_ckpt = sorted(glob.glob(file, recursive=False), key=os.path.getmtime, reverse=True)
            if len(all_ckpt) == 0:
                train_steps = 0
                start_epoch = 0
                if args.ema:
                    update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
            else:
                args.vq_ckpt = all_ckpt[0]
                checkpoint = torch.load(all_ckpt[0], map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if 'last' in args.vq_ckpt:
            start_epoch = checkpoint["epoch"]
            train_steps = checkpoint["steps"]
        else:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))+1
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))-50
        
        scheduler.step(train_steps)
        scheduler_disc.step(train_steps)
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    elif args.finetune_from_llmgen:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        
        
        vq_model.load_state_dict(checkpoint["model"], strict=False)
        
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # 手动将新模块的参数添加到优化器中
        new_params = vq_model.quantize.qbridge.parameters()
        optimizer.add_param_group({"params": new_params, "lr": optimizer.param_groups[0]['lr']})
        
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        
        train_steps = 0
        start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        pattern = '*.pt'
        file = os.path.join(args.results_dir, pattern)
        all_ckpt = sorted(glob(file, recursive=False), key=os.path.getmtime, reverse=True)
        if len(all_ckpt) > 0:
            raise ValueError(f"Please remove the previous ckpt folder in {args.results_dir}.")
        
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu])
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    epoch_loss = 0
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            imgs = x.to(device, non_blocking=True)
            if args.mode == 'qbridge-m':
                labels = y.to(device, non_blocking=True)
            elif args.mode == 'qbridge-o':
                labels = torch.zeros_like(y).to(device, non_blocking=True)
            
           
            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype): 
                recons_imgs, codebook_loss = vq_model(imgs, labels)
                loss_gen, loss_writer = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=vq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training   
                    
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            
            
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            epoch_loss += loss_gen.item() + loss_disc.item()
            
            log_steps += 1
            train_steps += 1
            
            if writer is not None and  train_steps % args.log_every == 0:
            #  codebook_loss: vq_loss, commit_loss, entropy_loss, usage
                writer.add_scalar("Loss/vq_loss", codebook_loss[0].item(), train_steps)
                writer.add_scalar("Loss/commit_loss", codebook_loss[1].item(), train_steps)
                writer.add_scalar("Loss/entropy_loss", codebook_loss[2].item(), train_steps)
                
                writer.add_scalar("Codebook Usage", codebook_loss[3]*100, train_steps)
            # rec_loss
                writer.add_scalar("Loss/rec_loss", loss_writer[0].item(), train_steps) # rec_loss, p_loss, generator_adv_loss
                writer.add_scalar("Loss/p_loss", loss_writer[1].item(), train_steps)
                writer.add_scalar("Loss/generator_adv_loss", loss_writer[2].item(), train_steps)
            
            # adv_loss
                writer.add_scalar("Loss/discriminator_adv_loss", loss_disc.item(), train_steps)
               
            # sum
                writer.add_scalar("Total_Loss/gen_loss", loss_gen.item(), train_steps)
                writer.add_scalar("Total_Loss/disc_loss", loss_disc.item(), train_steps)
            # SUM
                writer.add_scalar("Total_Loss/loss", running_loss/log_steps, train_steps)
                
            # LR
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], train_steps)
                
            
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            scheduler.step()
            scheduler_disc.step()
            
            
            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
        if rank == 0:
            if args.compile:
                model_weight = vq_model.module._orig_mod.state_dict()
            else:
                model_weight = vq_model.module.state_dict()  
            checkpoint = {
                "model": model_weight,
                "optimizer": optimizer.state_dict(),
                "discriminator": vq_loss.module.discriminator.state_dict(),
                "optimizer_disc": optimizer_disc.state_dict(),
                "steps": train_steps,
                "args": args,
                "epoch": epoch+1
            }
            if args.ema:
                checkpoint["ema"] = ema.state_dict()
            
            checkpoint_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        dist.barrier()
        epoch_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch} Loss: {epoch_loss}")
        if writer is not None:
            writer.add_scalar("Epoch_Loss", epoch_loss, epoch)
        epoch_loss = 0
        
    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    
    if writer is not None:
        writer.close()
        
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    # parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=False, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--finetune-from-llmgen", action='store_true')
    
    parser.add_argument("--QB_type", type=str, choices=list(QBridge_models.keys()), default="QBridge-S/4")
    parser.add_argument("--warmup_ep", type=int, default=1)
    parser.add_argument("--is_uncondition", action='store_true', default=False, help="Qbridge mode")
    parser.add_argument("--mode", type=str, choices=["qbridge-o", "qbridge-m"], default="qbridge-m")
    parser.add_argument("--lrT", default=0.3, type=float, help="learning rate change time")
    parser.add_argument("--emb_nograd", action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.finetune_from_llmgen:
        args.disc_start = 0
    
    main(args)
