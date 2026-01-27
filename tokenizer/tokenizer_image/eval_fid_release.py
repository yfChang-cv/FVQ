import os
import glob
import torch
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Callable, Optional, Iterator, Union
from torchvision.datasets import ImageFolder
import torch.distributed as dist
# from utils import dist, misc
# from models.vqvae import VQVAE
# from models.unitok import UniTok
# from utils.data import PlainDataset, normalize_01_into_pm1
# from utils.config import Args
import argparse
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# from vq_model_qbridge import VQ_models
from vq_model_qbridge_release import VQ_models

from torch.utils.data import Dataset, DataLoader
from qbridge_ViT import QBridge_models

import torch.nn.functional as F

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def eval_fid(args, vq_model, data_val, dir_raw, dir_recon, feature_extractor_path, result_path, ckpt_file):
    vq_model.eval()
    dataloader = prepare_eval_data(data_val, args)
    total_count = 0  # Initialize total count
    for files, labels, imgs in tqdm(dataloader, dynamic_ncols=False, leave=True, file=sys.stderr):
        imgs = imgs.to('cuda', non_blocking=True)

        labels = labels.to('cuda', non_blocking=True)
        # labels = torch.zeros_like(labels).to('cuda', non_blocking=True)
        # labels = torch.full_like(labels, 1000).to('cuda', non_blocking=True)
        # labels = torch.full_like(labels, args.num_classes).to('cuda', non_blocking=True)
        with torch.no_grad():
            # latent, _, [_, _, indices] = vq_model.encode(imgs, labels)
            # rec_imgs = vq_model.decode_code(indices, latent.shape, label=labels)
            rec_imgs, _ = vq_model(imgs)
        file_names = [os.path.join(dir_recon, f) for f in files]
        if args.image_size_eval != args.image_size:
            rec_imgs = F.interpolate(rec_imgs, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        save_img_tensor(rec_imgs, file_names)
        total_count += len(rec_imgs)  # Update total count
    dist.barrier()
    if args.rank == 0:
        print(f"Total images processed: {total_count}")  # Print total count
        # List all .JPEG files in the directory (excluding subdirectories)
        jpeg_files = [f for f in os.listdir(dir_recon) if f.lower().endswith('.jpeg') and os.path.isfile(os.path.join(dir_recon, f))]
        assert len(jpeg_files) == 50000, f"Expected 50000 JPEG files, got {len(jpeg_files)}"
        rfid, isc = get_fid_is(dir_raw, dir_recon, feature_extractor_path)
        print(f"FID: {rfid}, Inception Score: {isc}")

        result_file = os.path.join(result_path, "000_result.txt")
        with open(result_file, 'a') as f:
            print("%s, rfid: %f, IS: %f " % (ckpt_file, rfid, isc), file=f)
            
    
# class PlainDataset(Dataset):
#     def __init__(self, root, transform: Optional[Callable] = None):
#         self.root = root
#         self.transform = transform
#         self.img_files = os.listdir(root)
#         self.img_files = [f for f in self.img_files if f.lower().endswith('.jpeg')]

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx):
#         img_file = self.img_files[idx]
#         img_path = os.path.join(self.root, img_file)
#         with open(img_path, 'rb') as f:
#             img: Image.Image = Image.open(f).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img_file, img


class PlainDataset(ImageFolder):
    def __getitem__(self, index):
        path, class_index = self.samples[index]
        filename = path.split('/')[-1]
        image = self.loader(path)
        
        if self.transform:
            image = self.transform(image)
        
        return filename, class_index, image


def prepare_eval_data(dir_raw, args):
    # preprocess_val = transforms.Compose([
    #     transforms.CenterCrop((256, 256)),
    #     transforms.ToTensor(), normalize_01_into_pm1,
    # ])
    preprocess_val = transforms.Compose([
        # transforms.Resize(512),
        transforms.CenterCrop((args.image_size, args.image_size)),
        # transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Use the custom ImageFolderWithPaths
    # dataset = ImageFolderWithPaths(dir_raw, transform=preprocess_val)
    dataset = PlainDataset(dir_raw, transform=preprocess_val)
    # Use DistributedSampler for multi-GPU setup
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, pin_memory=True, batch_size=args.per_proc_batch_size, sampler=sampler, num_workers=args.num_workers)
    return dataloader


def save_img_tensor(recon_B3HW: torch.Tensor, paths: List[str]):  # img_tensor: [-1, 1]
    img_np_BHW3 = recon_B3HW.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255).to(
        dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    for bi, path in enumerate(paths):
        img_pil_HW3 = Image.fromarray(img_np_BHW3[bi])
        img_pil_HW3.save(path)


def get_fid_is(dir_raw: str, dir_recon: str, feature_extractor_path: str) -> Tuple[float, float]:
    import torch_fidelity
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dir_recon,
        input2=dir_raw,
        samples_shuffle=True,
        samples_find_deep=False,
        samples_find_ext='png,jpg,jpeg',
        samples_ext_lossy='jpg,jpeg',

        cuda=True,
        batch_size=1536,
        isc=True,
        fid=True,

        kid=False,
        kid_subsets=100,
        kid_subset_size=1000,

        ppl=False,
        prc=False,
        ppl_epsilon=1e-4 or 1e-2,
        ppl_sample_similarity_resize=64,
        feature_extractor='inception-v3-compat',
        feature_layer_isc='logits_unbiased',
        feature_layer_fid='2048',
        feature_layer_kid='2048',
        feature_extractor_weights_path=feature_extractor_path,
        verbose=True,

        save_cpu_ram=False,  # using num_workers=0 for any dataset input1 input2
        rng_seed=0,  # FID isn't sensitive to this
    )
    fid = metrics_dict['frechet_inception_distance']
    isc = metrics_dict['inception_score_mean']
    return fid, isc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--crop-raw-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--QB_type", type=str, choices=list(QBridge_models.keys()), default="QBridge-S/4")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=32) # ori:32
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--is-psnr-ssim", action='store_true', help="Enable PSNR andcalculation if this flag is set.") # TODO
    parser.add_argument("--codebook-l2-norm", action='store_true', default=False, help="l2 norm codebook")
    parser.add_argument("--is_uncondition", action='store_true', default=False, help="Qbridge mode")
    
    parser.add_argument("--num_classes", type=int, default=1000)
    
    args = parser.parse_args()
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    
    # create and load model
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    
    if 'args' in checkpoint:
        if args.rank == 0:
            print("original args")
            print(args)
            print("load args from ckpt")
            print(checkpoint['args'])
        old_args = checkpoint['args']
        args.codebook_size = old_args.codebook_size
        args.codebook_embed_dim = old_args.codebook_embed_dim
        args.QB_type = old_args.QB_type
        args.codebook_l2_norm = old_args.codebook_l2_norm
        if 'is_uncondition' in old_args:
            args.is_uncondition = old_args.is_uncondition
        if 'num_classes' in old_args:
            args.num_classes = old_args.num_classes
        if 'vq_model' in old_args:
            args.vq_model = old_args.vq_model
        if args.rank == 0:
            print("new args")
            print(args)
        
    # vq_model = VQ_models['VQ-16-z512'](
    #     codebook_size=65536,
    #     codebook_embed_dim=512,
    #     QB_type='QBridge-XL/8-cdp-02',
    #     codebook_l2_norm=False,
    #     is_uncondition=False,
    #     )
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        QB_type=args.QB_type,
        codebook_l2_norm=args.codebook_l2_norm,
        # is_uncondition=args.is_uncondition,
        )
    
    vq_model.to('cuda')
    vq_model.eval()
    
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    vq_model.load_state_dict(model_weight)
    
    # 将model_weight中的key embedding_monitor 替换成 qbridge    
    # new_model_weight = {}
    # for k, v in model_weight.items():
    #     new_key = k.replace('embedding_monitor', 'qbridge')
    #     new_model_weight[new_key] = v
    
    # vq_model.load_state_dict(new_model_weight)
    del checkpoint
    
    # Create folder to save samples:
    # folder_name = '/'.join(args.vq_ckpt.split('/log/')[-1].split('.')[:-1])
    # sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    folder_name = "qbridge-m-o"
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    # sample_folder_dir = sample_folder_dir + "_qbridge-m-o"
   
    if args.rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    
    data_val = args.dataset_path
    dir_raw = args.crop_raw_path
    dir_recon = sample_folder_dir
    feature_extractor_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/evaluator_gen/weights-inception-2015-12-05-6726825d.pth'
    
    result_path = os.path.dirname(args.vq_ckpt)
    ckpt_file = os.path.basename(args.vq_ckpt)
    eval_fid(args, vq_model, data_val, dir_raw, dir_recon, feature_extractor_path, result_path, ckpt_file)
    
    
