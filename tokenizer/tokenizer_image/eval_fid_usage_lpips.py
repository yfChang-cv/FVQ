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



from vq_model_qbridge import VQ_models

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from qbridge import QBridge_models
from misc import MetricLogger
import piq
import pyiqa

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.distributed import init_distributed_mode


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

# def eval_fid(args, vae, data_val, dir_raw, dir_recon, feature_extractor_path):
#     vae.eval()

    
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
        transforms.CenterCrop((256, 256)),
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
    
    args = parser.parse_args()
    
    # Setup DDP:
    init_distributed_mode(args)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    # args.local_rank = int(os.environ['LOCAL_RANK'])
    # args.world_size = dist.get_world_size()
    # args.rank = dist.get_rank()
    
    # create and load model
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    
    if 'args' in checkpoint:
        if rank == 0:
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
        
    
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        QB_type=args.QB_type,
        codebook_l2_norm=args.codebook_l2_norm,
        is_uncondition=args.is_uncondition,
        )
    
    vq_model.to(device)
    
    
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    vq_model.load_state_dict(model_weight)
    del checkpoint
    
    # Create folder to save samples:
    # folder_name = '/'.join(args.vq_ckpt.split('/log/')[-1].split('.')[:-1])
    folder_name = "qbridge-m-o"
    # sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    # sample_folder_dir = sample_folder_dir + "_qbridge-m-o"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    
    ### metriclogger
    metric_logger = MetricLogger(delimiter="  ")
    header = ""
    print_freq = 10
    
    #### psnr ssim lpips
    psnr_computer = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    # ssim_computer = pyiqa.create_metric('ssim', downsample=True, device=device)
    psnr_total = 0.0
    ssim_total = 0.0

    torch.hub.set_dir("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/ckpt")
    lpips_computer = pyiqa.create_metric('lpips', device=device, pretrained_model_path='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/ckpt/checkpoints/LPIPS_v0.1_alex-df73285e.pth')
    lpips_total = 0.0
    num_images = 0
    
    ### rfid
    data_val = args.dataset_path
    dir_raw = args.crop_raw_path
    dir_recon = sample_folder_dir
    feature_extractor_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/evaluator_gen/weights-inception-2015-12-05-6726825d.pth'
    
    dataloader = prepare_eval_data(data_val, args)
    vq_model.eval()
    total_count = 0  # Initialize total count
    
    usage = set()
    iter_count = 0
    for files, labels, imgs in tqdm(dataloader, dynamic_ncols=False, leave=True, file=sys.stderr):
        imgs = imgs.to(device, non_blocking=True)

        # labels = labels.to('cuda', non_blocking=True)
        # labels = torch.zeros_like(labels).to('cuda', non_blocking=True)
        labels = torch.full_like(labels, 1000).to(device, non_blocking=True)
        with torch.no_grad():
            latent, _, [_, _, indices] = vq_model.encode(imgs, labels)
            rec_imgs = vq_model.decode_code(indices, latent.shape, label=labels)
        
        file_names = [os.path.join(dir_recon, f) for f in files]
        save_img_tensor(rec_imgs, file_names)
        total_count += len(rec_imgs)  # Update total count
        
        ### lpips
        b = imgs.shape[0]
        num_images += b
        # save_x = (imgs + 1) * 127.5  
        # rec_imgs[rec_imgs > 1] = 1
        # rec_imgs[rec_imgs < -1] = -1
        # save_xrec = (rec_imgs + 1) * 127.5
        
        # save_x = save_x / 255.0
        # save_xrec = save_xrec / 255.0
        save_x = imgs.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255).div_(255.0)
        save_xrec = rec_imgs.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255).div_(255.0)
        
        lpips_score = lpips_computer(save_x, save_xrec)
        lpips_total += torch.sum(lpips_score)
        metric_logger.update(lpips=torch.sum(lpips_score)/b)
        
        ### psnr ssim
        psnr_score = psnr_computer(save_x, save_xrec)
        psnr_total += torch.sum(psnr_score)
        metric_logger.update(psnr=torch.sum(psnr_score)/b)
        
        # ssim_score = ssim_computer(save_x, save_xrec)
        # ssim_total += torch.sum(ssim_score)
        # metric_logger.update(ssim=torch.sum(ssim_score)/b)
        ssim_score = piq.ssim(save_x, save_xrec, data_range=2., reduction='none') # 2. follow llamagen
        ssim_total += torch.sum(ssim_score)
        metric_logger.update(ssim=torch.sum(ssim_score)/b)
         
        ### usage
        usage.update(indices.view(-1).cpu().numpy())
        iter_count += 1
        # if iter_count % 100 == 0:
        #    print(f"iter {iter_count}, usage: {len(usage)/args.codebook_size}")
    dist.barrier()
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # print("LPIPS:", lpips_total.item()/num_images)
    # print("PSNR:", psnr_total.item()/num_images)
    # print("SSIM:", ssim_total.item()/num_images)
    print("usage: ", len(usage)/args.codebook_size)
    
    if rank == 0:
        print(f"Total images processed: {total_count}")  # Print total count
        # List all .JPEG files in the directory (excluding subdirectories)
        jpeg_files = [f for f in os.listdir(dir_recon) if f.lower().endswith('.jpeg') and os.path.isfile(os.path.join(dir_recon, f))]
        assert len(jpeg_files) == 50000, f"Expected 50000 JPEG files, got {len(jpeg_files)}"
        rfid, isc = get_fid_is(dir_raw, dir_recon, feature_extractor_path)
        print(f"FID: {rfid}, Inception Score: {isc}")

        result_file = os.path.join(dir_recon, "000_result.txt")
        with open(result_file, 'w') as f:
            print(f"rfid: {rfid}, IS: {isc} , {metric_logger}, usage: {len(usage)/args.codebook_size}", file=f)
    
