<h1 align="center"> Scalable Training for Vector-Quantized Networks with 100% Codebook Utilization </h1>

<div align="center">
  <p>
   <strong>Yifan Chang</strong>
    ·
    <strong>Jie Qin</strong>
    ·
    <strong>Zicong Hong</strong>
    ·
    <strong>Limeng Qiao</strong>
    ·
    <strong>Xiaofeng Wang</strong>
    ·
    <strong>Zheng Zhu</strong>
    ·
    <strong>Lin Ma</strong>
    ·
    <strong>Xingang Wang</strong>
    <!-- <br><br>
    <b><sup>1</sup>Hong Kong University of Science and Technology &nbsp; | &nbsp; <sup>2</sup>ByteDance Seed</b>
    <br>
    <em>* Corresponding author</em> -->
  </p>
</div>
<p align="center">
  <a href="https://arxiv.org/abs/2509.10140">
    <img src="https://img.shields.io/badge/arXiv-2509.10140-b31b1b.svg" alt="arXiv">
  </a>
  <!-- <a href="https://WM-PO.github.io">
    <img src="https://img.shields.io/badge/Project%20Page-Website-1f6feb.svg?logo=googlechrome&logoColor=white" alt="Project Page">
  </a> -->
  <a href="https://huggingface.co/yfChang/FVQ">
    <img src="https://img.shields.io/badge/Checkpoints-%F0%9F%A4%97%20HuggingFace-ffd21e.svg?logo=huggingface" alt="Checkpoints">
  </a>
</p>



# FVQ
We’re releasing the initial version first, and will keep improving it, including uploading the model checkpoints later.


# Installation
1.Install python==3.10.6 pytroch==2.1.0+cu118

2.Install other pip packages via `pip3 install -r requirements.txt.`

3.Prepare the `ImageNet` dataset


# Training Scripts
Please refer to `scripts/recon`

VQBridge Implementation
In implementing VQBridge, we referenced the DiT design and provide two implementation methods. The first method directly uses `DiT`, please refer to `vq_train_qbridge_lr.py`. The second method uses `ViT blocks`, which is more streamlined and corresponds to the method described in the paper, making it more efficient. Please refer to `vq_train_qbridge_release.py`.

To convert FVQ models to VQGAN format, please refer to `scripts/convert_fuq2vq.sh` and `tokenizer/tokenizer_image/convert_fullvq2vq.py`.

# Eval FVQ
First, use `tokenizer/tokenizer_image/crop_image.py` to crop the ImageNet validation dataset to 256x256 resolution and save it to `data/evaluator_gen/imagenet_val_5w_256x256`.

We release FVQ models in VQGAN format for compatibility.

Evaluation Options

**Option 1: Using VQGAN Format (Recommended)**

If you are using our released FVQ models in VQGAN format, use the following evaluation script: `bash eval.sh`

**Option 2: Using FVQ Format**

If you have trained your own FVQ model and want to evaluate it directly, you can replace `eval_fid_vqgan.py` with `eval_fid.py` in eval.sh.

# Eval Generation

Please refer to the LlamaGen codebase for implementation details.

Recommended CFG Scale Settings:
```
LlamaGen-L: Use CFG scale 1.75
LlamaGen-XL: Use CFG scale 1.65
```