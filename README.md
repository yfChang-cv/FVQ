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