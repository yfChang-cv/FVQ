from PIL import Image
import requests
from transformers import AutoProcessor
from tokenizer.tokenizer_image.models.siglip.modeling_siglip_clip import SiglipVisionModel
import torch


model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/ckpt/google/siglip2-base-patch16-256"
model = SiglipVisionModel.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

url = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/vision/img/ILSVRC2012_val_00018009.JPEG"
image = Image.open(url)

# texts = ["a photo of a cat", "a photo of a dog"]
# inputs = processor(text=texts, images=image, return_tensors="pt")

def extract_features_siglip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        image_features = model(**inputs)
    return image_features.last_hidden_state

print(image.size)
image_features=extract_features_siglip(image)
print(image_features.shape)