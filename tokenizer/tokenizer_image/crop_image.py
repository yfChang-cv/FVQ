import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
def center_crop_images(input_folder, output_folder, crop_size=(256, 256), is_resize=False):
    """
    Center crop all images in the input folder and save them to the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save cropped images.
        crop_size (tuple): The size (height, width) to crop the images to.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    
    resize_p = transforms.Resize(256)

    # Define the center crop transform
    center_crop = transforms.CenterCrop(crop_size)

    # Process each image in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            # Open the image
            with Image.open(input_path) as img:
                # Apply the center crop
                
                cropped_img = center_crop(img)
                if is_resize:
                    cropped_img = resize_p(cropped_img)
                # Save the cropped image to the output folder
                cropped_img.save(output_path)
                # print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage
input_folder = "./data/dataset/imagenet/val"  # Replace with your input folder path
output_folder = "./data/evaluator_gen/imagenet_val_5w_256x256"  # Replace with your output folder path
center_crop_images(input_folder, output_folder, (256,256), False)
