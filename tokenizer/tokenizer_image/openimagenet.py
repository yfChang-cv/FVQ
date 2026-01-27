# 读取文件夹下的图像文件名，显示前5个
import os

# 指定图像文件夹路径
image_folder = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/OpenImages/data/train_0"

# 获取文件夹下的所有文件名
image_files = os.listdir(image_folder)

# 显示前5个图像文件名
print(image_files[:5])