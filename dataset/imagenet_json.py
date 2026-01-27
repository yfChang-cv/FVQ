import os
import json
from collections import defaultdict
from tqdm import tqdm
def collect_train_files(openimage_dir, output_json):
    result = []
    stats = defaultdict(int)
    # 遍历当前目录下所有以 "train" 开头的文件夹
    total = 0
    dirs = [d for d in os.listdir(openimage_dir) if os.path.isdir(os.path.join(openimage_dir, d))]
    for dir_name in dirs:
            dir_path = os.path.join(openimage_dir, dir_name)
            num = 0
            # 遍历文件夹中的所有文件
            for file_name in tqdm(os.listdir(dir_path), desc=f"Processing files in {dir_name}", leave=False):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path)  and file_path.endswith('.JPEG'):
                    result.append(file_path)
                    
                    file_ext = os.path.splitext(file_name)[1]
                    stats[file_ext] += 1
                    num += 1
                    total += 1
            print(f"total: {total}, dir_name: {dir_name}, indir_num: {num}")
    
    # 将结果写入 JSON 文件
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)

# 使用示例
base_path = ""  # json 文件的保存路径
output_json = "train_files.json"  # 输出的 JSON 文件名
output_json = os.path.join(base_path, output_json)

openimage_dir = ""
collect_train_files(openimage_dir, output_json)

# train total: 
# val total: 
# test total: 