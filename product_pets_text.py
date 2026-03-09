
# import os
# os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['MKL_NUM_THREADS'] = '4'

# import json
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image

# # ── 模型加载 ──────────────────────────────────────────────
# local_model_path = "blip-model"
# print("Loading model...", flush=True)
# processor = BlipProcessor.from_pretrained(local_model_path)
# model = BlipForConditionalGeneration.from_pretrained(local_model_path)

# # 将模型移动到 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# print(f"Model loaded on {device}", flush=True)

# # ── 路径配置 ──────────────────────────────────────────────
# BASE_DIR  = '/root/autodl-tmp/PIXIU'
# # Pets数据集：文件名本身带类名（如 Abyssinian_1.jpg），直接用 JSON 里的文件名
# jpg_dir   = os.path.join(BASE_DIR, 'data/Oxford_pets/jpg')
# json_path = os.path.join(BASE_DIR, 'data/Oxford_pets/split_zhou_OxfordPets.json')
# out_dir   = os.path.join(BASE_DIR, 'test/word')
# out_train = os.path.join(out_dir, 'pets_file_list_train.txt')
# out_test  = os.path.join(out_dir, 'pets_file_list_test.txt')
# os.makedirs(out_dir, exist_ok=True)

# # ── 读取官方 train/test 划分 ──────────────────────────────
# # JSON 格式：{"train": [["Abyssinian_1.jpg", label_int, "abyssinian"], ...], "test": [...]}
# with open(json_path, 'r') as f:
#     splits = json.load(f)

# train_entries = splits.get('train', [])
# test_entries  = splits.get('test',  [])
# print(f"Train: {len(train_entries)}, Test: {len(test_entries)}", flush=True)

# # ── 生成描述并写入文件 ────────────────────────────────────
# def generate(entries, out_file, split_name):
#     total = len(entries)
#     ok = 0
#     # buffering=1: 行缓冲，每行写完立即落盘
#     with open(out_file, 'w', buffering=1, encoding='utf-8') as f:
#         for idx, entry in enumerate(entries, 1):
#             filename   = entry[0]       # Abyssinian_1.jpg（实际存在）
#             label      = int(entry[1])  # 数字标签
#             class_name = entry[2]       # 品种名，如 abyssinian

#             abs_path = os.path.join(jpg_dir, filename)
#             # 写入相对路径，与训练脚本运行目录（~/autodl-tmp/PIXIU）匹配
#             rel_path = f"data/Oxford_pets/jpg/{filename}"

#             if not os.path.exists(abs_path):
#                 print(f"[WARN] 文件不存在，跳过：{abs_path}", flush=True)
#                 continue
#             try:
#                 image   = Image.open(abs_path).convert('RGB')
#                 inputs  = processor(images=image, return_tensors="pt").to(device)
#                 outputs = model.generate(**inputs, max_new_tokens=50)
#                 caption = processor.decode(outputs[0], skip_special_tokens=True)

#                 # 过滤类别词防止标签泄漏；替换分号防止解析错乱
#                 stop     = set(class_name.lower().replace('_', ' ').split())
#                 filtered = ' '.join(w for w in caption.split() if w.lower() not in stop)
#                 filtered = filtered.replace(';', ',')

#                 f.write(f"{rel_path};{filtered};{label}\n")
#                 f.flush()
#                 ok += 1

#                 if idx % 50 == 0 or idx == total:
#                     print(f"[{split_name} {idx}/{total}] {filename}: {filtered[:55]}", flush=True)
#             except Exception as e:
#                 print(f"[ERROR] {filename}: {e}", flush=True)
#     print(f"{split_name} done: {ok}/{total} written -> {out_file}", flush=True)

# print("\n>>> 生成训练集...", flush=True)
# generate(train_entries, out_train, 'TRAIN')

# print("\n>>> 生成测试集...", flush=True)
# generate(test_entries,  out_test,  'TEST')

# print("\nAll done!", flush=True)

import os
import random
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor
local_model_path = "blip-model"
processor = BlipProcessor.from_pretrained(local_model_path)
model = BlipForConditionalGeneration.from_pretrained(local_model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Define dataset path and output files
dataset_path = 'data/Oxford_pets/jpg'
output_train_file = "test/word/pets_file_list_train.txt"
output_test_file = "test/word/pets_file_list_test.txt"


os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
os.makedirs(os.path.dirname(output_test_file), exist_ok=True)


image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]


random.shuffle(image_files)

# Function: Extract category (including full model type) from filename
def get_category(filename):
    return filename.split('_')[0]

# Assign a numeric label to each category
categories = sorted(set(get_category(f) for f in image_files))
category_to_label = {category: i for i, category in enumerate(categories)}

# Split dataset into 80% training and 20% test
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
test_files = image_files[split_index:]


def generate_descriptions(image_files, output_file):
    with open(output_file, 'w') as f:
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            image = Image.open(image_path)


            category = get_category(image_file)
            label = category_to_label[category]


            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            filtered_caption = ' '.join([word for word in caption.split() if category.lower() not in word.lower()])

            f.write(f"{image_path};{filtered_caption};{label}\n")

            print(f"Processed {image_file}")

generate_descriptions(train_files, output_train_file)

generate_descriptions(test_files, output_test_file)

print("All descriptions have been generated and saved to train_descriptions.txt and test_descriptions.txt")