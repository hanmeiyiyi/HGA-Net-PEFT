import os
import json

# JSON file path
json_file_path = "data/Oxford_pets/split_zhou_OxfordPets.json"
image_dir = "data/Oxford_pets/jpg"
# json_file_path = "/root/autodl-tmp/PIXIU/data/Oxford_pets/split_zhou_OxfordPets.json"
# image_dir = "/root/autodl-tmp/PIXIU/data/Oxford_pets/jpg"

with open(json_file_path, 'r') as f:
    data = json.load(f)

renamed_files = set()

for item in data['train']:
    original_filename = item[0].strip('\\')

    # 从 JSON 中获取标签（数字）
    label = str(item[2])

    # 从原始文件名中提取类别名称和数字编号
    # 例如：english_cocker_spaniel_172.jpg -> english_cocker_spaniel, 172
    filename_without_ext = os.path.splitext(original_filename)[0]  # 去掉 .jpg 扩展名
    last_underscore_idx = filename_without_ext.rfind('_')  # 找到最后一个下划线的位置

    if last_underscore_idx != -1:
        category_name = filename_without_ext[:last_underscore_idx]  # 类别名称
        number = filename_without_ext[last_underscore_idx + 1:]  # 数字编号

        # 新文件名格式：{类别名称}_{数字编号}.jpg
        new_filename = f"{category_name}_{number}.jpg"

        original_filepath = os.path.join(image_dir, original_filename)
        new_filepath = os.path.join(image_dir, new_filename)

        if os.path.exists(original_filepath):
            os.rename(original_filepath, new_filepath)
            renamed_files.add(new_filepath)
            print(f"Renamed: {original_filepath} -> {new_filepath}")
        else:
            print(f"File not found: {original_filepath}")
    else:
        print(f"Invalid filename format: {original_filename}")

for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)
    if filepath not in renamed_files and filename.startswith('image_'):
        os.remove(filepath)
        print(f"Deleted: {filepath}")

