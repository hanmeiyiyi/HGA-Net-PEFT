# 把阈值改成 0.6 修改后
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, ViTModel
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile
import time

from dhg import Hypergraph
from dhg.models import HGNN
import torch.nn.init as init

bert_base_path = "bert-base-uncased"
default_label = 0

def load_data(label_file):
    images = []
    texts = []
    labels = []
    skipped_count = 0
    not_found_count = 0
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 3:
                print(f"Skipping incomplete line: {line}")
                continue
            image_path, text_description, label = parts
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}, skipping.")
                not_found_count += 1
                continue
            if not is_image_file(image_path):
                print(f"File is not an image: {image_path}, skipping.")
                skipped_count += 1
                continue
            try:
                label = int(label)
            except ValueError:
                print(f"Invalid label for image {image_path}, using default label.")
                label = default_label
            images.append(image_path)
            texts.append(text_description)
            labels.append(label)
    print(f"Loaded {len(images)} images, {len(texts)} texts, and {len(labels)} labels.")
    print(f"Skipped {skipped_count} non-image files, {not_found_count} files not found.")
    return images, texts, labels

def is_image_file(filepath):
    """Check if the file is an image file"""
    image_formats = ("jpg", "jpeg", "png", "bmp", "gif")
    _, ext = os.path.splitext(filepath)
    ext = ext[1:].lower()
    # 只检查扩展名，不检查文件内容（避免读取文件导致的性能问题）
    if ext in image_formats:
        return True
    return False

class CustomDataset(Dataset):
    def __init__(self, label_file, default_label=0):
        self.image_paths, self.texts, self.labels = load_data(label_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def preprocess_texts(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        text = self.texts[idx]
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = self.process_image(image_path)
        return text, image, torch.tensor(label)

    def __len__(self):
        return len(self.image_paths)

def freeze_parameters(model):
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.visual_encoder.parameters():
        param.requires_grad = False

def create_hypergraph_based_on_similarity(texts, images, model, dataset):
    text_inputs = dataset.preprocess_texts(texts).to(device)
    images = images.to(device)
    with torch.no_grad():
        text_features = model.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        visual_features = model.visual_encoder(pixel_values=images).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)

    similarity_matrix = cosine_similarity(combined_features.cpu().numpy())

    num_nodes = len(combined_features)
    hyperedges = []
    threshold = 0.6

    for i in range(num_nodes):
        hyperedges.append([i])

    for i in range(num_nodes):
        nodes_in_hyperedge = [i]
        for j in range(num_nodes):
            if i != j and similarity_matrix[i][j] > threshold:
                nodes_in_hyperedge.append(j)
        if len(nodes_in_hyperedge) > 1:
            hyperedges.append(nodes_in_hyperedge)

    hg = Hypergraph(num_nodes, hyperedges)
    if torch.cuda.is_available():
        hg = hg.to(device)

    return hg, combined_features

class GraphAdapter(nn.Module):
    def __init__(self, num_classes):
        super(GraphAdapter, self).__init__()
        self.text_encoder = BertModel.from_pretrained(bert_base_path)
        self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.visual_encoder.pooler = nn.Identity()

    def forward(self, text_inputs, image_inputs):
        text_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        visual_features = self.visual_encoder(pixel_values=image_inputs).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)
        return combined_features

class AdapterHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(AdapterHGNN, self).__init__()
        self.down = nn.Linear(in_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, in_dim)
        self.hgnn = HGNN(
            in_channels=hidden_dim,
            hid_channels=hidden_dim,
            num_classes=hidden_dim
        )
        self.classifier = nn.Linear(in_dim, num_classes)

        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.down.bias)
        nn.init.kaiming_normal_(self.up.weight, nonlinearity='relu')
        nn.init.zeros_(self.up.bias)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, combined_features, H):
        down_features = self.down(combined_features)
        hgnn_features = self.hgnn(down_features, H)
        up_features = self.up(hgnn_features)
        enhanced_features = combined_features + up_features
        outputs = self.classifier(enhanced_features)
        return outputs

def evaluate(model, dataloader, adapter_hgnn):
    model.eval()
    adapter_hgnn.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for texts, images, labels in dataloader:
            texts = [text for text in texts]
            text_inputs = dataloader.dataset.preprocess_texts(texts).to(device)
            labels = labels.to(torch.long).to(device)
            images = images.to(device)

            features = model(text_inputs, images)

            batch_size = features.size(0)
            H = Hypergraph(batch_size, [[i] for i in range(batch_size)])
            if torch.cuda.is_available():
                H = H.to(device)

            outputs = adapter_hgnn(features, H)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def log_experiment_results(epoch, loss, accuracy, filepath='experiment_results.csv'):
    with open(filepath, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss.item(), accuracy])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_gpu_memory_usage():
    """获取当前 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved
    return 0, 0

def measure_inference_time_and_memory(model, adapter_hgnn, dataloader, num_batches=10):
    """测量推理时间和内存使用"""
    print("测量推理时间和内存使用...")

    model.eval()
    adapter_hgnn.eval()

    times = []
    memory_readings = []

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, (texts, images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)
            texts = [text for text in texts]
            text_inputs = dataloader.dataset.preprocess_texts(texts).to(device)

            # 记录初始内存
            initial_allocated, initial_reserved = get_gpu_memory_usage()

            starter.record()

            features = model(text_inputs, images)

            batch_size = features.size(0)
            H = Hypergraph(batch_size, [[i] for i in range(batch_size)])
            if torch.cuda.is_available():
                H = H.to(device)

            outputs = adapter_hgnn(features, H)

            ender.record()
            torch.cuda.synchronize()

            # 记录峰值内存
            peak_allocated, peak_reserved = get_gpu_memory_usage()

            curr_time = starter.elapsed_time(ender)
            batch_size = len(texts)
            per_sample_time = curr_time / batch_size

            times.append(per_sample_time)

            # 记录内存使用情况
            memory_readings.append({
                'initial_allocated': initial_allocated,
                'peak_allocated': peak_allocated,
                'initial_reserved': initial_reserved,
                'peak_reserved': peak_reserved,
                'used_by_model': peak_allocated - initial_allocated
            })

    if times and memory_readings:
        avg_time = np.mean(times)
        std_time = np.std(times)

        avg_memory = np.mean([r['used_by_model'] for r in memory_readings])
        max_memory = np.max([r['used_by_model'] for r in memory_readings])

        print(f"推理时间和内存统计:")
        print(f"  平均时间：{avg_time:.4f} ms")
        print(f"  时间标准差：{std_time:.4f} ms")
        print(f"  平均内存使用：{avg_memory:.0f} bytes ({avg_memory/1024/1024:.2f} MB)")
        print(f"  最大内存使用：{max_memory:.0f} bytes ({max_memory/1024/1024:.2f} MB)")

        return avg_time, std_time, avg_memory, max_memory

    return 0, 0, 0, 0

def measure_test_memory(model, adapter_hgnn, dataloader, sample_batches=3):
    """测量测试过程中的内存使用"""
    print("测量测试过程中的内存使用...")

    memory_readings = []
    model.eval()
    adapter_hgnn.eval()

    with torch.no_grad():
        for i, (texts, images, labels) in enumerate(dataloader):
            if i >= sample_batches:
                break

            images = images.to(device)
            texts = [text for text in texts]
            text_inputs = dataloader.dataset.preprocess_texts(texts).to(device)

            # 记录初始内存
            initial_allocated, initial_reserved = get_gpu_memory_usage()

            # 执行前向传播
            features = model(text_inputs, images)
            batch_size = features.size(0)
            H = Hypergraph(batch_size, [[i] for i in range(batch_size)])
            if torch.cuda.is_available():
                H = H.to(device)
            outputs = adapter_hgnn(features, H)

            # 记录峰值内存
            peak_allocated, peak_reserved = get_gpu_memory_usage()

            # 记录内存使用情况
            memory_readings.append({
                'initial_allocated': initial_allocated,
                'peak_allocated': peak_allocated,
                'initial_reserved': initial_reserved,
                'peak_reserved': peak_reserved,
                'used_by_model': peak_allocated - initial_allocated
            })

            # 清理缓存
            del H, features, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if memory_readings:
        avg_allocated = np.mean([r['used_by_model'] for r in memory_readings])
        max_allocated = np.max([r['used_by_model'] for r in memory_readings])
        print(f"测试内存使用:")
        print(f"  平均内存使用：{avg_allocated:.0f} bytes ({avg_allocated/1024/1024:.2f} MB)")
        print(f"  最大内存使用：{max_allocated:.0f} bytes ({max_allocated/1024/1024:.2f} MB)")
        return avg_allocated, max_allocated

    return 0, 0

def calculate_detailed_flops(original_model, compressed_model, adapter_hgnn, train_dataset):
    """计算详细的 FLOPs 统计"""
    print("计算详细的 FLOPs 统计...")

    # 参数数量
    original_params = count_parameters(original_model)
    compressed_params = count_parameters(compressed_model)

    # 总参数数量
    original_total_params = count_total_parameters(original_model)
    compressed_total_params = count_total_parameters(compressed_model)

    # 计算参数压缩比（使用实际参数量比值）
    param_compression_ratio = compressed_params / original_params if original_params > 0 else 0

    # 创建示例输入
    batch_size = 1
    dummy_texts = ["This is a sample text description."]
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_text_inputs = train_dataset.preprocess_texts(dummy_texts).to(device)

    with torch.no_grad():
        dummy_text_features = original_model.text_encoder(**dummy_text_inputs).last_hidden_state[:, 0, :]
        dummy_visual_features = original_model.visual_encoder(pixel_values=dummy_images).last_hidden_state[:, 0, :]
        dummy_combined_features = torch.cat([dummy_text_features, dummy_visual_features], dim=1)
        dummy_H = Hypergraph(batch_size, [[0]])
        if torch.cuda.is_available():
            dummy_H = dummy_H.to(device)

    # 分别计算各部分 FLOPs
    # 1. 原始模型（BERT + ViT）FLOPs
    original_flops, _ = profile(original_model, inputs=(dummy_text_inputs, dummy_images))

    # 2. AdapterHGNN FLOPs
    adapter_flops, _ = profile(adapter_hgnn, inputs=(dummy_combined_features, dummy_H))

    # 3. 特征提取部分 FLOPs（BERT 文本编码 + ViT 视觉编码）
    # 创建仅包含特征提取器的模型
    class FeatureExtractor(nn.Module):
        def __init__(self, text_encoder, visual_encoder):
            super(FeatureExtractor, self).__init__()
            self.text_encoder = text_encoder
            self.visual_encoder = visual_encoder

        def forward(self, text_inputs, image_inputs):
            text_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
            visual_features = self.visual_encoder(pixel_values=image_inputs).last_hidden_state[:, 0, :]
            combined_features = torch.cat([text_features, visual_features], dim=1)
            return combined_features

    feature_extractor = FeatureExtractor(original_model.text_encoder, original_model.visual_encoder)
    feature_flops, _ = profile(feature_extractor, inputs=(dummy_text_inputs, dummy_images))

    # 4. 实际推理总 FLOPs（特征提取 + AdapterHGNN）
    total_inference_flops = feature_flops + adapter_flops

    # 计算 FLOPs 压缩比（相对于原始完整模型）
    flops_compression_ratio = total_inference_flops / original_flops if original_flops > 0 else 0

    print(f"参数统计:")
    print(f"  原始模型可训练参数：{original_params}")
    print(f"  压缩模型可训练参数：{compressed_params}")
    print(f"  原始模型总参数：{original_total_params}")
    print(f"  压缩模型总参数：{compressed_total_params}")
    print(f"  参数压缩比：{param_compression_ratio:.6f}x")
    print(f"FLOPs 详细统计:")
    print(f"  BERT 模型 FLOPs: {original_flops - 17130000000}")  # ViT 约为 17.13G FLOPs
    print(f"  ViT 模型 FLOPs: {17130000000}")  # 固定值
    print(f"  原始完整模型 FLOPs: {original_flops}")
    print(f"  特征提取部分 FLOPs: {feature_flops}")
    print(f"  AdapterHGNN 部分 FLOPs: {adapter_flops}")
    print(f"  实际推理总 FLOPs: {total_inference_flops}")
    print(f"  FLOPs 压缩比：{flops_compression_ratio:.6f}x")

    return {
        'original_params': original_params,
        'compressed_params': compressed_params,
        'original_total_params': original_total_params,
        'compressed_total_params': compressed_total_params,
        'param_compression_ratio': param_compression_ratio,
        'bert_flops': original_flops - 17130000000,  # 近似计算
        'vit_flops': 17130000000,
        'original_flops': original_flops,
        'feature_flops': feature_flops,
        'adapter_flops': adapter_flops,
        'total_inference_flops': total_inference_flops,
        'flops_compression_ratio': flops_compression_ratio
    }

if not torch.cuda.is_available():
    raise RuntimeError("This code requires a GPU to run, but CUDA is not available.")
device = torch.device("cuda")
print("Using GPU for computation")

# Data paths configuration
train_label_file = 'test/word/flowers_file_list_train.txt'
test_label_file = 'test/word/flowers_file_list_test.txt'

# 创建数据集和数据加载器
train_dataset = CustomDataset(train_label_file, default_label)
test_dataset = CustomDataset(test_label_file, default_label)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 创建模型实例
model = GraphAdapter(num_classes=102).to(device)
adapter_hgnn = AdapterHGNN(768 + 768, 512, 102).to(device)

# 冻结参数
freeze_parameters(model)

# 调整学习率
optimizer = torch.optim.Adam(list(model.parameters()) + list(adapter_hgnn.parameters()), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 在训练开始前记录时间
training_start_time = time.time()
num_epochs = 50
# 在主训练循环中调用 create_hypergraph_based_on_similarity
for epoch in range(num_epochs):
    model.train()
    adapter_hgnn.train()
    running_loss = 0.0
    for i, (texts, images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        texts = [text for text in texts]
        text_inputs = train_dataset.preprocess_texts(texts).to(device)
        labels = labels.to(torch.long).to(device)
        images = images.to(device)
        H, combined_features = create_hypergraph_based_on_similarity(texts, images, model, train_dataset)
        features = model(text_inputs, images)
        outputs = adapter_hgnn(features, H)
        loss = F.cross_entropy(outputs, labels)
        total_loss = loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    epoch_loss = running_loss / len(train_dataloader)
    accuracy = evaluate(model, test_dataloader, adapter_hgnn)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    log_experiment_results(epoch + 1, total_loss, accuracy)
    scheduler.step()

# 训练结束后记录时间
training_end_time = time.time()

# 性能评估
print("\n" + "="*50)
print("性能评估报告")
print("="*50)

# 1. 参数和 FLOPs 统计
flops_stats = calculate_detailed_flops(model, adapter_hgnn, adapter_hgnn, train_dataset)

# 2. 推理时间和内存使用
inference_time, time_std, inference_memory_avg, inference_memory_max = measure_inference_time_and_memory(model, adapter_hgnn, test_dataloader, num_batches=5)

# 3. 测试内存使用
test_memory_avg, test_memory_max = measure_test_memory(model, adapter_hgnn, test_dataloader, sample_batches=3)

print("\n" + "="*50)
print("性能评估完成")
print("="*50)
