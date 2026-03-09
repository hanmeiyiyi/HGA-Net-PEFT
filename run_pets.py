
# 把阈值改成 0.6 修改后
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
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
from sklearn.metrics.pairwise import cosine_similarity

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

if not torch.cuda.is_available():
    raise RuntimeError("This code requires a GPU to run, but CUDA is not available.")
device = torch.device("cuda")
print("Using GPU for computation")

# Data paths configuration
train_label_file = '/root/autodl-tmp/PIXIU/test/word/pets_file_list_train.txt'
test_label_file = '/root/autodl-tmp/PIXIU/test/word/pets_file_list_test.txt'

# 创建数据集和数据加载器
train_dataset = CustomDataset(train_label_file, default_label)
test_dataset = CustomDataset(test_label_file, default_label)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 创建模型实例
model = GraphAdapter(num_classes=37).to(device)
adapter_hgnn = AdapterHGNN(768 + 768, 512, 37).to(device)

# 冻结参数
freeze_parameters(model)

# 调整学习率
optimizer = torch.optim.Adam(list(model.parameters()) + list(adapter_hgnn.parameters()), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

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

print("\n训练完成!")