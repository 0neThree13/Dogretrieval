#使用clip处理数据库中图片与文本
import os
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# 路径设置
IMAGE_DIR = "DogUI/static/data"
CAPTION_CSV = "DogUI/static/captions.csv"

SAVE_IMAGE_EMB = "DogUI/static/image_embeddings.npy"
SAVE_TEXT_EMB = "DogUI/static/image_captions.npy"


class Args:
    #Training arguments
    lr = 5e-4
    n_iters = 10
    batch_size = 256
    backbone= 'ViT-B/16'
    dataset = "Dog100K"

    # LoRA arguments
    position = 'all'  # 位置参数，可选：'bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'
    encoder = 'both'  # 'text', 'vision', 'both'
    params = ['q', 'k', 'v']  # 注意力矩阵位置
    r = 8  # 低秩矩阵的秩
    alpha = 1  # 缩放系数
    dropout_rate = 0.25  # LoRA dropout

    save_path = 'Dogretrieval/Dogclip'  # 保存路径
    lora_path = "DogUI/static/lora_weights.pt"
    filename = 'lora_weights'
    eval_only = False  # 是否只评估 LoRA 模块


# 使用方式：
args = Args()
print(args.lr)  # 访问学习率
print(args.params)  # 访问注意力矩阵位置


from Dogclip.loralib import *

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# 注入 LoRA 层
list_lora_layers = apply_lora(args, model)

# 加载 LoRA 权重
load_lora(args, list_lora_layers)

# 将所有 LoRA 层迁移到相同设备
for layer in list_lora_layers:
    layer.to(device)



# 加载 caption 表格
df = pd.read_csv(CAPTION_CSV)

image_embeddings = []
text_embeddings = []


for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["filename"]
    caption = row["caption"]

    img_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(img_path):
        print(f"跳过未找到的图片: {filename}")
        continue

    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"跳过无法处理的图片: {filename}")
        continue

    text = clip.tokenize([caption], truncate=True).to(device)

    with torch.no_grad():
        image_emb = model.encode_image(image)
        text_emb = model.encode_text(text)

        # L2 归一化
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    image_embeddings.append(image_emb.cpu().numpy())
    text_embeddings.append(text_emb.cpu().numpy())


# 保存所有数据
np.save(SAVE_IMAGE_EMB, np.concatenate(image_embeddings, axis=0))  # [N, 512]
np.save(SAVE_TEXT_EMB, np.concatenate(text_embeddings, axis=0))    # [N, 512]
