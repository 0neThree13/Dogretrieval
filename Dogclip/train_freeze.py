# 冻结参数微调
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Dog100K import *
import clip_pretrained
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_clip_layers(model, freeze_until=8):
    for name, param in model.named_parameters():
        # 冻结视觉分支
        if name.startswith("visual.transformer.resblocks"):
            try:
                block_id = int(name.split('.')[3])  # visual.transformer.resblocks.0....
                param.requires_grad = block_id >= freeze_until
            except:
                param.requires_grad = True  # fallback
        # 冻结文本分支
        elif name.startswith("transformer.resblocks"):
            try:
                block_id = int(name.split('.')[2])  # transformer.resblocks.0....
                param.requires_grad = block_id >= freeze_until
            except:
                param.requires_grad = True  # fallback
        else:
            param.requires_grad = True


def evaluate(clip_model, loader, device='cuda'):
    clip_model.eval()
    clip_model = clip_model.to(device)

    recall_at_1 = 0
    recall_at_5 = 0
    mrr = 0
    total = 0
    all_scores = []

    with torch.no_grad():
        for images, captions in tqdm(loader):
            images = images.to(device)

            image_features = clip_model.encode_image(images)
            text_tokens = clip.tokenize(captions).cuda()
            text_features = clip_model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Similarity
            sim = image_features @ text_features.T  # [B, B]
            scores = sim.diag()
            all_scores.extend(scores.cpu().tolist())
            avg_score = sum(all_scores) / len(all_scores)

            B = sim.size(0)
            ranks = sim.argsort(dim=1, descending=True)
            correct = torch.arange(B, device=device)

            for i in range(B):
                rank = (ranks[i] == correct[i]).nonzero(as_tuple=True)[0].item()
                if rank == 0:
                    recall_at_1 += 1
                if rank < 5:
                    recall_at_5 += 1
                mrr += 1.0 / (rank + 1)
                total += 1

    return recall_at_1 / total, recall_at_5 / total, mrr / total, avg_score


def train_one_epoch_amp(model, loader, optimizer, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    # tqdm 进度条
    loop = tqdm(loader, desc="Training", leave=True)

    for images, captions in loop:
        images = images.to(device)
        texts = clip.tokenize(captions).to(device)

        with torch.cuda.amp.autocast():
            # 编码图像和文本
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # L2 归一化
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # 相似度矩阵
            logits_per_image = model.logit_scale.exp() * image_features @ text_features.T
            labels = torch.arange(len(images), device=device)

            # 双向对比损失
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_image.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # tqdm 实时更新 loss
        loop.set_postfix(loss=loss.item())

    return total_loss / total_samples


def run_training():
    batch_size = 256
    num_epochs = 10
    seed = 42
    save_dir = 'output'
    os.makedirs(save_dir, exist_ok=True)
    root_dir = 'Dogretrieval/Dog100K'
    img_dataset = 'data'

    writer = SummaryWriter(log_dir="tf-logs")

    train_set = Dog100K(root=root_dir, subset='train', img_dataset=img_dataset, seed=seed)
    test_set = Dog100K(root=root_dir, subset='test', img_dataset=img_dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)

    # 加载 CLIP 模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    freeze_clip_layers(model, freeze_until=8)
    model = model.to(device)

    trainable_params = filter(lambda p: p.requires_grad and p.dtype == torch.float32, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    scaler = torch.cuda.amp.GradScaler()
    best_sim = 0.0  # 使用匹配相似度作为保存标准

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch_amp(model, train_loader, optimizer, device, scheduler, scaler)

        recall_at_1, recall_at_5, mrr, sim = evaluate(model, test_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Score/Recall@1', recall_at_1, epoch)
        writer.add_scalar('Score/Recall@5', recall_at_5, epoch)
        writer.add_scalar('Score/MRR', mrr, epoch)
        writer.add_scalar('Score/Match similarity', sim, epoch)

        print(f"Train Loss:        {train_loss:.4f}")
        print(f"Match similarity:  {sim:.4f}")
        print(f"Recall@1:          {recall_at_1:.4f}")
        print(f"Recall@5:          {recall_at_5:.4f}")
        print(f"MRR:               {mrr:.4f}")

        if sim > best_sim:
            best_sim = sim
            torch.save({'model_state_dict': model.state_dict()},
                       os.path.join(save_dir, f"model_best.pth"))
            print(f"Model saved with match similarity = {sim:.4f}")

    writer.close()


if __name__ == "__main__":
    run_training()