import torch
from torch.utils.data import DataLoader
import clip
from Dogclip.lora import apply_lora, load_lora
from Dogclip.Dog100K import Dog100K, collate_fn


logit_scale = 100
root_dir = 'Dogretrieval/DogUI/static'
img_dataset = 'pre_data'
seed = 42

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
    lora_path = "Dogretrieval/Dogclip/vitb16/Dog100K/lora_weights.pt"
    filename = 'lora_weights'  # 保存文件名（会自动加 .pt）

    eval_only = False  # 是否只评估 LoRA 模块


# 使用方式：
args = Args()
print(args.lr)  # 访问学习率
print(args.params)  # 访问注意力矩阵位置


def evaluate_lora(clip_model, loader):
    clip_model.eval()
    all_scores = []
    recall_at_1 = 0
    recall_at_5 = 0
    mrr = 0
    total = 0

    with torch.no_grad():
        for i, (images, captions) in enumerate(loader):
            images = images.cuda()
            text_tokens = clip.tokenize(captions).cuda()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
                text_features = clip_model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Similarity scores
            sim = image_features @ text_features.T  # shape: [B, B]

            # 计算匹配分数（对角线）
            scores = sim.diag()
            all_scores.extend(scores.cpu().tolist())

            B = sim.size(0)
            total += B

            # Rank each row
            ranks = sim.argsort(dim=1, descending=True)  # shape: [B, B]
            correct = torch.arange(B, device=sim.device)

            for i in range(B):
                rank = (ranks[i] == correct[i]).nonzero(as_tuple=True)[0].item()
                if rank == 0:
                    recall_at_1 += 1
                if rank < 5:
                    recall_at_5 += 1
                mrr += 1.0 / (rank + 1)

    avg_score = sum(all_scores) / len(all_scores)

    recall1 = recall_at_1 / total
    recall5 = recall_at_5 / total
    mrr_value = mrr / total

    print(f"Average matching score: {avg_score:.4f}")
    print(f"Recall@1: {recall1:.4f} | Recall@5: {recall5:.4f} | MRR: {mrr_value:.4f}")

    return recall1, recall5, mrr_value, avg_score



# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载原始 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载保存的参数
checkpoint = torch.load("output/model_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
model.eval()

train_set = Dog100K(root=root_dir, subset='train', img_dataset=img_dataset, seed=seed)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=collate_fn)


with torch.no_grad():
    recall_at_1, recall_at_5, mrr, avg_score = evaluate_lora(model, train_loader)


print(f"Recall@1: {recall_at_1:.4f}, Recall@5: {recall_at_5:.4f}, MRR: {mrr:.4f}, Score: {avg_score:.4f}")


#clip冻结参数微调
# Average matching score: 0.2841
# Recall@1: 0.1954, Recall@5: 0.4789, MRR: 0.3286, Score: 0.2841
