from Dogclip.utils import *
from Dogclip.loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora, load_lora
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import clip

class Args:
    #Training arguments
    lr = 2e-4
    n_iters = 500
    batch_size = 32
    backbone= 'ViT-B/16'
    dataset = "Dog100K"

    # LoRA arguments
    position = 'all'  # 位置参数，可选：'bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'
    encoder = 'both'  # 'text', 'vision', 'both'
    params = ['q', 'k', 'v']  # 注意力矩阵位置
    r = 2  # 低秩矩阵的秩
    alpha = 1  # 缩放系数
    dropout_rate = 0.25  # LoRA dropout
    lora_path = "lora_weights.pt"

    save_path = None  # 保存路径
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


def run_lora(args, clip_model, logit_scale, train_loader, test_loader):


    writer = SummaryWriter(log_dir="runs")

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        recall_at_1, recall_at_5, mrr, avg_score = evaluate_lora(clip_model, test_loader)
        print("**** Evaluation Results ****")
        print(f"Recall@1: {recall_at_1:.4f}, Recall@5: {recall_at_5:.4f}, MRR: {mrr:.4f}, Avg Score: {avg_score:.4f}")
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    count_iters = 0
    while count_iters < total_iters:
        clip_model.train()
        loss_epoch = 0
        tot_samples = 0

        for images, captions in tqdm(train_loader, desc=f"Epoch Iter {count_iters}/{total_iters}"):
            images = images.cuda()
            texts = clip.tokenize(captions).cuda()

            with torch.cuda.amp.autocast():
                image_features = clip_model.encode_image(images)
                text_features = clip_model.encode_text(texts)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits_per_image = logit_scale * image_features @ text_features.t()
                labels = torch.arange(len(images)).cuda()
                loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_image.T, labels)) / 2

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_epoch += loss.item() * len(images)
            tot_samples += len(images)
            count_iters += 1

            if count_iters >= total_iters:
                break

        # Evaluation
        recall_at_1, recall_at_5, mrr, avg_score = evaluate_lora(clip_model, test_loader)
        clip_model.eval()

        writer.add_scalar('Loss/train', loss_epoch / tot_samples, count_iters)
        writer.add_scalar('Score/Recall@1', recall_at_1, count_iters)
        writer.add_scalar('Score/Recall@5', recall_at_5, count_iters)
        writer.add_scalar('Score/MRR', mrr, count_iters)
        writer.add_scalar('Score/Match similarity', avg_score, count_iters)

        print(f"Iter {count_iters}/{total_iters} | Loss: {loss_epoch / tot_samples:.4f} | "
              f"Recall@1: {recall_at_1:.4f}, Recall@5: {recall_at_5:.4f}, MRR: {mrr:.4f}, Score: {avg_score:.4f}")

    writer.close()

    print("**** Final retrieval Recall@1: {:.4f} ****\n".format(recall_at_1))

    if args.save_path is not None:
        save_lora(args, list_lora_layers)