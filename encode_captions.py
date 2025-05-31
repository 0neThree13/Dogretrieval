#使用基于bert的Sentence-BERT 模型实现本地文本检索
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载 caption.csv
df = pd.read_csv("Dogbert/captions.csv")

# 加载 Sentence-BERT 模型
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# 获取所有 captions
captions = df["caption"].tolist()

# 编码为向量
embeddings = model.encode(captions, batch_size=32, show_progress_bar=True)

# 保存向量到 .npy
np.save("Dogbert/caption_embeddings.npy", embeddings)

# 可选：保存 filename 列，方便对应回图片
df["filename"].to_csv("filenames.txt", index=False, header=False)

print(f"已保存 {len(embeddings)} 条 caption 向量。")