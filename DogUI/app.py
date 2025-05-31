from zhipuai import ZhipuAI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from flask import Flask, request, jsonify, send_from_directory
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml import etree
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import clip
import torch
from PIL import Image
import io
import base64
import torchvision.transforms as transforms

from Dogclip.loralib import *


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
    filename = 'lora_weights'  # 保存文件名（会自动加 .pt）

    eval_only = False  # 是否只评估 LoRA 模块


# 使用方式：
args = Args()
print(args.lr)  # 访问学习率
print(args.params)  # 访问注意力矩阵位置


device = "cuda" if torch.cuda.is_available() else "cpu"

#模型加载
model_bert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
clip_model, preprocess = clip.load("ViT-B/16", device=device)


# 注入 LoRA 层
list_lora_layers = apply_lora(args, clip_model)

# 加载 LoRA 权重
load_lora(args, list_lora_layers)

# 将所有 LoRA 层迁移到相同设备
for layer in list_lora_layers:
    layer.to(device)





#数据加载
captions_df = pd.read_csv("DogUI/static/captions.csv")
image_embeddings = np.load("DogUI/static/image_embeddings.npy")
caption_embeddings = np.load("DogUI/static/caption_embeddings.npy")
image_captions= np.load("DogUI/static/image_captions.npy")

with open("DogUI/static/captions.csv", "r") as f:
    captions = [line.strip() for line in f]

with open("DogUI/static/filenames.txt", "r") as f:
    filenames = [line.strip() for line in f]

#GLM-4 api设置
client = ZhipuAI(api_key="fce8366b01e449bd9172b9f8e163985d.1EoenTlzdaMxJ1oW")
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
API_KEY = "71a8b807092b4f00a4972d18fc4554c5.7B84Fb4CBec31qGt"


#百度识图爬虫
UPLOAD_URL = "https://graph.baidu.com/upload"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://graph.baidu.com/"
}

SAVE_DIR='Dog100K'

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=chrome_options)



app = Flask(__name__)
CORS(app)  # 允许前端跨域访问

# 页面路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retrieval')
def retrieval():
    return render_template('retrieval.html')

@app.route('/generation')
def generation():
    return render_template('generation.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')




@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    file_type = data.get("type", "")
    prompt = data.get("prompt", "")
    base64_data = data.get("data", "")
    model = data.get("model", "").lower()

    try:
        if file_type.startswith("image/"):
            return handle_image(base64_data, prompt, model)
        elif file_type.startswith("audio/"):
            return handle_audio(base64_data, prompt, model)
        else:
            return jsonify({"reply": "❌ 不支持的文件类型"}), 400
    except Exception as e:
        return jsonify({"reply": f"❌ 处理失败: {str(e)}"}), 500



# 图像处理分发逻辑
def handle_image(base64_data, prompt, model):
    base64_str = base64_data.split(",")[1]
    img_base = "data:image/jpeg;base64," + base64_str

    if model == "glm-4":
        response = client.chat.completions.create(
            model="GLM-4V-Flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_base}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})


    elif model == "clip":

        image_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])
        image = preprocess(img).unsqueeze(0)


        with torch.no_grad():
            query_embedding = clip_model.encode_image(image)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        # image ➜ 文本
        text_sims = (torch.tensor(query_embedding) @ image_captions.T).squeeze().numpy()
        top_text_indices = text_sims.argsort()[-10:][::-1]  # 前10个最相似文本
        text_results = []
        for idx in top_text_indices:
            text_results.append({
                "caption": captions[idx],
                "score": float(text_sims[idx]),
                "filename": filenames[idx]  # 也提供图片路径
            })

        # image ➜ 图像
        image_sims = (torch.tensor(query_embedding) @ image_embeddings.T).squeeze().numpy()
        top_image_indices = image_sims.argsort()[-10:][::-1]  # 前10个最相似图像
        image_results = []
        for idx in top_image_indices:
            image_results.append({
                "filename": filenames[idx],
                "caption": captions[idx],
                "score": float(image_sims[idx])
            })

        return jsonify({
            "i2text_results": text_results,
            "i2image_results": image_results
        })


    elif model == "clip-lora":

        image_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])
        image = preprocess(img).unsqueeze(0)


        with torch.no_grad():
            query_embedding = clip_model.encode_image(image)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        # image ➜ 文本
        text_sims = (torch.tensor(query_embedding) @ image_captions.T).squeeze().numpy()
        top_text_indices = text_sims.argsort()[-10:][::-1]  # 前10个最相似文本
        text_results = []
        for idx in top_text_indices:
            text_results.append({
                "caption": captions[idx],
                "score": float(text_sims[idx]),
                "filename": filenames[idx]  # 也提供图片路径
            })

        # image ➜ 图像
        image_sims = (torch.tensor(query_embedding) @ image_embeddings.T).squeeze().numpy()
        top_image_indices = image_sims.argsort()[-10:][::-1]  # 前10个最相似图像
        image_results = []
        for idx in top_image_indices:
            image_results.append({
                "filename": filenames[idx],
                "caption": captions[idx],
                "score": float(image_sims[idx])
            })

        return jsonify({
            "i2text_results": text_results,
            "i2image_results": image_results
        })



    elif model == "blip":
        # 假设你有本地 BLIP 模型接口
        return jsonify({"reply": "✅ BLIP 图像描述模型：待接入本地模型推理"})

    else:
        return jsonify({"reply": f"⚠️ 模型 {model} 不支持图像输入"}), 400



# 音频处理分发逻辑
def handle_audio(base64_data, prompt, model):
    if model == "clap":
        return jsonify({"reply": "✅ Whisper 音频识别功能：请集成本地 Whisper 模型或 API 服务"})
    else:
        return jsonify({"reply": f"⚠️ 模型 {model} 不支持音频输入"}), 400



# 文本对话通用接口
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    model = data.get("model", "glm-4").lower()

    if model in ["glm-4", "glm-3-turbo"]:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            data = response.json()
            reply = data["choices"][0]["message"]["content"] if "choices" in data else "⚠️ 无返回结果"
            return jsonify({"reply": reply})
        except Exception as e:
            return jsonify({"reply": f"❌ 请求出错: {str(e)}"})

    elif model == "gpt-2":
        return jsonify({"reply": "✅ GPT-2：请接入本地模型（如 transformers）进行响应"})


    # 文生图模型
    elif model == "cogview-3-flash":
        try:
            response = client.images.generations(
                model="cogview-3-flash",
                prompt=user_message
            )
            image_url = response.data[0].url
            return jsonify({
                "results": [
                    {"url": image_url}
                ]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    elif model == "cogvideox-flash":
        try:
            response = client.videos.generations(
                model="cogvideox-flash",
                prompt=user_message,
                quality="speed",
                with_audio=True,
                size="1920x1080",
                fps=30,
            )

            video_id = getattr(response, "id", None) or getattr(response, "request_id", None)
            if not video_id:
                return jsonify({"error": "未获取到有效视频ID"}), 500

            print(f"请求ID: {video_id}")
            time.sleep(3)  # 防止“任务不存在”问题

            for _ in range(30):
                result = client.videos.retrieve_videos_result(id=video_id)
                print(f"当前状态: {result.task_status}")

                if result.task_status == "SUCCESS" and result.video_result:
                    url = result.video_result[0].url
                    return jsonify({
                        "results": [
                            {"url": url}
                        ]
                    })
                elif result.task_status == "FAILED":
                    return jsonify({"error": "视频生成失败"}), 500

                time.sleep(5)  # 间隔5秒查询一次

            return jsonify({"error": "视频生成超时"}), 500

        except Exception as e:
            return jsonify({"error": str(e)}), 500


    elif model == "clip":
        #用户查询向量化
        text = user_message
        text_tokens = clip.tokenize([text]).to(device)

        with torch.no_grad():
            query_embedding = clip_model.encode_text(text_tokens)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        # 文本 ➜ 文本
        text_sims = (torch.tensor(query_embedding) @ image_captions.T).squeeze().numpy()
        top_text_indices = text_sims.argsort()[-10:][::-1]  # 前10个最相似文本
        text_results = []
        for idx in top_text_indices:
            text_results.append({
                "caption":captions[idx],
                "score": float(text_sims[idx]),
                "filename": filenames[idx]  # 也提供图片路径
            })

        # 文本 ➜ 图像
        image_sims = (torch.tensor(query_embedding) @ image_embeddings.T).squeeze().numpy()
        top_image_indices = image_sims.argsort()[-10:][::-1]  # 前10个最相似图像
        image_results = []
        for idx in top_image_indices:
            image_results.append({
                "filename": filenames[idx],
                "caption": captions[idx],
                "score": float(image_sims[idx])
            })

        return jsonify({
            "text_results": text_results,
            "image_results": image_results
        })

    elif model == "clip-lora":
        # 用户查询向量化
        text = user_message
        text_tokens = clip.tokenize([text]).to(device)

        with torch.no_grad():
            query_embedding = clip_model.encode_text(text_tokens)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        # 文本 ➜ 文本
        text_sims = (torch.tensor(query_embedding) @ image_captions.T).squeeze().numpy()
        top_text_indices = text_sims.argsort()[-10:][::-1]  # 前10个最相似文本
        text_results = []
        for idx in top_text_indices:
            text_results.append({
                "caption": captions[idx],
                "score": float(text_sims[idx]),
                "filename": filenames[idx]  # 也提供图片路径
            })

        # 文本 ➜ 图像
        image_sims = (torch.tensor(query_embedding) @ image_embeddings.T).squeeze().numpy()
        top_image_indices = image_sims.argsort()[-10:][::-1]  # 前10个最相似图像
        image_results = []
        for idx in top_image_indices:
            image_results.append({
                "filename": filenames[idx],
                "caption": captions[idx],
                "score": float(image_sims[idx])
            })

        return jsonify({
            "text_results": text_results,
            "image_results": image_results
        })



    elif model == "bert":

        # 编码用户输入
        query_embedding = model_bert.encode([user_message])  # model_bert 是预加载好的模型
        similarities = cosine_similarity(query_embedding, caption_embeddings)[0]

        top_k = 10
        top_indices = similarities.argsort()[::-1][:top_k]

        filename2caption = dict(zip(captions_df['filename'], captions_df['caption']))

        results = []
        for i in top_indices:
            fname = filenames[i]
            caption = filename2caption.get(fname, "⚠️ 未找到对应描述")
            score = float(similarities[i])
            results.append({
                "filename": fname,
                "caption": caption,
                "score": score
            })

        return jsonify({"results": results})


    else:
        return jsonify({"reply": f"⚠️ 不支持的模型类型：{model}"}), 400



#网页爬虫
def get_similar_image_urls(image_path):
    """
    上传图片 -> 解析详情页 -> 用 selenium 获取大图 URL -> 返回 URL 列表
    """
    with open(image_path, "rb") as f:
        files = {"image": f}
        res = requests.post(UPLOAD_URL, headers=headers, files=files)
        res_json = res.json()
        page_url = res_json['data']['url']

    resp = requests.get(page_url, headers=headers)
    tree = etree.HTML(resp.text)
    detail_urls = tree.xpath("//div[@class='img-item']/@data-obj-url")

    image_urls_all = []

    for detail_url in detail_urls:
        driver.get(detail_url)
        time.sleep(1)  # 等待JS加载
        html = driver.page_source
        tree = etree.HTML(html)
        image_urls = tree.xpath("//img[@preview='usemap']/@src")

        if not image_urls:
            continue

        img_url = image_urls[0].replace('&amp;', '&')
        image_urls_all.append(img_url)

    return image_urls_all

#网页检索功能
@app.route('/upload_and_search', methods=['POST'])
def upload_and_search():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    tmp_path = os.path.join(SAVE_DIR, file.filename)
    file.save(tmp_path)

    try:
        urls = get_similar_image_urls(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)

    return jsonify({
        "message": f"找到 {len(urls)} 张相似图片",
        "images": urls
    })


if __name__ == '__main__':
    app.run(debug=True)



