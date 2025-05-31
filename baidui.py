#从百度识图检索互联网🛜相似图片
import requests
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml import etree


image_path = 'DogUI/static/dog.jpeg'
save_dir = 'downloads'
os.makedirs(save_dir, exist_ok=True)

upload_url = "https://graph.baidu.com/upload"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://graph.baidu.com/"
}


with open(image_path, "rb") as f:
    files = {"image": f}
    res = requests.post(upload_url, headers=headers, files=files)
    res_json = res.json()
    page_url = res_json['data']['url']
    print(f"[✔] 上传成功，页面 URL：{page_url}")


resp = requests.get(page_url, headers=headers)
tree = etree.HTML(resp.text)

detail_urls = tree.xpath("//div[@class='img-item']/@data-obj-url")
print(f"[✔] 共找到 {len(detail_urls)} 个详情页")


chrome_options = Options()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(options=chrome_options)

for idx, detail_url in enumerate(detail_urls):
    driver.get(detail_url)
    time.sleep(1)
    html = driver.page_source
    tree = etree.HTML(html)
    image_urls = tree.xpath("//img[@preview='usemap']/@src")

    if not image_urls:
        print(f"[×] 第{idx + 1}张相似图未找到图片 URL")
        continue

    img_url = image_urls[0].replace('&amp;', '&')

    try:
        img_resp = requests.get(img_url, headers=headers)
        img_resp.raise_for_status()
    except Exception as e:
        print(f"[×] 下载第{idx + 1}张图片失败：{e}")
        continue

    img_name = f'dog_similar_{idx + 1}.jpg'
    img_path = os.path.join(save_dir, img_name)

    with open(img_path, 'wb') as f:
        f.write(img_resp.content)

    print(f"成功下载：{img_path}")

driver.quit()