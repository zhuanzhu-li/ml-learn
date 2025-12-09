

import torch
import clip
from PIL import Image

# 加载模型和预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)  # 使用 ViT-B/32 模型

# 准备图像和文本候选
image = preprocess(Image.open("D:\\download\\c55a6a540166dda33e107983ac668db1.jpg")).unsqueeze(0).to(device)  # 你的图像
text_descriptions = ["a photo of a cat", "a photo of a dog", "a car on the street"]
text_tokens = clip.tokenize(text_descriptions).to(device)

# 计算特征向量
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

# 计算相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (image_features @ text_features.T).softmax(dim=-1)  # 得到每个文本的匹配概率

# 获取最匹配的文本
values, indices = similarity[0].topk(1)
best_match = text_descriptions[indices[0]]
print(f"最匹配的描述是: '{best_match}'，相似度得分: {values[0]:.4f}")