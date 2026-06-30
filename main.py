import torch
import cv2
import numpy as np
from model.u2net import U2NET
import os

# ====== 你的函数 ======
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess(image):
    image = cv2.resize(image, (320, 320))
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)

    return image


# ====== 主程序 ======
device = torch.device("cpu")

model = U2NET(3, 1)
model.load_state_dict(torch.load("u2net.pth", map_location=device))
model.to(device)
model.eval()

image_folder = "data/images"
output_folder = "outputs"

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(image_folder):

    if not img_name.endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_folder, img_name)

    # 1. 读图
    image = load_image(img_path)

    # 2. 预处理
    tensor = preprocess(image).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(tensor)

    saliency = output[0]

    # 4. 后处理
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    mask = (saliency * 255).astype(np.uint8)

    # 5. 保存
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, mask)

    print(f"saved: {save_path}")