import cv2
import os
import random
from PIL import Image, ImageDraw

# 載入 OpenCV 的人臉偵測模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 路徑設定
input_image_path = "photo.jpg"  # 原始圖片路徑
cat_faces_dir = "cat_faces"  # 貓臉素材庫資料夾
output_image_path = "output.jpg"  # 輸出圖片路徑

# 載入貓臉素材庫
cat_faces = [os.path.join(cat_faces_dir, f) for f in os.listdir(cat_faces_dir) if f.endswith(".png")]

# 確認素材庫是否有貓咪圖片
if not cat_faces:
    raise ValueError("貓咪素材庫為空，請在資料夾中放入 PNG 圖片！")

# 讀取背景圖片
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 偵測人臉
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 將背景圖片轉為 RGBA 格式
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")

# 替換每一個人臉
for (x, y, w, h) in faces:
    # 隨機選擇一張貓臉
    cat_face_path = random.choice(cat_faces)
    cat_face = Image.open(cat_face_path).convert("RGBA")

    # 計算人臉中心點和新的尺寸（放大 30%）
    face_center_x, face_center_y = x + w // 2, y + h // 2
    new_w, new_h = int(w * 1.5), int(h * 1.5)

    # 調整貓臉大小
    resized_cat = cat_face.resize((new_w, new_h), Image.LANCZOS)

    # 計算貓臉放置位置（讓貓臉中心與人臉中心對齊）
    top_left_x = face_center_x - new_w // 2
    top_left_y = face_center_y - new_h // 2

    # 提取透明遮罩
    cat_mask = resized_cat.split()[3]  # 第 3 通道是透明度

    # 將貓臉貼到背景圖上
    image_pil.paste(resized_cat, (top_left_x, top_left_y), cat_mask)

# 儲存結果，使用支援透明度的格式（如 PNG）
image_pil.save(output_image_path, "PNG")
print(f"處理完成！結果已儲存至：{output_image_path}")
