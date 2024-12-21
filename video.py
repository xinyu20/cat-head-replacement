import cv2
import os
import random
import numpy as np
from PIL import Image

# 載入 OpenCV 的人臉偵測模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 路徑設定
input_video_path = "input_video.mp4"  # 原始影片路徑
cat_faces_dir = "clown"  # 貓臉素材庫資料夾
output_video_path = "output_video.mp4"  # 輸出影片路徑

# 載入貓臉素材庫
cat_faces = [os.path.join(cat_faces_dir, f) for f in os.listdir(cat_faces_dir) if f.endswith(".png")]

# 確認素材庫是否有貓咪圖片
if not cat_faces:
    raise ValueError("貓咪素材庫為空，請在資料夾中放入 PNG 圖片！")

# 讀取影片
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"無法讀取影片檔案：{input_video_path}")

# 獲取影片資訊
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 設定輸出影片的編碼與參數
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 處理每一幀
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 影片讀取結束

    frame_count += 1
    print(f"處理中：第 {frame_count}/{total_frames} 幀")

    # 將影格轉為灰階進行人臉偵測
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 將影格轉為 PIL 格式以便貼合貓臉
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # 替換每一個人臉
    for (x, y, w, h) in faces:
        # 隨機選擇一張貓臉
        cat_face_path = random.choice(cat_faces)
        cat_face = Image.open(cat_face_path).convert("RGBA")

        # 計算人臉中心點和新的尺寸（放大 50%）
        face_center_x, face_center_y = x + w // 2, y + h // 2
        new_w, new_h = int(w * 1.5), int(h * 1.5)

        # 調整貓臉大小
        resized_cat = cat_face.resize((new_w, new_h), Image.LANCZOS)

        # 計算貓臉放置位置（讓貓臉中心與人臉中心對齊）
        top_left_x = face_center_x - new_w // 2
        top_left_y = face_center_y - new_h // 2

        # 提取透明遮罩
        cat_mask = resized_cat.split()[3]  # 第 3 通道是透明度

        # 將貓臉貼到影格上
        frame_pil.paste(resized_cat, (top_left_x, top_left_y), cat_mask)

    # 將處理後的影格轉回 OpenCV 格式
    frame_processed = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)

    # 寫入輸出影片
    out.write(frame_processed)

# 釋放資源
cap.release()
out.release()

print(f"影片處理完成！結果已儲存至：{output_video_path}")
