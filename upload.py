import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import random
import numpy as np
import threading


# 設定貓臉素材路徑
cat_faces_dir = "cat_faces"
cat_faces = [os.path.join(cat_faces_dir, f) for f in os.listdir(cat_faces_dir) if f.endswith(".png")]

if not cat_faces:
    raise ValueError("貓咪素材庫為空，請在 cat_faces 資料夾中放入 PNG 圖片！")

# 載入 OpenCV 的人臉偵測模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 主處理函數
def process_file(file_path):
    file_ext = file_path.split('.')[-1].lower()
    if file_ext in ["jpg", "jpeg", "png"]:
        result_path = process_image(file_path)
        show_image(result_path)
        show_save_button(result_path)  # 顯示儲存按鈕
    elif file_ext in ["mp4", "avi", "mov"]:
        result_path = process_video(file_path)
        # 播放處理後的影片
        play_video_in_tkinter(result_path)
        show_save_button(result_path)  # 顯示儲存按鈕
        # messagebox.showinfo("完成", f"影片處理完成！結果儲存為：{result_path}")
    else:
        messagebox.showerror("錯誤", "不支援的檔案格式！")

# 處理圖片
def process_image(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    for (x, y, w, h) in faces:
        cat_face = Image.open(random.choice(cat_faces)).convert("RGBA")

        cat_ratio = cat_face.width / cat_face.height
        if w / h > cat_ratio:
            new_h = int(h * 1.5)
            new_w = int(new_h * cat_ratio)
        else:
            new_w = int(w * 1.5)
            new_h = int(new_w / cat_ratio)

        resized_cat = cat_face.resize((new_w, new_h), Image.LANCZOS)

        face_center_x, face_center_y = x + w // 2, y + h // 2
        top_left_x, top_left_y = face_center_x - new_w // 2, face_center_y - new_h // 2

        cat_mask = resized_cat.split()[3]
        image_pil.paste(resized_cat, (top_left_x, top_left_y), cat_mask)

    result_path = f"{os.path.splitext(file_path)[0]}_result.png"
    image_pil.save(result_path, "PNG")
    return result_path

# 處理影片
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    result_path = f"{os.path.splitext(file_path)[0]}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        for (x, y, w, h) in faces:
            cat_face = Image.open(random.choice(cat_faces)).convert("RGBA")

            cat_ratio = cat_face.width / cat_face.height
            if w / h > cat_ratio:
                new_h = int(h * 1.5)
                new_w = int(new_h * cat_ratio)
            else:
                new_w = int(w * 1.5)
                new_h = int(new_w / cat_ratio)

            resized_cat = cat_face.resize((new_w, new_h), Image.LANCZOS)

            face_center_x, face_center_y = x + w // 2, y + h // 2
            top_left_x, top_left_y = face_center_x - new_w // 2, face_center_y - new_h // 2

            cat_mask = resized_cat.split()[3]
            frame_pil.paste(resized_cat, (top_left_x, top_left_y), cat_mask)

        frame_processed = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        out.write(frame_processed)

    cap.release()
    out.release()
    return result_path

def play_video_in_tkinter(video_path):
    cap = cv2.VideoCapture(video_path)

    # 最大顯示尺寸（例如 400x400）
    max_width, max_height = 500, 400

    def update_frame():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 計算縮放比例，保持寬高比例不變
            frame_height, frame_width = frame.shape[:2]
            scale = min(max_width / frame_width, max_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # 調整大小並轉換顏色
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(img)

            # 更新影片畫面
            result_label.config(image=img_tk)
            result_label.image = img_tk
            result_label.update()

        cap.release()

    # 開始新執行緒來播放影片
    threading.Thread(target=update_frame, daemon=True).start()
# 儲存按鈕的全域變數
save_btn = None

# 顯示儲存按鈕
def show_save_button(result_path):
    global save_btn

    # 如果按鈕已經存在，先刪除
    if save_btn is not None:
        save_btn.destroy()

    # 定義儲存檔案的功能
    def save_file():
        save_path = filedialog.asksaveasfilename(
            defaultextension=os.path.splitext(result_path)[-1],
            filetypes=[("所有檔案", "*.*")]
        )
        if save_path:
            os.rename(result_path, save_path)
            messagebox.showinfo("完成", f"檔案已儲存至：{save_path}")

    # 創建新的儲存按鈕
    save_btn = tk.Button(root, text="儲存檔案", command=save_file, font=("Arial", 12))
    save_btn.pack(pady=10)



# 顯示處理後的圖片
def show_image(image_path):
    result_img = Image.open(image_path)
    result_img.thumbnail((500, 400))  # 縮放圖片以適應介面大小
    img_tk = ImageTk.PhotoImage(result_img)

    # 更新顯示區域
    result_label.config(image=img_tk)
    result_label.image = img_tk
    result_label.pack()

# 開啟檔案
def open_file():
    file_path = filedialog.askopenfilename(
        title="選擇圖片或影片",
        filetypes=[("圖片和影片", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
    )
    if file_path:
        process_file(file_path)

# 建立主介面
root = tk.Tk()
root.title("piyan")
root.geometry("500x500")

# 標題
title_label = tk.Label(root, text="上傳圖片或影片進行貓咪換臉", font=("Arial", 14))
title_label.pack(pady=20)

# 選擇檔案按鈕
upload_btn = tk.Button(root, text="選擇", command=open_file, font=("Arial", 12))
upload_btn.pack(pady=10)

# 結果顯示區域
result_label = tk.Label(root)
result_label.pack(pady=20)

# 開始主迴圈
root.mainloop()