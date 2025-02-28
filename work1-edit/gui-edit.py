import tkinter as tk
from tkinter import filedialog
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np
from PIL import Image, ImageTk

# โหลดโมเดลที่เทรนไว้
try:
    model = load_model("model-edit.keras")  # เปลี่ยนเป็นโมเดลใหม่
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    model = None

# คลาสของผลไม้ (ต้องเรียงตามลำดับที่ `flow_from_directory` ใช้)
class_labels = ["Carabaoteats", "Flacourtia rukam", "durian", "rambutan"]
emoji_map = {
    "rambutan": "🍒",
    "Flacourtia rukam": "🌿",
    "durian": "🍈",
    "Carabaoteats": "🥥"
}

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg"), ("PNG files", ".png"), ("JPEG files", "*.jpeg")])
    if not file_path:
        return

    global img_path
    img_path = file_path  # เก็บ path ของภาพ
    img_display = Image.open(file_path)
    img_display = img_display.resize((250, 250))
    img_display = ImageTk.PhotoImage(img_display)

    # แสดงภาพ
    image_label.config(image=img_display)
    image_label.image = img_display
    result_label.config(text="กรุณากด 'ทายผล' เพื่อพยากรณ์", font=("Arial", 16, "bold"), fg="white")

def predict_image():
    if model is None:
        result_label.config(text="ไม่สามารถโหลดโมเดลได้")
        return

    if not img_path:
        result_label.config(text="กรุณาเลือกภาพก่อน")
        return

    # โหลดภาพและพยากรณ์
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype('float32')
    img_array /= 255.0  # Normalize

    # พยากรณ์ภาพ
    predictions = model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)  # หา index ของ class ที่มีค่าความมั่นใจสูงสุด
    predicted_label = class_labels[predicted_class_index]  # ชื่อผลไม้
    confidence = predictions[predicted_class_index] * 100  # เปอร์เซ็นต์ความมั่นใจ
    emoji = emoji_map.get(predicted_label, "🍏")  # ใช้ emoji ที่ตรงกับผลไม้

    # อัปเดต UI
    result_text = f"ผลลัพธ์:\n{predicted_label} {emoji}\n({confidence:.2f}%)"
    result_label.config(text=result_text, font=("Arial", 16, "bold"), fg="lightgreen")

def clear_image():
    global img_path
    img_path = None  # รีเซ็ต path ของภาพ
    image_label.config(image="")
    result_label.config(text="ผลลัพธ์จะปรากฏที่นี่", font=("Arial", 16, "bold"), fg="white")

# สร้าง GUI
root = tk.Tk()
root.title("พยากรณ์ผลไม้ 🍒🍈")
root.geometry("600x800")
root.configure(bg="#27445D")  # สีพื้นหลัง

# ตั้งให้หน้าต่างอยู่ตรงกลางหน้าจอ
window_width, window_height = 600, 800
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# กำหนดการจัดตำแหน่งให้แน่นอนโดยใช้ grid
root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=0)
root.grid_rowconfigure(3, weight=0)
root.grid_rowconfigure(4, weight=0)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

name_label = tk.Label(root, text="พยากรณ์ผลไม้ 🍒🍈", font=("Arial", 22, "bold"), fg="white", bg="#27445D")
name_label.grid(row=0, column=0, columnspan=3, pady=40)

# สร้าง Frame สำหรับแสดงภาพ
image_frame = tk.Frame(root, bg="#FFFFFF", bd=5, relief="sunken", width=300, height=300)
image_frame.grid(row=1, column=0, columnspan=3, pady=20)
image_frame.pack_propagate(False)

placeholder = Image.new("RGB", (250, 250), (240, 240, 240))
placeholder = ImageTk.PhotoImage(placeholder)

# แสดงภาพ
image_label = tk.Label(image_frame, bg="#FFFFFF", image=placeholder)
image_label.image = placeholder
image_label.pack(expand=True)

# สร้าง Frame สำหรับปุ่ม
button_frame = tk.Frame(root, bg="#27445D")
button_frame.grid(row=3, column=0, columnspan=3, pady=10)

# ใช้ Grid เพื่อจัดตำแหน่งปุ่ม
btn_select = tk.Button(button_frame, text="เลือกภาพ 📷", command=select_image, font=("Arial", 14), width=15, relief="raised", bg="#FFA823", fg="white", bd=3, activebackground="#FFD35A")
btn_select.grid(row=1, column=0, padx=10, sticky="ew")

btn_predict = tk.Button(button_frame, text="ทายผล 🔍", command=predict_image, font=("Arial", 14), width=15, relief="raised", bg="#6499E9", fg="white", bd=3, activebackground="#9EDDFF")
btn_predict.grid(row=1, column=1, padx=10, sticky="ew")

# ปุ่มลบ
btn_clear = tk.Button(root, text="ลบรูปภาพ ❌", command=clear_image, font=("Arial", 14), width=2, relief="raised", bg="#900C3F", fg="white", bd=3, activebackground="#C70039")
btn_clear.grid(row=4, column=1, pady=10, sticky="ew")

# แสดงผลลัพธ์
result_label = tk.Label(root, text="ผลลัพธ์จะปรากฏที่นี่", font=("Arial", 16, "bold"), fg="white", bg="#27445D", justify="center")
result_label.grid(row=2, column=0, columnspan=3, pady=20)

# เริ่ม GUI
img_path = None  # ตัวแปรเก็บที่อยู่ของภาพ
root.mainloop()
