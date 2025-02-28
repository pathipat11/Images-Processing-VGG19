from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# โหลดโมเดลที่เทรนไว้
model = load_model("model-edit.keras")

# Class Labels (ต้องเรียงตามที่ใช้ตอน train)
class_labels = ["Carabaoteats 🍏", "Flacourtia rukam 🍇", "Durian 🍈", "Rambutan 🍒"]

# โหลดและพรีโปรเซสภาพ
img_path = "/content/images.jpg"
img = load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.axis("off")
plt.show()

img = img_to_array(img)
img = img.reshape(1, 224, 224, 3)
img = img.astype("float32") / 255.0  # Normalize ค่าให้อยู่ระหว่าง 0-1

# พยากรณ์ผล
result = model.predict(img)[0]  # จะได้ array ของค่าความน่าจะเป็นของแต่ละคลาส
predicted_class = np.argmax(result)  # หาคลาสที่ค่าความน่าจะเป็นสูงสุด
confidence = result[predicted_class] * 100  # เปลี่ยนเป็นเปอร์เซ็นต์

# แสดงผล
print(f"✅ Predict: {class_labels[predicted_class]} ({confidence:.2f}%)")
