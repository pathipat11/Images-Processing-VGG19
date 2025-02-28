from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("model.keras")

img_path = "/content/download.jpg"
img = load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.axis("off")
plt.show()

img = img_to_array(img)
img = img.reshape(1, 224, 224, 3)
img = img.astype("float32") / 255.0

result = model.predict(img)

if result[0] >= 0.5:
    print("✅ Predict: Rambutan 🍒")
else:
    print("✅ Predict: Durian 🍈")
