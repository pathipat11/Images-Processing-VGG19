from keras.applications import VGG19
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# โหลดโมเดล VGG19 (Pretrained)
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Fine-tune 4 Layers
for layer in base_model.layers[:15]:
    layer.trainable = False

# เพิ่ม Fully Connected Layer ใหม่
x = Flatten()(base_model.output)
x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
x = Dropout(0.4)(x)  # ลด Overfitting
output = Dense(4, activation='softmax')(x)  # 4 Classes → Softmax

# สร้างโมเดล
model = Model(inputs=base_model.input, outputs=output)

# ใช้ Adam Optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation สำหรับ Train Set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# โหลด Train Set
train_it = train_datagen.flow_from_directory(
    "/content/TrainData-for-palm-edit/train",
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'  # เปลี่ยนเป็น categorical
)

# โหลด Test Set
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_it = test_datagen.flow_from_directory(
    "/content/TrainData-for-palm-edit/test",
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',  # เปลี่ยนเป็น categorical
    shuffle=False
)

# Train Model (ไม่ใช้ Validation)
model.fit(train_it, epochs=50, verbose=1)

# ทดสอบโมเดล
loss, acc = model.evaluate(test_it, verbose=1)
print(f"Test Accuracy: {acc*100:.2f}%")

# บันทึกโมเดล
model.save("model-edit.keras")
