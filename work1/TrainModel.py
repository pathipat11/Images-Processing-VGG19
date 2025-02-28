from keras.applications import VGG19
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# load model VGG19 (Pretrained)
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Fine-tune 4 Layers back
for layer in base_model.layers[:15]:
    layer.trainable = False

# add Fully Connected Layer new
x = Flatten()(base_model.output)
x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
x = Dropout(0.4)(x)  # ลด Overfitting
output = Dense(1, activation='sigmoid')(x)  # 2 Classes → Sigmoid

# build model
model = Model(inputs=base_model.input, outputs=output)

# use Adam Optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation for Train Set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# load Train Set
train_it = train_datagen.flow_from_directory(
    "/content/TrainData/train/",
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# load Test Set
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_it = test_datagen.flow_from_directory(
    "/content/TrainData/test/",
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False  # test not shuffle
)

# Train Model (not have Validation)
model.fit(train_it, epochs=20, verbose=1)

# Test model with Set
loss, acc = model.evaluate(test_it, verbose=1)
print(f"Test Accuracy: {acc*100:.2f}%")

# save model
model.save("model.keras")
