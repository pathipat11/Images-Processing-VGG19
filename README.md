# Images-Processing-VGG19

## Overview
Images-Processing-VGG19 is a project that uses **VGG19** to train and predict fruit types using **Deep Learning** on **Keras** and **TensorFlow**. The project is divided into two main sections:

1. **work1/** - Training a model to classify **Durian** and **Rambutan**.
2. **work1-edit/** - Training a model to classify **Durian**, **Flacourtia rukam**, and **Carabaoteats**.

Each folder contains code for training, testing the model, and a GUI for users to predict images of their choice.

---

## Project Structure
```
Images-Processing-VGG19/
│── work1/            # Train & Predict (Durian & Rambutan)
│   │── train-model.py   # Train Model
│   │── test-model.py    # Test Model
│   │── images/           # Images for training models or you can search for them yourself
│   │── gui.py           # GUI Program for Prediction
│
│── work1-edit/       # Train & Predict (Durian, Flacourtia rukam, Carabaoteats)
│   │── train-model-edit.py   # Train Model
│   │── test-model-edit.py    # Test Model
│   │── images/           # Images for training models or you can search for them yourself
│   │── gui-edit.py           # GUI Program for Prediction
│
│── README.md         # Project Documentation
```

---

## Installation & Requirements
### 🔧 **Dependencies**
- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- Tkinter

To install all dependencies, run:
```sh
pip install -r requirements.txt
```

---

## How to Use
### 1️⃣ **Train Model**
Run the **train-model.py** script to train a new model.
```sh
python work1/train-model.py
# Or for work1-edit
python work1-edit/train-model.py
```

### 2️⃣ **Test Model**
Run the **test-model.py** script to test the model with any image of your choice.
```sh
python work1/test-model.py
# Or for work1-edit
python work1-edit/test-model.py
```

### 3️⃣ **Run GUI**
You can use the GUI to predict fruit types from selected images.
```sh
python work1/gui.py
# Or for work1-edit
python work1-edit/gui.py
```

---

## Model Details
- **VGG19** is used as the base model (**Pretrained Model**) with **Fine-tuning** applied.
- **Image Augmentation** is used to enhance the model's performance.
- **Softmax Activation** is used for multi-class classification and **Sigmoid Activation** for binary classification.

---

## Results & Accuracy
- **work1** (Durian vs Rambutan) → Accuracy: **xx%**
- **work1-edit** (Durian, Flacourtia rukam, Carabaoteats) → Accuracy: **xx%**

*(Please update the Accuracy based on your model's training results)*

---

## Future Improvements
- Expand the dataset to improve accuracy.
- Experiment with other models like **ResNet50** or **EfficientNet**.
- Fine-tune hyperparameters like Learning Rate and Batch Size.

---

## Contributors
- ** Pathipat Mattra **

If you have any questions, feel free to contact me at Mail: pathipat.mattra@gmail.com 🙌
