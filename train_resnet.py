# train_resnet.py
import os, json, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from data_finder import find_data_root

DATA_ROOT = find_data_root([
    r"C:\Users\HP\Desktop\yes\fruits-360",
    r"C:\Users\HP\Desktop\fruits-360",
    r"C:\Users\HP\Desktop",
    Path.cwd(),
])
if DATA_ROOT is None:
    raise FileNotFoundError("Could not find 'Training' and 'Test'.")
TRAIN_DIR = Path(DATA_ROOT) / "Training"
TEST_DIR  = Path(DATA_ROOT) / "Test"
print("Using DATA_ROOT:", DATA_ROOT)

all_classes = sorted([d for d in os.listdir(TRAIN_DIR) if (TRAIN_DIR / d).is_dir()])
classes = all_classes[:10]
print("Classes:", classes)

# 224x224 is typical for ResNet
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

train_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.2, horizontal_flip=True
)
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, classes=classes,
    batch_size=BATCH, subset='training', shuffle=True, seed=SEED
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, classes=classes,
    batch_size=BATCH, subset='validation', shuffle=False
)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, classes=classes,
    batch_size=BATCH, shuffle=False
)

base = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
for l in base.layers: l.trainable = False

inp = layers.Input(shape=IMG_SIZE+(3,))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(len(classes), activation='softmax')(x)
model = models.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=6)

# Fine-tune top layers (skip BatchNorm)
for l in base.layers[-30:]:
    if not isinstance(l, layers.BatchNormalization):
        l.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=8)

# Evaluate + save
y_true = test_gen.classes
probs = model.predict(test_gen); y_pred = np.argmax(probs, axis=1)
print("Test accuracy:", (y_true == y_pred).mean())
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
cm = confusion_matrix(y_true, y_pred)

os.makedirs("outputs", exist_ok=True)
sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - ResNet50"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
plt.savefig("outputs/confusion_resnet.png"); plt.close()

os.makedirs("models", exist_ok=True)
model.save("models/fruit_resnet50_ft.keras")
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
with open("class_indices.json", "w") as f:
    json.dump(idx_to_class, f)

print("Saved: models/fruit_resnet50_ft.keras, class_indices.json and outputs/confusion_resnet.png")