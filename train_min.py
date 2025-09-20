# train_min.py
import os, json, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from data_finder import find_data_root

# 1) Locate dataset root that contains 'Training' and 'Test'
DATA_ROOT = find_data_root([
    r"C:\Users\HP\Desktop\yes\fruits-360",
    r"C:\Users\HP\Desktop\fruits-360",
    r"C:\Users\HP\Desktop",
    Path.cwd(),
])
if DATA_ROOT is None:
    raise FileNotFoundError("Could not find 'Training' and 'Test'. Make sure the dataset is extracted.")
TRAIN_DIR = Path(DATA_ROOT) / "Training"
TEST_DIR  = Path(DATA_ROOT) / "Test"
print("Using DATA_ROOT:", DATA_ROOT)

# 2) Pick 10 classes (you can manually list instead)
all_classes = sorted([d for d in os.listdir(TRAIN_DIR) if (TRAIN_DIR / d).is_dir()])
classes = all_classes[:10]
print("Classes:", classes)

# 3) Data generators (100x100 for fast training)
IMG_SIZE = (100, 100)
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

# 4) Simple CNN (baseline)
def build_cnn(input_shape=(100,100,3), num_classes=10):
    i = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(i); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3, activation='relu', padding='same')(x); x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(i, o)

model = build_cnn(IMG_SIZE+(3,), len(classes))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

# 5) Evaluate on test set
y_true = test_gen.classes
probs = model.predict(test_gen)
y_pred = np.argmax(probs, axis=1)
print("Test accuracy:", (y_true == y_pred).mean())
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
cm = confusion_matrix(y_true, y_pred)

# Save visuals
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - CNN"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
plt.savefig("outputs/confusion_cnn.png"); plt.close()

# Training curves
import matplotlib.pyplot as plt
plt.figure(); plt.plot(history.history['accuracy'], label='train'); plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend(); plt.tight_layout(); plt.savefig("outputs/acc_cnn.png"); plt.close()
plt.figure(); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend(); plt.tight_layout(); plt.savefig("outputs/loss_cnn.png"); plt.close()

# 6) Save model + label map
os.makedirs("models", exist_ok=True)
model.save("models/fruit_cnn.keras")
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
with open("class_indices.json", "w") as f:
    json.dump(idx_to_class, f)

print("Saved: models/fruit_cnn.keras, class_indices.json and outputs/*.png")