import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# ⚙️ ตั้งค่าพื้นฐาน
img_height, img_width = 224, 224
batch_size = 16
base_path = 'dataset'  # ⚠️ ต้องแน่ใจว่าโฟลเดอร์ dataset อยู่ในโฟลเดอร์เดียวกับไฟล์นี้

# 📦 เตรียมข้อมูล
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=15)
val_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    os.path.join(base_path, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val = val_datagen.flow_from_directory(
    os.path.join(base_path, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# 🧠 สร้างโมเดลจาก ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(train.class_indices), activation='softmax')(x)
model = Model(base_model.input, output)

# 🔒 Freeze ชั้นฐานของ ResNet50
for layer in base_model.layers:
    layer.trainable = False

# ✅ คอมไพล์และฝึกโมเดล
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train,
    validation_data=val,
    epochs=10
)

# 💾 บันทึกโมเดล
os.makedirs('model', exist_ok=True)
model.save('model/custard_model.h5')

# 📈 วาดกราฟ Accuracy และ Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_plot.png')  # บันทึกเป็นรูปสำหรับรายงาน
plt.show()
