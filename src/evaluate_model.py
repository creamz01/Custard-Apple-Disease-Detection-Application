import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, f1_score

# --- ใช้ชื่อ model จาก argument ---
if len(sys.argv) < 2:
    print("❌ Usage: python evaluate_model.py model_path.h5")
    sys.exit(1)

model_path = sys.argv[1]
print(f"\n🚀 Evaluating model: {model_path}\n")
model = load_model(model_path)

# Prepare test data
img_height, img_width = 224, 224
batch_size = 16
test_path = 'dataset/test'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Predict
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

# Classification report
print("\n📄 Classification Report:")
report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {os.path.basename(model_path)}")
plt.tight_layout()
plt.savefig(f"confusion_matrix_{os.path.basename(model_path).replace('.h5','')}.png")
plt.show()

# Bar chart F1-score per class
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, f1, color='skyblue')
plt.ylim(0, 1.05)
plt.title(f"F1-Score per Class - {os.path.basename(model_path)}")
plt.xlabel("Class")
plt.ylabel("F1-Score")

# Add value labels
for bar, score in zip(bars, f1):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{score:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"f1_score_per_class_{os.path.basename(model_path).replace('.h5','')}.png")
plt.show()

# --- Summary: Accuracy, F1 macro average, Total Params ---
# Accuracy
accuracy = np.mean(y_true == y_pred) * 100

# F1 macro average
f1_macro = f1_score(y_true, y_pred, average='macro')

# Total params
total_params = model.count_params()
total_params_million = total_params / 1e6

# Print summary
print(f"\n📊 Summary for {os.path.basename(model_path)}:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1-score (macro average): {f1_macro:.4f}")
print(f"Total Parameters: {total_params_million:.2f} M")

print("\n✅ Done! รูป saved: Confusion Matrix + F1-score per class")
