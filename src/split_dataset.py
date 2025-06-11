import os, shutil
from sklearn.model_selection import train_test_split

base_dir = 'data'
target_dir = 'dataset'

for disease in os.listdir(base_dir):
    class_path = os.path.join(base_dir, disease)
    images = os.listdir(class_path)

    train, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    for group, name in zip([train, val, test], ['train', 'val', 'test']):
        save_dir = os.path.join(target_dir, name, disease)
        os.makedirs(save_dir, exist_ok=True)
        for img in group:
            src = os.path.join(class_path, img)
            dst = os.path.join(save_dir, img)
            shutil.copy2(src, dst)
