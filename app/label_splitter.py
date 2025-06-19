import os
import shutil

dataset_dir = "C:/Users/ezgit/Documents/moodiplant/data"
input_dirs = [os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "test")]

output_dir = "PlantDoc_Binary"
os.makedirs(f"{output_dir}/healthy", exist_ok=True)
os.makedirs(f"{output_dir}/unhealthy", exist_ok=True)

# Disease keywords to detect unhealthy samples
unhealthy_keywords = ["rust", "mold", "blight", "spot", "mildew", "scab", "bacterial", "late", "early", "virus", "rot", "alternaria", "canker"]

# Go through each class folder
for parent in input_dirs:
    for class_folder in os.listdir(parent):
        folder_path = os.path.join(parent, class_folder)
        if not os.path.isdir(folder_path):
            continue

        label = "unhealthy" if any(word in class_folder.lower() for word in unhealthy_keywords) else "healthy"

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".jpg"):
                continue

            src = os.path.join(folder_path, filename)
            dst = os.path.join(output_dir, label, filename)
            shutil.copy(src, dst)

print("✅ Done! Files are organized for healthy/unhealthy classification.")
