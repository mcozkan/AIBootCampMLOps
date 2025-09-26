import pandas as pd
import kagglehub 
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")

print("Path to dataset files:", path)

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

target_file = None
for f in csv_files:
    if "training-set" in f.lower():
        target_file = f
        break

if target_file:
    file_path = os.path.join(path, target_file)
    df = pd.read_csv(file_path)
    print("file loaded!...")
    print(df.head())


    dest_dir = "data/raw"
    os.makedirs(dest_dir, exist_ok = True)
    dest_path = os.path.join(dest_dir, os.path.basename(file_path))

    shutil.copy(file_path, dest_path)
    print(f"✅ Copied to project folder: {dest_path}")
    print("ℹ️ Now run this in terminal to track with DVC:")
    print(f"   dvc add {dest_path}")

else:
    print("❌ The csv file is not found! CSV's in Folder:", csv_files)

