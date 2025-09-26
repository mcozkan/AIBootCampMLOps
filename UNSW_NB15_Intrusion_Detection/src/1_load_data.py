import pandas as pd
import kagglehub 
import os
import shutil
import mlflow

def main():
    with mlflow.start_run(run_name = "download_dataset"):
        print("=== 1. ADIM: VERİ İNDİRME ===")

        # Download latest version
        path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
        mlflow.log_param("dataset_path", path)

        print("Path to dataset files:", path)

        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        mlflow.log_param("cvs_files_found", len(csv_files))

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

            mlflow.log_artifact(dest_path, artifact_path = "raw_data")

        else:
            print("❌ The csv file is not found! CSV's in Folder:", csv_files)
            mlflow.log_param("status", "file_not_found")
            return "No file found!"

    
        print(f"✅ Dosya boyutu: {df.shape}")
        print(f"✅ Kaydedilen dosya: data/raw/UNSW_NB15_training-set.csv")
    
        # Veri hakkında temel bilgiler
        print(f"\n📊 Veri Seti Bilgileri:")
        print(f"- Satır sayısı: {len(df)}")
        print(f"- Sütun sayısı: {len(df.columns)}")
        print(f"- Sütunlar: {list(df.columns[:5])}...")
        print(f"- Eksik değerler: {df.isnull().sum().sum()}")


        # MLflow log
        mlflow.log_param("rows", len(df))
        mlflow.log_param("columns", len(df.columns))
        mlflow.log_param("first_columns", list(df.columns[:5]))
        mlflow.log_metric("missing_values", int(df.isnull().sum().sum()))


if __name__ == "__main__":
    main()