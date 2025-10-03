import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import shutil
import os
import datetime



def preprocess_data():
    start_func = datetime.datetime.now()
    with mlflow.start_run(run_name = "preprocess_data"):
        print("=== 2. STEP : PREPROCESS DATA ===")

        df = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")
        # Checking if all the features are numeric

        df_num = df.select_dtypes(include = ['float', 'int'])

        num_original = len(df)
        num_new = len(df_num)
       
        if num_original == num_new:
            print("No categorical data dropped from the dataset!")
        else:
            print(f"New non-cat dataset features count : {num_new}")

        mlflow.log_param('total_columns', num_original)
        mlflow.log_param('numeric_columns', num_new)
        mlflow.log_param('dropped_columns', num_original - num_new)

        # Drawing sample from the population
        normal = df_num[df_num['label'] == 0].sample(2000, random_state=42)
        attack = df_num[df_num['label'] == 1].sample(500, random_state = 42)
        df_sample = pd.concat([normal, attack]).sample(frac=1, random_state=42)


        # create new folder for the processed data
        processed_dir = "processed"
        os.makedirs(processed_dir, exist_ok = True)
        
        # Save the dataframes 
        normal_path = os.path.join(processed_dir, "normal_samples.csv")
        attack_path = os.path.join(processed_dir, "attack_samples.csv")
        df_sample_path = os.path.join(processed_dir, "df_sample.csv")

        normal.to_csv(normal_path, index=False)
        attack.to_csv(attack_path, index=False)
        df_sample.to_csv(df_sample_path, index=False)

        mlflow.log_artifact(normal_path)
        mlflow.log_artifact(attack_path)
        mlflow.log_artifact(df_sample_path)

        print(f"Saved processed data to {processed_dir}")

        # Dataframes 
        mlflow.log_param('normal_sample_count', normal.shape[0])
        mlflow.log_param('attack_sample_count', attack.shape[0])
        
        mlflow.log_param('df_sample_n_features', df_sample.shape[1])
        mlflow.log_param('df_sample_n_samples', df_sample.shape[0])

        # Data slicing as X and y

        X = df_sample.drop('label', axis = 1)
        y = df_sample['label']

        mlflow.log_param('n_features', X.shape[1])
        mlflow.log_param('n_samples', X.shape[0])

        


        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        end_func = datetime.datetime.now()

        duration = end_func - start_func
        mlflow.log_metric("execution_time_seconds_preprocess", duration.total_seconds())
        print(f"✅ Model çalışma süresi: {duration.total_seconds():.2f} saniye")

        return X_scaled, X, y

if __name__ == "__main__":
    mlflow.set_experiment("UNSW_NB15_Experiment")
    X_scaled, X, y= preprocess_data()
    print(f"\nFinal shapes - X: {X.shape}, y: {y.shape}")
    