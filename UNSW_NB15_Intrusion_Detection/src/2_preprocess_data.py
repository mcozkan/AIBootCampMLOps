import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    with mlflow.start_run(run_name = "preprocess_data"):
        print("=== 1. STEP : PREPROCESS DATA ===")
        df = pd.read_csv("/data/raw/UNSW_NB15_training-set.csv")
        df_num = df.select_dtypes(include = ['float', 'int'])


        def compare(df1_num, df2_all):
            number = []
            new = []
            for i in df1_num.columns:
                number.append(i)
            for j in df2_all.columns:
                new.append(j)
            if len(number) == len(new):
            print("No categorical data dropped from the dataset!")
        else:
            print(f"New non-cat dataset features count : {len(new)}")