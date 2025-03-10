import pandas as pd
import numpy as np

def train_test_split(data, test_size=0.2, random_state=42):
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return data.iloc[train_indices], data.iloc[test_indices]

def main():
    # ヘッダーがない状態でデータを読み込む
    data = pd.read_csv("./resources/data.csv", header=None)
    
    # ここでは列名の変更や診断値の変換を行わず、元のまま保持する
    # （evaluation.pyと同様にIDとM/Bの値は変更されません）
    
    # データを分割
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # ヘッダーなし、インデックスなしで保存する
    train_data.to_csv("./resources/data_training.csv", index=False, header=False)
    valid_data.to_csv("./resources/data_test.csv", index=False, header=False)

if __name__ == "__main__":
    main()
