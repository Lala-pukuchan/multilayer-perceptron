import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# データの読み込み（列名なし）
data = pd.read_csv('./resources/data.csv', header=None)

# 'M'または'B'が含まれる列を検索
diagnosis_column = None
for col in data.columns:
    if data[col].isin(['M', 'B']).any():
        diagnosis_column = col
        break

if diagnosis_column is None:
    raise ValueError("Failed to find the diagnosis column.")

# 列名を設定
columns = ['feature' + str(i) if i != diagnosis_column else 'diagnosis' for i in range(len(data.columns))]
data.columns = columns

# 特徴量とラベルに分割
X = data.drop(columns=['diagnosis'])
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Mを1, Bを0に変換

# 訓練データと検証データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target for training and validation sets
train_data = pd.concat([X_train, y_train], axis=1)
valid_data = pd.concat([X_valid, y_valid], axis=1)

# Save the split data to CSV files
train_data.to_csv('./resources/data_training.csv', index=False)
valid_data.to_csv('./resources/data_validation.csv', index=False)

# データの正規化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

print("Data preparation completed. Split data saved to CSV files.")