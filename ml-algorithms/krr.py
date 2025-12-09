import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 生成模拟的股票历史价格数据（示例中使用虚拟数据）
# 假设有500个交易日的收盘价，并生成一些简单的技术指标作为特征
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
# 模拟收盘价：假设有一个基础趋势和一些随机波动
base_trend = np.linspace(100, 150, 500)
random_noise = np.random.normal(0, 2, 500)
prices = base_trend + random_noise

# 创建特征DataFrame：这里简单使用滞后价格和移动平均线作为示例特征
data = pd.DataFrame({'Close': prices})
data['Close_Lag1'] = data['Close'].shift(1)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_10'] = data['Close'].rolling(window=10).mean()
data = data.dropna() # 删除因计算指标产生的缺失值

# 2. 修改目标变量：预测价格变化量（Diff），而不是绝对价格
# 这样可以去除趋势（Detrending），使数据平稳，适合RBF核
data['Target_Diff'] = data['Close'].shift(-1) - data['Close']
data = data.dropna()

X = data[['Close_Lag1', 'MA_5', 'MA_10']]
y = data['Target_Diff']  # 现在的目标是预测涨跌额

# 3. 划分训练集和测试集（按时间顺序划分）
# 注意：我们需要保留划分前的原始价格数据，用于后续还原预测值
train_size = int(len(data) * 0.8)
X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 4. 数据预处理 - 标准化特征
# 关键修复：只在训练集上fit，避免数据泄露
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 5. 创建并训练KRR模型
# 使用RBF（高斯）核。由于现在预测的是平稳的Diff，RBF效果会好很多
krr_model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
krr_model.fit(X_train, y_train)

# 6. 进行预测
y_pred_train_diff = krr_model.predict(X_train)
y_pred_test_diff = krr_model.predict(X_test)

# 7. 还原预测结果（从 Diff 还原回绝对价格）
# 预测的明天价格 = 今天的价格 + 预测的涨跌额
# 对应关系：X_test的每一行对应当天的特征，我们要预测明天的价格
# 所以基准价格是 X_test 对应的当天的 'Close' 价格
# 注意：X 中没有包含当天的 'Close'，但我们可以从原始 data 中获取
test_indices = X_test_raw.index
current_prices_test = data.loc[test_indices, 'Close'].values
current_prices_train = data.loc[X_train_raw.index, 'Close'].values

y_pred_test_price = current_prices_test + y_pred_test_diff
y_pred_train_price = current_prices_train + y_pred_train_diff

# 真实的目标价格（用于对比）
y_test_price = current_prices_test + y_test.values
y_train_price = current_prices_train + y_train.values

# 8. 评估与可视化
train_rmse = np.sqrt(np.mean((y_train_price - y_pred_train_price)**2))
test_rmse = np.sqrt(np.mean((y_test_price - y_pred_test_price)**2))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(data.loc[test_indices].index, y_test_price, label='Actual Price', color='blue')
plt.plot(data.loc[test_indices].index, y_pred_test_price, label='Predicted Price', color='red', linestyle='--')
plt.title('Stock Price Prediction using KRR (Predicting Diff & Reconstructing)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()