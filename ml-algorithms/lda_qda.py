import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# 生成合成数据
def generate_data(cov_class_1, cov_class_2, n_samples=300):
    rng = np.random.RandomState(0)
    X1 = rng.randn(n_samples, 2) @ cov_class_1
    X2 = rng.randn(n_samples, 2) @ cov_class_2 + np.array([1, 1])
    X = np.concatenate([X1, X2])
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    return X, y

def plot_decision_boundary(ax, clf, X, y, title):
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制轮廓和数据点
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=25)
    ax.set_title(title)

#  scenario 1: 相同协方差矩阵（LDA理想场景）
covariance = np.array([[0.0, -0.23], [0.83, 0.23]])
X1, y1 = generate_data(covariance, covariance)

#  scenario 2: 不同协方差矩阵（QDA理想场景）
cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
cov_class_2 = cov_class_1.T  # 转置矩阵，结构不同
X2, y2 = generate_data(cov_class_1, cov_class_2)

# 训练并比较模型
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, (X, y) in enumerate([(X1, y1), (X2, y2)]):
    lda.fit(X, y)
    qda.fit(X, y)
    
    plot_decision_boundary(axes[i, 0], lda, X, y, f"Dataset {i+1} : LDA")
    plot_decision_boundary(axes[i, 1], qda, X, y, f"Dataset {i+1} : QDA")

plt.tight_layout()
plt.show()