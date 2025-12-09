# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征矩阵
y = iris.target  # 目标标签

# 为了方便可视化，我们只使用前两个特征（花萼长度和花萼宽度）
# 在实际项目中，通常会使用所有特征，或者使用降维技术（如PCA）降到2-3维进行可视化
X = X[:, :2]

# 划分数据集：70%用于训练，30%用于测试
# random_state=42 保证每次运行划分结果一致，便于复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征（SVM对数据尺度敏感，通常需要进行标准化）
# SVM 试图最大化决策边界的间隔，如果特征尺度差异大，大尺度的特征会主导距离计算
scaler = StandardScaler()
# fit_transform: 在训练集上计算均值和方差，并应用标准化
X_train_scaled = scaler.fit_transform(X_train)
# transform: 使用训练集的均值和方差对测试集进行标准化（防止数据泄露）
X_test_scaled = scaler.transform(X_test)

# 创建SVM分类器，使用线性核函数
# C是正则化参数，控制错误分类的惩罚力度
# C值较小 -> 容忍更多的误分类（软间隔），边界更平滑，泛化能力可能更强
# C值较大 -> 尽量减少误分类（硬间隔），边界更复杂，可能导致过拟合
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# 使用标准化后的训练数据训练模型
svm_classifier.fit(X_train_scaled, y_train)

# 使用训练好的模型对标准化后的测试集进行预测
y_pred = svm_classifier.predict(X_test_scaled)

# 计算并打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 打印详细的分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))


# 定义要搜索的参数网格
# SVM的效果高度依赖于参数的选择
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正则化参数候选列表
    'kernel': ['linear', 'rbf'],  # 核函数：线性核 vs 径向基函数核（高斯核）
    'gamma': ['scale', 'auto', 0.1, 1]  # 核系数：影响RBF核的分布范围
    # gamma越大 -> 高斯分布越窄 -> 模型只关注支持向量附近的样本 -> 易过拟合
    # gamma越小 -> 高斯分布越宽 -> 模型受更多样本影响 -> 边界更平滑
}

# 创建GridSearchCV对象
# GridSearchCV 会遍历 param_grid 中的所有组合
# cv=5 表示使用 5 折交叉验证：将训练集分成 5 份，轮流用 4 份训练，1 份验证
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, verbose=1)

# 在训练数据上执行网格搜索（自动寻找最佳参数）
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合
print("找到的最佳参数: ", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"调优后模型准确率: {best_accuracy:.2f}")

def plot_decision_boundary(model, X, y, title):
    """
    绘制SVM的决策边界和支持向量
    """
    # 创建网格点，用于绘制背景的决策区域
    h = 0.02  # 网格步长
    # 确定网格的边界范围，稍微留出一点边距
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # np.meshgrid 生成二维网格坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测整个网格中每个点的类别
    # np.c_ 将 xx 和 yy 拉平后按列拼接，构造出测试点矩阵
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # 将预测结果重塑回网格形状，以便绘制等高线
    Z = Z.reshape(xx.shape)
    
    # 绘制填充等高线图（决策区域）
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # 绘制原始数据散点图
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    
    # 标记支持向量（决定决策边界的关键样本点）
    sv = model.support_vectors_
    # 用空心圆圈圈出支持向量
    plt.scatter(sv[:, 0], sv[:, 1], facecolors='none', edgecolors='k', s=100, linewidths=1, label='Support Vectors')
    
    plt.colorbar(scatter)
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')
    plt.title(title)
    plt.legend()
    plt.show()

# 绘制调优后的模型决策边界
plot_decision_boundary(best_model, X_train_scaled, y_train, "SVM Decision Boundary (Optimized)")