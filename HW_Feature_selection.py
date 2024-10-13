import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
#toxicity
# 读取数据时指定low_memory=False以避免警告
data = pd.read_csv("HW-2D-Toxicity.csv", low_memory=False)
smiles = data.iloc[:, 0]  # 第一列
target = data.iloc[:, -1]  # 最后一列

# 假设第一列是SMILES字符串，忽略第一列
numeric_data = data.iloc[:, 1:]

# 处理非数值字符，将其替换为NaN
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# 移除包含NaN值的行
numeric_data.dropna(inplace=True)

# 分离特征和目标变量，假设最后一列是目标变量
X = numeric_data.iloc[:, :-1]
y = numeric_data.iloc[:, -1]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林进行特征选择
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 使用SelectFromModel进行特征选择
model_rf = SelectFromModel(rf, threshold=0.0025, prefit=True)
X_train_selected = model_rf.transform(X_train)
selected_features = X.columns[model_rf.get_support()]

# 使用交叉验证评估选定特征的模型性能
rf_selected = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_selected, X_train_selected, y_train, cv=5)

print("交叉验证的平均准确率:", np.mean(cv_scores))

# 获取特征重要性
importances = rf.feature_importances_

# 过滤掉特征重要性小于或等于0的特征
indices = np.where(importances > 0.001)[0]
filtered_importances = importances[indices]
filtered_features = X.columns[indices]

# 排序特征重要性
sorted_indices = np.argsort(filtered_importances)[::-1]

# 绘制特征重要性大于0的特征
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Greater than 0.001)")
plt.bar(range(len(sorted_indices)), filtered_importances[sorted_indices], align="center")
plt.xticks(range(len(sorted_indices)), filtered_features[sorted_indices], rotation=90)
plt.savefig("feature importance_1.png")

# 从原始数据中提取选定的特征列（包括SMILES和目标变量）
selected_data = data[['SMILES'] + list(selected_features) + [target.name]]

# 将选定的特征、SMILES和目标变量保存到新的CSV文件
selected_data.to_csv("selected_features_with_smiles_target_Toxcity.csv", index=False)


# # Flammability
# # 读取数据时指定low_memory=False以避免警告
# data = pd.read_csv("HW-2D-1-Flam.csv", low_memory=False)
# smiles = data.iloc[:, 0]  # 第一列
# target = data.iloc[:, -1]  # 最后一列
#
# # 假设第一列是SMILES字符串，忽略第一列
# numeric_data = data.iloc[:, 1:]
#
# # 处理非数值字符，将其替换为NaN
# numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
#
# # 移除包含NaN值的行
# numeric_data.dropna(inplace=True)
#
# # 分离特征和目标变量，假设最后一列是目标变量
# X = numeric_data.iloc[:, :-1]
# y = numeric_data.iloc[:, -1]
#
# # 分割数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 使用随机森林进行特征选择
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)
#
# # 使用SelectFromModel进行特征选择
# model_rf = SelectFromModel(rf, threshold=0.0025, prefit=True)
# X_train_selected = model_rf.transform(X_train)
# selected_features = X.columns[model_rf.get_support()]
#
# # 使用交叉验证评估选定特征的模型性能
# rf_selected = RandomForestClassifier(random_state=42)
# cv_scores = cross_val_score(rf_selected, X_train_selected, y_train, cv=5)
#
# print("交叉验证的平均准确率:", np.mean(cv_scores))
#
# # 获取特征重要性
# importances = rf.feature_importances_
#
# # 过滤掉特征重要性小于或等于0的特征
# indices = np.where(importances > 0.001)[0]
# filtered_importances = importances[indices]
# filtered_features = X.columns[indices]
#
# # 排序特征重要性
# sorted_indices = np.argsort(filtered_importances)[::-1]
#
# # 绘制特征重要性大于0的特征
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances (Greater than 0.001)")
# plt.bar(range(len(sorted_indices)), filtered_importances[sorted_indices], align="center")
# plt.xticks(range(len(sorted_indices)), filtered_features[sorted_indices], rotation=90)
# plt.savefig("feature importance_1.png")
#
# # 从原始数据中提取选定的特征列（包括SMILES和目标变量）
# selected_data = data[['SMILES'] + list(selected_features) + [target.name]]
#
# # 将选定的特征、SMILES和目标变量保存到新的CSV文件
# selected_data.to_csv("selected_features_with_smiles_target_Flam.csv", index=False)
#
# print("选定的特征已保存到selected_features_with_smiles_target.csv")


# # Reactivity
# # 读取数据时指定low_memory=False以避免警告
# data = pd.read_csv("HW-2D-1-Reactivity.csv", low_memory=False)
# smiles = data.iloc[:, 0]  # 第一列
# target = data.iloc[:, -1]  # 最后一列
#
# # 假设第一列是SMILES字符串，忽略第一列
# numeric_data = data.iloc[:, 1:]
#
# # 处理非数值字符，将其替换为NaN
# numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
#
# # 移除包含NaN值的行
# numeric_data.dropna(inplace=True)
#
# # 分离特征和目标变量，假设最后一列是目标变量
# X = numeric_data.iloc[:, :-1]
# y = numeric_data.iloc[:, -1]
#
# # 分割数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 使用随机森林进行特征选择
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)
#
# # 使用SelectFromModel进行特征选择
# model_rf = SelectFromModel(rf, threshold=0.0025, prefit=True)
# X_train_selected = model_rf.transform(X_train)
# selected_features = X.columns[model_rf.get_support()]
#
# # 使用交叉验证评估选定特征的模型性能
# rf_selected = RandomForestClassifier(random_state=42)
# cv_scores = cross_val_score(rf_selected, X_train_selected, y_train, cv=5)
#
# print("交叉验证的平均准确率:", np.mean(cv_scores))
#
# # 获取特征重要性
# importances = rf.feature_importances_
#
# # 过滤掉特征重要性小于或等于0的特征
# indices = np.where(importances > 0.001)[0]
# filtered_importances = importances[indices]
# filtered_features = X.columns[indices]
#
# # 排序特征重要性
# sorted_indices = np.argsort(filtered_importances)[::-1]
#
# # 绘制特征重要性大于0的特征
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances (Greater than 0.001)")
# plt.bar(range(len(sorted_indices)), filtered_importances[sorted_indices], align="center")
# plt.xticks(range(len(sorted_indices)), filtered_features[sorted_indices], rotation=90)
# plt.savefig("feature importance_1.png")
#
# # 从原始数据中提取选定的特征列（包括SMILES和目标变量）
# selected_data = data[['SMILES'] + list(selected_features) + [target.name]]
#
# # 将选定的特征、SMILES和目标变量保存到新的CSV文件
# selected_data.to_csv("selected_features_with_smiles_target_Reactivity.csv", index=False)
#
# print("选定的特征已保存到selected_features_with_smiles_target.csv")

# # Water
# # 读取数据时指定low_memory=False以避免警告
# data = pd.read_csv("HW-2D-1-Reactivity.csv", low_memory=False)
# smiles = data.iloc[:, 0]  # 第一列
# target = data.iloc[:, -1]  # 最后一列
#
# # 假设第一列是SMILES字符串，忽略第一列
# numeric_data = data.iloc[:, 1:]
#
# # 处理非数值字符，将其替换为NaN
# numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
#
# # 移除包含NaN值的行
# numeric_data.dropna(inplace=True)
#
# # 分离特征和目标变量，假设最后一列是目标变量
# X = numeric_data.iloc[:, :-1]
# y = numeric_data.iloc[:, -1]
#
# # 分割数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 使用随机森林进行特征选择
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)
#
# # 使用SelectFromModel进行特征选择
# model_rf = SelectFromModel(rf, threshold=0.0025, prefit=True)
# X_train_selected = model_rf.transform(X_train)
# selected_features = X.columns[model_rf.get_support()]
#
# # 使用交叉验证评估选定特征的模型性能
# rf_selected = RandomForestClassifier(random_state=42)
# cv_scores = cross_val_score(rf_selected, X_train_selected, y_train, cv=5)
#
# print("交叉验证的平均准确率:", np.mean(cv_scores))
#
# # 获取特征重要性
# importances = rf.feature_importances_
#
# # 过滤掉特征重要性小于或等于0的特征
# indices = np.where(importances > 0.001)[0]
# filtered_importances = importances[indices]
# filtered_features = X.columns[indices]
#
# # 排序特征重要性
# sorted_indices = np.argsort(filtered_importances)[::-1]
#
# # 绘制特征重要性大于0的特征
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances (Greater than 0.001)")
# plt.bar(range(len(sorted_indices)), filtered_importances[sorted_indices], align="center")
# plt.xticks(range(len(sorted_indices)), filtered_features[sorted_indices], rotation=90)
# plt.savefig("feature importance_1.png")
#
# # 从原始数据中提取选定的特征列（包括SMILES和目标变量）
# selected_data = data[['SMILES'] + list(selected_features) + [target.name]]
#
# # 将选定的特征、SMILES和目标变量保存到新的CSV文件
# selected_data.to_csv("selected_features_with_smiles_target_W.csv", index=False)
#
# print("选定的特征已保存到selected_features_with_smiles_target.csv")
