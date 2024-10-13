import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# # 加载CSV文件
# file_path = 'imputed_selected_features_Flam.csv'  # 更新路径
# data = pd.read_csv(file_path)
#
# # 提取SMILES, 标签和分类
# smiles_list = data['SMILES'].tolist()
# labels = data['Flammability'].values
# classifications = data['Classification'].values
#
# # 移除目标变量（标签）列
# data_features = data.drop(['SMILES', 'Flammability', 'Classification'], axis=1)
#
#
# # 函数：将SMILES转换为分子描述符和指纹特征
# def smiles_to_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     descriptors = [
#         Descriptors.MolWt(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.NumHDonors(mol),
#         Descriptors.NumHAcceptors(mol)
#     ]
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#     fingerprint_array = np.zeros((2048,))
#     Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
#     features = np.concatenate([descriptors, fingerprint_array])
#     return features
#
#
# # 将SMILES转换为特征
# features = []
# valid_indices = []  # 记录有效行的索引
# for i, smiles in enumerate(smiles_list):
#     feature = smiles_to_features(smiles)
#     if feature is not None:
#         features.append(feature)
#         valid_indices.append(i)
#
# # 转换为numpy数组
# features = np.array(features)
#
# # 基于有效索引过滤其他数据
# labels = labels[valid_indices]
# classifications = classifications[valid_indices]
# additional_features = data_features.iloc[valid_indices, :].values  # 确保只使用特征列
#
# # 合并所有特征
# all_features = np.hstack((features, additional_features))
#
# # 标准化特征
# scaler = StandardScaler()
# all_features = scaler.fit_transform(all_features)
#
# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)
#
# # 训练XGBoost模型进行SHAP分析
# xgb_model = XGBClassifier(eval_metric='logloss')
# xgb_model.fit(X_train, y_train)
#
# # 使用SHAP解释训练集上的模型预测
# explainer = shap.TreeExplainer(xgb_model)
# shap_values_train = explainer.shap_values(X_train)
#
# # 将X_train转换回DataFrame，便于操作
# X_train_df = pd.DataFrame(X_train,
#                           columns=['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in
#                                                                                          range(2048)] + list(
#                               data_features.columns))
#
# # 添加分类到训练集DataFrame
# X_train_df['Classification'] = classifications[:X_train.shape[0]]
#
# # 确保输出目录存在
# output_dir = 'Out_result_Flam'  # 更新为你想保存的路径
# os.makedirs(output_dir, exist_ok=True)
#
# # 为训练集中每个分类绘制SHAP值并保存为CSV文件
# for class_name in X_train_df['Classification'].unique():
#     class_indices = X_train_df['Classification'] == class_name
#
#     # 清理类名称以便文件保存
#     sanitized_class_name = str(class_name).replace('/', '_')
#
#     # 提取当前类的SHAP值
#     shap_values_class = shap_values_train[class_indices]
#     X_train_class = X_train[class_indices]
#
#     # 创建SHAP值的DataFrame并保存为CSV
#     shap_df = pd.DataFrame(shap_values_class, columns=X_train_df.columns[:-1])
#     shap_df.to_csv(os.path.join(output_dir, f'SHAP_values_train_{sanitized_class_name}_F.csv'), index=False)
#
#     # 绘制SHAP值
#     plt.figure(figsize=(10, 6))  # 调整图形尺寸
#     shap.summary_plot(shap_values_class, X_train_class, feature_names=X_train_df.columns[:-1], show=False)
#
#     plt.title(f'SHAP Summary Plot for {class_name} ', fontsize=16, pad=20, fontweight='bold')  # 调整标题的填充距离
#
#     plt.xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')  # 调整X轴标签字体
#     plt.ylabel('Feature', fontsize=12, fontweight='bold')  # 调整Y轴标签字体
#     plt.xticks(fontsize=10, fontweight='bold')  # 调整X轴刻度字体
#     plt.yticks(fontsize=10, fontweight='bold')  # 调整Y轴刻度字体
#
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整图形布局，确保标题和图例不会超出边界
#
#     # 保存图像到文件
#     plt.savefig(os.path.join(output_dir, f'SHAP_Summary_Plot_for_{sanitized_class_name}_F.png'))



'''
# 加载CSV文件
file_path = 'imputed_selected_features_Reactivity.csv'  # 更新路径
data = pd.read_csv(file_path)

# 提取SMILES, 标签和分类
smiles_list = data['SMILES'].tolist()
labels = data['Reactivity'].values
classifications = data['Classification'].values

# 移除目标变量（标签）列
data_features = data.drop(['SMILES', 'Reactivity', 'Classification'], axis=1)


# 函数：将SMILES转换为分子描述符和指纹特征
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint_array = np.zeros((2048,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    features = np.concatenate([descriptors, fingerprint_array])
    return features


# 将SMILES转换为特征
features = []
valid_indices = []  # 记录有效行的索引
for i, smiles in enumerate(smiles_list):
    feature = smiles_to_features(smiles)
    if feature is not None:
        features.append(feature)
        valid_indices.append(i)

# 转换为numpy数组
features = np.array(features)

# 基于有效索引过滤其他数据
labels = labels[valid_indices]
classifications = classifications[valid_indices]
additional_features = data_features.iloc[valid_indices, :].values  # 确保只使用特征列

# 合并所有特征
all_features = np.hstack((features, additional_features))

# 标准化特征
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

# 训练XGBoost模型进行SHAP分析
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 使用SHAP解释训练集上的模型预测
explainer = shap.TreeExplainer(xgb_model)
shap_values_train = explainer.shap_values(X_train)

# 将X_train转换回DataFrame，便于操作
X_train_df = pd.DataFrame(X_train,
                          columns=['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in
                                                                                         range(2048)] + list(
                              data_features.columns))

# 添加分类到训练集DataFrame
X_train_df['Classification'] = classifications[:X_train.shape[0]]

# 确保输出目录存在
output_dir = 'Out_result_Reactivity'  # 更新为你想保存的路径
os.makedirs(output_dir, exist_ok=True)

# 为训练集中每个分类绘制SHAP值并保存为CSV文件
for class_name in X_train_df['Classification'].unique():
    class_indices = X_train_df['Classification'] == class_name

    # 清理类名称以便文件保存
    sanitized_class_name = str(class_name).replace('/', '_')

    # 提取当前类的SHAP值
    shap_values_class = shap_values_train[class_indices]
    X_train_class = X_train[class_indices]

    # 创建SHAP值的DataFrame并保存为CSV
    shap_df = pd.DataFrame(shap_values_class, columns=X_train_df.columns[:-1])
    shap_df.to_csv(os.path.join(output_dir, f'SHAP_values_train_{sanitized_class_name}_R.csv'), index=False)

    # 绘制SHAP值
    plt.figure(figsize=(10, 6))  # 调整图形尺寸
    shap.summary_plot(shap_values_class, X_train_class, feature_names=X_train_df.columns[:-1], show=False)

    plt.title(f'SHAP Summary Plot for {class_name} in Reactivity ', fontsize=16, pad=20, fontweight='bold')  # 调整标题的填充距离

    plt.xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')  # 调整X轴标签字体
    plt.ylabel('Feature', fontsize=12, fontweight='bold')  # 调整Y轴标签字体
    plt.xticks(fontsize=10, fontweight='bold')  # 调整X轴刻度字体
    plt.yticks(fontsize=10, fontweight='bold')  # 调整Y轴刻度字体

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整图形布局，确保标题和图例不会超出边界

    # 保存图像到文件
    plt.savefig(os.path.join(output_dir, f'SHAP_Summary_Plot_for_{sanitized_class_name}_R.png'))

'''


# 加载CSV文件
file_path = 'imputed_selected_features_Toxcity.csv'  # 更新路径
data = pd.read_csv(file_path)

# 提取SMILES, 标签和分类
smiles_list = data['SMILES'].tolist()
labels = data['Toxicity'].values
classifications = data['Classification'].values

# 移除目标变量（标签）列
data_features = data.drop(['SMILES', 'Toxicity', 'Classification'], axis=1)


# 函数：将SMILES转换为分子描述符和指纹特征
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint_array = np.zeros((2048,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    features = np.concatenate([descriptors, fingerprint_array])
    return features


# 将SMILES转换为特征
features = []
valid_indices = []  # 记录有效行的索引
for i, smiles in enumerate(smiles_list):
    feature = smiles_to_features(smiles)
    if feature is not None:
        features.append(feature)
        valid_indices.append(i)

# 转换为numpy数组
features = np.array(features)

# 基于有效索引过滤其他数据
labels = labels[valid_indices]
classifications = classifications[valid_indices]
additional_features = data_features.iloc[valid_indices, :].values  # 确保只使用特征列

# 合并所有特征
all_features = np.hstack((features, additional_features))

# 标准化特征
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

# 训练XGBoost模型进行SHAP分析
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 使用SHAP解释训练集上的模型预测
explainer = shap.TreeExplainer(xgb_model)
shap_values_train = explainer.shap_values(X_train)

# 将X_train转换回DataFrame，便于操作
X_train_df = pd.DataFrame(X_train,
                          columns=['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in
                                                                                         range(2048)] + list(
                              data_features.columns))

# 添加分类到训练集DataFrame
X_train_df['Classification'] = classifications[:X_train.shape[0]]

# 确保输出目录存在
output_dir = 'Out_result_Toxicity'  # 更新为你想保存的路径
os.makedirs(output_dir, exist_ok=True)

# 为训练集中每个分类绘制SHAP值并保存为CSV文件
for class_name in X_train_df['Classification'].unique():
    class_indices = X_train_df['Classification'] == class_name

    # 清理类名称以便文件保存
    sanitized_class_name = str(class_name).replace('/', '_')

    # 提取当前类的SHAP值
    shap_values_class = shap_values_train[class_indices]
    X_train_class = X_train[class_indices]

    # 创建SHAP值的DataFrame并保存为CSV
    shap_df = pd.DataFrame(shap_values_class, columns=X_train_df.columns[:-1])
    shap_df.to_csv(os.path.join(output_dir, f'SHAP_values_train_{sanitized_class_name}_T.csv'), index=False)

    # 绘制SHAP值
    plt.figure(figsize=(10, 6))  # 调整图形尺寸
    shap.summary_plot(shap_values_class, X_train_class, feature_names=X_train_df.columns[:-1], show=False)

    plt.title(f'SHAP Summary Plot for {class_name} in Toxicity ', fontsize=16, pad=20, fontweight='bold')  # 调整标题的填充距离

    plt.xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')  # 调整X轴标签字体
    plt.ylabel('Feature', fontsize=12, fontweight='bold')  # 调整Y轴标签字体
    plt.xticks(fontsize=10, fontweight='bold')  # 调整X轴刻度字体
    plt.yticks(fontsize=10, fontweight='bold')  # 调整Y轴刻度字体

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整图形布局，确保标题和图例不会超出边界

    # 保存图像到文件
    plt.savefig(os.path.join(output_dir, f'SHAP_Summary_Plot_for_{sanitized_class_name}_T.png'))



'''
# 加载CSV文件
file_path = 'imputed_selected_features_W.csv'  # 更新路径
data = pd.read_csv(file_path)

# 提取SMILES, 标签和分类
smiles_list = data['SMILES'].tolist()
labels = data['W'].values
classifications = data['Classification'].values

# 移除目标变量（标签）列
data_features = data.drop(['SMILES', 'W', 'Classification'], axis=1)


# 函数：将SMILES转换为分子描述符和指纹特征
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint_array = np.zeros((2048,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    features = np.concatenate([descriptors, fingerprint_array])
    return features


# 将SMILES转换为特征
features = []
valid_indices = []  # 记录有效行的索引
for i, smiles in enumerate(smiles_list):
    feature = smiles_to_features(smiles)
    if feature is not None:
        features.append(feature)
        valid_indices.append(i)

# 转换为numpy数组
features = np.array(features)

# 基于有效索引过滤其他数据
labels = labels[valid_indices]
classifications = classifications[valid_indices]
additional_features = data_features.iloc[valid_indices, :].values  # 确保只使用特征列

# 合并所有特征
all_features = np.hstack((features, additional_features))

# 标准化特征
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

# 训练XGBoost模型进行SHAP分析
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 使用SHAP解释训练集上的模型预测
explainer = shap.TreeExplainer(xgb_model)
shap_values_train = explainer.shap_values(X_train)

# 将X_train转换回DataFrame，便于操作
X_train_df = pd.DataFrame(X_train,
                          columns=['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in
                                                                                         range(2048)] + list(
                              data_features.columns))

# 添加分类到训练集DataFrame
X_train_df['Classification'] = classifications[:X_train.shape[0]]

# 确保输出目录存在
output_dir = 'Out_result_W'  # 更新为你想保存的路径
os.makedirs(output_dir, exist_ok=True)

# 为训练集中每个分类绘制SHAP值并保存为CSV文件
for class_name in X_train_df['Classification'].unique():
    class_indices = X_train_df['Classification'] == class_name

    # 清理类名称以便文件保存
    sanitized_class_name = str(class_name).replace('/', '_')

    # 提取当前类的SHAP值
    shap_values_class = shap_values_train[class_indices]
    X_train_class = X_train[class_indices]

    # 创建SHAP值的DataFrame并保存为CSV
    shap_df = pd.DataFrame(shap_values_class, columns=X_train_df.columns[:-1])
    shap_df.to_csv(os.path.join(output_dir, f'SHAP_values_train_{sanitized_class_name}_W.csv'), index=False)

    # 绘制SHAP值
    plt.figure(figsize=(10, 6))  # 调整图形尺寸
    shap.summary_plot(shap_values_class, X_train_class, feature_names=X_train_df.columns[:-1], show=False)

    plt.title(f'SHAP Summary Plot for {class_name} in W', fontsize=16, pad=20, fontweight='bold')  # 调整标题的填充距离

    plt.xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')  # 调整X轴标签字体
    plt.ylabel('Feature', fontsize=12, fontweight='bold')  # 调整Y轴标签字体
    plt.xticks(fontsize=10, fontweight='bold')  # 调整X轴刻度字体
    plt.yticks(fontsize=10, fontweight='bold')  # 调整Y轴刻度字体

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整图形布局，确保标题和图例不会超出边界

    # 保存图像到文件
    plt.savefig(os.path.join(output_dir, f'SHAP_Summary_Plot_for_{sanitized_class_name}_W.png'))
    
'''