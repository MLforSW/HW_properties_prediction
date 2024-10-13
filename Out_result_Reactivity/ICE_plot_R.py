import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import os
import re

#制定feature
'''
# ICE_plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier

# 加载原始数据
file_path = 'imputed_selected_features_Reactivity.csv'  # 替换为你原始数据集的路径
data = pd.read_csv(file_path)

# 提取SMILES和标签
smiles_list = data['SMILES'].tolist()
labels = data['Reactivity'].values

# 函数：将SMILES转换为分子描述符和指纹
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 提取描述符
    descriptors = [
        Descriptors.MolWt(mol),  # 分子量
        Descriptors.MolLogP(mol),  # LogP
        Descriptors.NumHDonors(mol),  # 氢键供体数量
        Descriptors.NumHAcceptors(mol)  # 氢键受体数量
    ]
    # 生成Morgan指纹
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint_array = np.zeros((2048,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    # 合并描述符和指纹
    features = np.concatenate([descriptors, fingerprint_array])
    return features

# 将SMILES转换为特征
features = []
for smiles in smiles_list:
    feature = smiles_to_features(smiles)
    if feature is not None:
        features.append(feature)

# 转换为numpy数组
features = np.array(features)

# 获取原始数据集中的其他特征（从第二列到倒数第二列）
additional_features = data.iloc[:, 1:-2].values

# 合并所有特征
all_features = np.hstack((features, additional_features))

# 标准化特征
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

# 特征名称列表
feature_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in range(2048)] + list(data.columns[1:-2])

# 加载CSV文档中各类别的特征重要性
importance_data = pd.read_excel('Top_4_Features_Per_Category.xlsx')

# 获取类别名称
categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别

# 创建输出目录
output_dir_ice = 'Out_result_Flam_R_ICE'
os.makedirs(output_dir_ice, exist_ok=True)

# 计算总的图像数量和行数
total_plots = len(categories) * 2  # 每个类别2个图
n_cols = 4  # 每行4个图
n_rows = (total_plots + n_cols - 1) // n_cols  # 计算需要的行数

# 创建一个大的图像面板，用于合并所有子图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 6))  # 每行4个图，总共n_rows行
axes = axes.flatten()

plot_index = 0
# 添加全局字体设置，确保字体大小和粗细应用于整个图形
plt.rcParams['axes.labelsize'] = 14  # 设置x轴和y轴标签的字体大小
plt.rcParams['axes.labelweight'] = 'bold'  # 设置x轴和y轴标签的字体粗细
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
# 全局设置线条粗细
plt.rcParams['lines.linewidth'] = 2  # 设置线条粗细，默认值为1.5，你可以根据需要调整这个值
plt.rcParams['lines.color'] = 'blue'  # 设置线条颜色（可选）
# 针对每个类别生成 ICE 图
for category in categories:
    # 找到当前类别中最重要的两个特征
    top_features = importance_data.nlargest(2, category)['Feature'].values  # 获取两个最重要的特征

    # 获取特征在原始特征集中的索引
    feature_indices = [feature_names.index(feature) for feature in top_features]

    # 重新训练模型，针对每个类别进行单独的特征重要性评估
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # 为每个最重要的特征生成 ICE 图
    for i in feature_indices:
        feature_name = feature_names[i]
        print(f"Generating ICE Plot for {feature_name} in category {category}")

        # 使用 PartialDependenceDisplay 生成 ICE 图
        display = PartialDependenceDisplay.from_estimator(
            xgb_model, X_train, [i], kind="both", ax=axes[plot_index], grid_resolution=50,
            feature_names=feature_names
        )

        # 设置X轴和Y轴标签字体大小和粗细
        axes[plot_index].set_xlabel(feature_name, fontsize=14, fontweight='bold')
        axes[plot_index].set_ylabel("Partial dependence", fontsize=14, fontweight='bold')

        # 设置标题
        axes[plot_index].set_title(f"ICE Plot for {feature_name} in {category}", fontsize=16, fontweight='bold')

        plot_index += 1

# 调整布局以适应所有子图
plt.tight_layout()
# 保存合并的图像
plt.savefig(os.path.join(output_dir_ice, 'Combined_ICE_Plots_R.png'))
'''

#3D-dimension plot


# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import partial_dependence
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
#
# # 设置全局字体大小和粗细
# plt.rcParams['font.size'] = 14  # 设置全局字体大小
# plt.rcParams['font.weight'] = 'bold'  # 设置全局字体粗细
# plt.rcParams['axes.labelsize'] = 14  # 设置轴标签的字体大小
# plt.rcParams['axes.labelweight'] = 'bold'  # 设置轴标签的字体粗细
# plt.rcParams['axes.titlesize'] = 14  # 设置标题的字体大小
# plt.rcParams['axes.titleweight'] = 'bold'  # 设置标题的字体粗细
# plt.rcParams['xtick.labelsize'] = 12  # 设置x轴刻度标签的字体大小
# plt.rcParams['ytick.labelsize'] = 12  # 设置y轴刻度标签的字体大小
# plt.rcParams['legend.fontsize'] = 14  # 设置图例的字体大小
#
# # 加载原始数据
# file_path = 'imputed_selected_features_Reactivity.csv'  # 替换为你原始数据集的路径
# data = pd.read_csv(file_path)
#
# # 提取SMILES和标签
# smiles_list = data['SMILES'].tolist()
# labels = data['Reactivity'].values
#
# # 函数：将SMILES转换为分子描述符和指纹
# def smiles_to_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     # 提取描述符
#     descriptors = [
#         Descriptors.MolWt(mol),  # 分子量
#         Descriptors.MolLogP(mol),  # LogP
#         Descriptors.NumHDonors(mol),  # 氢键供体数量
#         Descriptors.NumHAcceptors(mol)  # 氢键受体数量
#     ]
#     # 生成Morgan指纹
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#     fingerprint_array = np.zeros((2048,))
#     Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
#     # 合并描述符和指纹
#     features = np.concatenate([descriptors, fingerprint_array])
#     return features
#
# # 将SMILES转换为特征
# features = []
# for smiles in smiles_list:
#     feature = smiles_to_features(smiles)
#     if feature is not None:
#         features.append(feature)
#
# # 转换为numpy数组
# features = np.array(features)
#
# # 获取原始数据集中的其他特征（从第二列到倒数第二列）
# additional_features = data.iloc[:, 1:-2].values
#
# # 合并所有特征
# all_features = np.hstack((features, additional_features))
#
# # 标准化特征
# scaler = StandardScaler()
# all_features = scaler.fit_transform(all_features)
#
# # 将数据集拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)
#
# # 特征名称列表
# feature_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in range(2048)] + list(data.columns[1:-2])
#
# # 加载Excel文档中各类别的特征重要性
# importance_data = pd.read_excel('Top_4_Features_Per_Category.xlsx')
#
# # 获取类别名称
# categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别
#
# # 创建输出目录
# output_dir_ice = 'Out_result_Flam_R_3D_PDP'
# os.makedirs(output_dir_ice, exist_ok=True)
#
# # 计算总的图像数量和行数
# total_plots = len(categories) * 2  # 每个类别2个图
# n_cols = 4  # 每行4个图
# n_rows = (total_plots + n_cols - 1) // n_cols  # 计算需要的行数
#
# # 创建一个大的图像面板，用于合并所有子图
# fig = plt.figure(figsize=(24, n_rows * 6))  # 设置总图的大小
#
# axes = []
#
# # 创建每个子图
# for i in range(total_plots):
#     if i % 2 == 0:  # 3D 图的子图
#         ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
#     else:  # 2D 图的子图
#         ax = fig.add_subplot(n_rows, n_cols, i + 1)
#     axes.append(ax)
#
# plot_index = 0
#
# # 针对每个类别生成3D和2D图
# for category in categories:
#     # 找到当前类别中最重要的两个特征
#     top_features = importance_data.nlargest(2, category)['Feature'].values  # 获取两个最重要的特征
#
#     # 获取特征在原始特征集中的索引
#     feature_indices = [feature_names.index(feature) for feature in top_features]
#
#     # 重新训练模型，针对每个类别进行单独的特征重要性评估
#     rf_model = RandomForestRegressor()
#     rf_model.fit(X_train, y_train)
#
#     # 生成3D部分依赖图
#     pd_results = partial_dependence(rf_model, X_train, features=feature_indices, grid_resolution=50)
#     XX, YY = np.meshgrid(pd_results['values'][0], pd_results['values'][1])
#     Z = pd_results['average'].reshape(XX.shape)
#
#     # 绘制3D图
#     ax_3d = axes[plot_index]
#     surf = ax_3d.plot_surface(XX, YY, Z, cmap='coolwarm')
#     cset = ax_3d.contour(XX, YY, Z, zdir='z', offset=np.min(Z), colors='black', linewidths=0.5)  # 在表面底部绘制
#     ax_3d.clabel(cset, fontsize=10, fmt="%.2f")  # 在等高线处标注数值
#     ax_3d.set_xlabel(f'{feature_names[feature_indices[0]]}')
#     ax_3d.set_ylabel(f'{feature_names[feature_indices[1]]}')
#     #ax_3d.set_zlabel('Partial Dependence')
#     cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, pad=0.1)  # 使用 pad 参数增加间距
#     cbar.set_label('Partial Dependence Value', fontsize=12, fontweight='bold')  # 设置颜色条标签
#     ax_3d.set_title(f'{category} in XGBoost\nfor reactivity')
#
#     # 绘制2D图
#     ax_2d = axes[plot_index + 1]
#     c = ax_2d.contourf(XX, YY, Z, cmap='coolwarm')
#     cset2 = ax_2d.contour(XX, YY, Z, colors='black', linewidths=0.5)  # 添加黑色等高线
#     ax_2d.clabel(cset2, fontsize=10, fmt="%.2f")  # 在等高线处标注数值
#     ax_2d.set_xlabel(f'{feature_names[feature_indices[0]]}')
#     ax_2d.set_ylabel(f'{feature_names[feature_indices[1]]}')
#     cbar=fig.colorbar(c, ax=ax_2d, shrink=0.4)
#     cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')
#     ax_2d.set_title(f'{category} in XGBoost \nfor reactivity')
#
#     plot_index += 2
#
# # 调整布局以适应所有子图
# plt.tight_layout()
# # 保存合并的图像
# plt.savefig(os.path.join(output_dir_ice, 'Combined_3D_2D_Plots.png'))
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import partial_dependence
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
#
# # 设置全局字体大小和粗细
# plt.rcParams['font.size'] = 14  # 设置全局字体大小
# plt.rcParams['font.weight'] = 'bold'  # 设置全局字体粗细
# plt.rcParams['axes.labelsize'] = 14  # 设置轴标签的字体大小
# plt.rcParams['axes.labelweight'] = 'bold'  # 设置轴标签的字体粗细
# plt.rcParams['axes.titlesize'] = 14  # 设置标题的字体大小
# plt.rcParams['axes.titleweight'] = 'bold'  # 设置标题的字体粗细
# plt.rcParams['xtick.labelsize'] = 12  # 设置x轴刻度标签的字体大小
# plt.rcParams['ytick.labelsize'] = 12  # 设置y轴刻度标签的字体大小
# plt.rcParams['legend.fontsize'] = 14  # 设置图例的字体大小
#
# # 加载原始数据
# file_path = 'imputed_selected_features_Reactivity.csv'  # 替换为你原始数据集的路径
# data = pd.read_csv(file_path)
#
# # 提取SMILES和标签
# smiles_list = data['SMILES'].tolist()
# labels = data['Reactivity'].values
#
# # 函数：将SMILES转换为分子描述符和指纹
# def smiles_to_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     # 提取描述符
#     descriptors = [
#         Descriptors.MolWt(mol),  # 分子量
#         Descriptors.MolLogP(mol),  # LogP
#         Descriptors.NumHDonors(mol),  # 氢键供体数量
#         Descriptors.NumHAcceptors(mol)  # 氢键受体数量
#     ]
#     # 生成Morgan指纹
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#     fingerprint_array = np.zeros((2048,))
#     Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
#     # 合并描述符和指纹
#     features = np.concatenate([descriptors, fingerprint_array])
#     return features
#
# # 将SMILES转换为特征
# features = []
# for smiles in smiles_list:
#     feature = smiles_to_features(smiles)
#     if feature is not None:
#         features.append(feature)
#
# # 转换为numpy数组
# features = np.array(features)
#
# # 获取原始数据集中的其他特征（从第二列到倒数第二列）
# additional_features = data.iloc[:, 1:-2].values
#
# # 合并所有特征
# all_features = np.hstack((features, additional_features))
#
# # 标准化特征
# scaler = StandardScaler()
# all_features = scaler.fit_transform(all_features)
#
# # 将数据集拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)
#
# # 特征名称列表
# feature_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + [f'Fingerprint_{i}' for i in range(2048)] + list(data.columns[1:-2])
#
# # 加载Excel文档中各类别的特征重要性
# importance_data = pd.read_excel('Top_4_Features_Per_Category.xlsx')
#
# # 获取类别名称
# categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别
#
# # 创建输出目录
# output_dir_ice = 'Out_result_Flam_R_3D_PDP'
# os.makedirs(output_dir_ice, exist_ok=True)
#
# # 确保目录存在的函数
# def ensure_directory_exists(file_path):
#     directory = os.path.dirname(file_path)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
# # 针对每个类别生成3D和2D图
# for category in categories:
#     # 找到当前类别中最重要的两个特征
#     top_features = importance_data.nlargest(2, category)['Feature'].values  # 获取两个最重要的特征
#
#     # 获取特征在原始特征集中的索引
#     feature_indices = [feature_names.index(feature) for feature in top_features]
#
#     # 重新训练模型，针对每个类别进行单独的特征重要性评估
#     rf_model = RandomForestRegressor()
#     rf_model.fit(X_train, y_train)
#
#     # 生成3D部分依赖图
#     pd_results = partial_dependence(rf_model, X_train, features=feature_indices, grid_resolution=50)
#     XX, YY = np.meshgrid(pd_results['values'][0], pd_results['values'][1])
#     Z = pd_results['average'].reshape(XX.shape)
#
#     # 创建并保存3D图
#     fig_3d = plt.figure(figsize=(8, 6))
#     ax_3d = fig_3d.add_subplot(111, projection='3d')
#     surf = ax_3d.plot_surface(XX, YY, Z, cmap='coolwarm')
#     cset = ax_3d.contour(XX, YY, Z, zdir='z', offset=np.min(Z), colors='black', linewidths=0.5)  # 在表面底部绘制
#     ax_3d.set_xlabel(f'{feature_names[feature_indices[0]]}')
#     ax_3d.set_ylabel(f'{feature_names[feature_indices[1]]}')
#     ax_3d.set_title(f'{category} in RF for Reactivity')
#     cbar = fig_3d.colorbar(surf, shrink=0.5, pad=0.1)
#     cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')
#
#     # 保存3D图
#     file_3d_path = os.path.join(output_dir_ice, f'{category}_3D_Partial_Dependence.png')
#     ensure_directory_exists(file_3d_path)
#     plt.savefig(file_3d_path)
#     plt.close(fig_3d)
#
#     # 创建并保存2D图
#     fig_2d = plt.figure(figsize=(8, 6))
#     ax_2d = fig_2d.add_subplot(111)
#     c = ax_2d.contourf(XX, YY, Z, cmap='coolwarm')
#     cset2 = ax_2d.contour(XX, YY, Z, colors='black', linewidths=0.5)  # 添加黑色等高线
#     ax_2d.clabel(cset2, fontsize=10, fmt="%.2f")  # 在等高线处标注数值
#     ax_2d.set_xlabel(f'{feature_names[feature_indices[0]]}')
#     ax_2d.set_ylabel(f'{feature_names[feature_indices[1]]}')
#     ax_2d.set_title(f'{category} in RF for Reactivity')
#     cbar = fig_2d.colorbar(c, shrink=0.4)
#     cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')
#
#     # 保存2D图
#     file_2d_path = os.path.join(output_dir_ice, f'{category}_2D_Partial_Dependence.png')
#     ensure_directory_exists(file_2d_path)
#     plt.savefig(file_2d_path)
#     plt.close(fig_2d)

#XGBoost
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xgboost import XGBRegressor  # 导入 XGBRegressor
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from matplotlib.ticker import FuncFormatter  # 导入科学计数法格式器

# 设置全局字体大小和粗细
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['font.weight'] = 'bold'  # 设置全局字体粗细
plt.rcParams['axes.labelsize'] = 14  # 设置轴标签的字体大小
plt.rcParams['axes.labelweight'] = 'bold'  # 设置轴标签的字体粗细
plt.rcParams['axes.titlesize'] = 14  # 设置标题的字体大小
plt.rcParams['axes.titleweight'] = 'bold'  # 设置标题的字体粗细
plt.rcParams['xtick.labelsize'] = 12  # 设置x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置y轴刻度标签的字体大小
plt.rcParams['legend.fontsize'] = 14  # 设置图例的字体大小

# 加载原始数据
file_path = 'imputed_selected_features_Reactivity.csv'  # 替换为你原始数据集的路径

# 检查文件是否存在
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"指定的文件路径不存在: {file_path}")

data = pd.read_csv(file_path)

# 提取SMILES和标签
smiles_list = data['SMILES'].tolist()
labels = data['Reactivity'].values

# 函数：将SMILES转换为分子描述符和指纹
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 提取描述符
    descriptors = [
        Descriptors.MolWt(mol),          # 分子量
        Descriptors.MolLogP(mol),        # LogP
        Descriptors.NumHDonors(mol),     # 氢键供体数量
        Descriptors.NumHAcceptors(mol)   # 氢键受体数量
    ]
    # 生成Morgan指纹
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint_array = np.zeros((2048,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    # 合并描述符和指纹
    features = np.concatenate([descriptors, fingerprint_array])
    return features

# 将SMILES转换为特征
features = []
for smiles in smiles_list:
    feature = smiles_to_features(smiles)
    if feature is not None:
        features.append(feature)

# 检查是否有有效的特征
if not features:
    raise ValueError("没有有效的特征被提取。请检查SMILES字符串和特征提取函数。")

# 转换为numpy数组
features = np.array(features)

# 获取原始数据集中的其他特征（从第二列到倒数第二列）
additional_features = data.iloc[:, 1:-2].values

# 合并所有特征
all_features = np.hstack((features, additional_features))

# 标准化特征
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

# 特征名称列表
feature_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors'] + \
                [f'Fingerprint_{i}' for i in range(2048)] + \
                list(data.columns[1:-2])

# 加载Excel文档中各类别的特征重要性
importance_data = pd.read_excel('Top_4_Features_Per_Category.xlsx')

# 获取类别名称
categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别

# 创建输出目录
output_dir_ice = 'Out_result_reactivity_3D_PDP'
os.makedirs(output_dir_ice, exist_ok=True)

# 确保目录存在的函数
def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 自定义科学计数法格式
def scientific_formatter(x, pos):
    return f'{x:.3e}'

# 针对每个类别生成3D和2D图
for category in categories:
    print(f"正在处理类别: {category}")
    # 找到当前类别中最重要的两个特征
    top_features = importance_data.nlargest(2, category)['Feature'].values  # 获取两个最重要的特征

    # 获取特征在原始特征集中的索引
    try:
        feature_indices = [feature_names.index(feature) for feature in top_features]
    except ValueError as e:
        print(f"特征名称错误: {e}")
        continue  # 跳过当前类别

    # 重新训练模型，针对每个类别进行单独的特征重要性评估
    rf_model = XGBRegressor()
    rf_model.fit(X_train, y_train)

    # 生成3D部分依赖图
    pd_results = partial_dependence(rf_model, X_train, features=feature_indices, grid_resolution=50)
    XX, YY = np.meshgrid(pd_results['values'][0], pd_results['values'][1])
    Z = pd_results['average'].reshape(XX.shape)

    # 绘制并保存3D图
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    surf = ax_3d.plot_surface(XX, YY, Z, cmap='coolwarm', edgecolor='none')
    ax_3d.contour(XX, YY, Z, zdir='z', offset=np.min(Z), colors='black', linewidths=0.5)  # 在表面底部绘制等高线
    ax_3d.set_xlabel(f'{feature_names[feature_indices[0]]}')
    ax_3d.set_ylabel(f'{feature_names[feature_indices[1]]}')
    ax_3d.set_title(f'{category} in XGBoost\nfor Reactivity')
    cbar = fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, pad=0.1)
    cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')
    cbar.formatter = FuncFormatter(scientific_formatter)  # 应用科学计数法格式
    cbar.update_ticks()

    # 保存3D图，确保路径存在
    file_3d_path = os.path.join(output_dir_ice, f'{category}_3D_Partial_Dependence.png')
    ensure_directory_exists(file_3d_path)  # 确保目录存在
    plt.savefig(file_3d_path)
    plt.close(fig_3d)
    print(f"已保存3D图: {file_3d_path}")

    # 绘制并保存2D图
    fig_2d = plt.figure(figsize=(8, 6))
    ax_2d = fig_2d.add_subplot(111)
    c = ax_2d.contourf(XX, YY, Z, cmap='coolwarm')
    cset2 = ax_2d.contour(XX, YY, Z, colors='black', linewidths=0.5)  # 添加黑色等高线
    ax_2d.clabel(cset2, fontsize=10, fmt="%.2f")  # 在等高线处标注数值
    ax_2d.set_xlabel(f'{feature_names[feature_indices[0]]}')
    ax_2d.set_ylabel(f'{feature_names[feature_indices[1]]}')
    ax_2d.set_title(f'{category} in XGBoost\nfor Reactivity')
    cbar = fig_2d.colorbar(c, ax=ax_2d, shrink=0.4)
    cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')
    cbar.formatter = FuncFormatter(scientific_formatter)  # 应用科学计数法格式
    cbar.update_ticks()

    # 保存2D图，确保路径存在
    file_2d_path = os.path.join(output_dir_ice, f'{category}_2D_Partial_Dependence.png')
    ensure_directory_exists(file_2d_path)  # 确保目录存在
    plt.savefig(file_2d_path)
    plt.close(fig_2d)
    print(f"已保存2D图: {file_2d_path}")

print("所有图像已生成并保存。")
