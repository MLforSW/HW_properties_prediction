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
'''
# 加载数据
file_path = 'imputed_selected_features_W.csv'
data = pd.read_csv(file_path)

# 提取SMILES和标签
smiles_list = data['SMILES'].tolist()
labels = data['W'].values

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

# 获取CSV文件中的其他特征（从第二列到倒数第二列）
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

# 将特征名称应用到 DataFrame
X_train_df = pd.DataFrame(X_train, columns=feature_names)
classifications = data['Classification'].values
X_train_df['Classification'] = classifications[:X_train.shape[0]]

# 函数：清理文件名中的无效字符
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# 确保输出目录存在
output_dir_ice = 'Out_result_Flam_WR_ICE'
os.makedirs(output_dir_ice, exist_ok=True)

# 创建一个大的图像面板，用于合并所有子图
fig, axes = plt.subplots(5, 4, figsize=(24, 30))  # 5行4列，总共20个图
axes = axes.flatten()  # 将axes数组展平，方便索引

plot_index = 0
# 添加全局字体设置，确保字体大小和粗细应用于整个图形
plt.rcParams['axes.labelsize'] = 14  # 设置x轴和y轴标签的字体大小
plt.rcParams['axes.labelweight'] = 'bold'  # 设置x轴和y轴标签的字体粗细
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
# 全局设置线条粗细
plt.rcParams['lines.linewidth'] = 2  # 设置线条粗细，默认值为1.5，你可以根据需要调整这个值
plt.rcParams['lines.color'] = 'blue'  # 设置线条颜色（可选）

# 针对每个分类生成 ICE 图
for class_name in X_train_df['Classification'].unique():
    # 生成布尔索引，提取当前分类的训练数据
    class_indices = X_train_df['Classification'] == class_name
    X_train_class = X_train[class_indices]

    # 重新训练模型，针对每个分类进行单独的特征重要性评估
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_class, y_train[class_indices])

    # 计算当前分类的特征重要性
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-2:]  # 找到最重要的两个特征索引

    # 为每个最重要的特征生成 ICE 图
    for i in indices:
        feature_name = feature_names[i]  # 使用实际列名
        print(f"Generating ICE Plot for {feature_name} in class {class_name}")

        # 使用 PartialDependenceDisplay 生成 ICE 图
        display = PartialDependenceDisplay.from_estimator(
            rf_model, X_train_class, [i], kind="both", ax=axes[plot_index], grid_resolution=50,
            feature_names=feature_names
        )

        # 手动设置 X 和 Y 轴标签字体大小和粗细
        axes[plot_index].set_xlabel(feature_name, fontsize=18, fontweight='bold')
        axes[plot_index].set_ylabel("Partial dependence", fontsize=18, fontweight='bold')
        axes[plot_index].tick_params(axis='both', which='major', labelsize=16, width=2)

        # 强制修改x轴和y轴标签字体大小和粗细
        axes[plot_index].xaxis.label.set_size(14)
        axes[plot_index].xaxis.label.set_weight('bold')
        axes[plot_index].yaxis.label.set_size(14)
        axes[plot_index].yaxis.label.set_weight('bold')

        # 加粗线条
        for line in axes[plot_index].lines:
            line.set_linewidth(2)

        # 手动设置标题的字体大小和粗细
        axes[plot_index].set_title(f"ICE Plot for {feature_name} in  {class_name}", fontsize=16, fontweight='bold')

        plot_index += 1

# 强制所有 X 和 Y 轴标签的字体加粗
for ax in axes:
    ax.xaxis.label.set_size(14)
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_size(14)
    ax.yaxis.label.set_weight('bold')
    ax.title.set_size(16)
    ax.title.set_weight('bold')

# 调整布局以适应所有子图
plt.tight_layout()
# 保存合并的图像
plt.savefig(os.path.join(output_dir_ice, 'Combined_ICE_Plots_WR_Larger_Fonts.png'))
'''


#制定feature
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

# # 加载原始数据
# file_path = 'imputed_selected_features_W.csv'  # 替换为你原始数据集的路径
# data = pd.read_csv(file_path)
#
# # 提取SMILES和标签
# smiles_list = data['SMILES'].tolist()
# labels = data['W'].values
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
# # 加载CSV文档中各类别的特征重要性
# importance_data = pd.read_excel('Top_4_Features_Per_Category_WR.xlsx')
#
# # 获取类别名称
# categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别
#
# # 创建输出目录
# output_dir_ice = 'Out_result_Flam_WR_ICE_re'
# os.makedirs(output_dir_ice, exist_ok=True)
#
# # 计算总的图像数量和行数
# total_plots = len(categories) * 2  # 每个类别2个图
# n_cols = 4  # 每行4个图
# n_rows = (total_plots + n_cols - 1) // n_cols  # 计算需要的行数
#
# # 创建一个大的图像面板，用于合并所有子图
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 6))  # 每行4个图，总共n_rows行
# axes = axes.flatten()
#
# plot_index = 0
# # 添加全局字体设置，确保字体大小和粗细应用于整个图形
# plt.rcParams['axes.labelsize'] = 14  # 设置x轴和y轴标签的字体大小
# plt.rcParams['axes.labelweight'] = 'bold'  # 设置x轴和y轴标签的字体粗细
# plt.rcParams['font.size'] = 14
# plt.rcParams['font.weight'] = 'bold'
# # 全局设置线条粗细
# plt.rcParams['lines.linewidth'] = 2  # 设置线条粗细，默认值为1.5，你可以根据需要调整这个值
# plt.rcParams['lines.color'] = 'blue'  # 设置线条颜色（可选）
# # 针对每个类别生成 ICE 图
# for category in categories:
#     # 找到当前类别中最重要的两个特征
#     top_features = importance_data.nlargest(2, category)['Feature'].values  # 获取两个最重要的特征
#
#     # 获取特征在原始特征集中的索引
#     feature_indices = [feature_names.index(feature) for feature in top_features]
#
#     # 重新训练模型，针对每个类别进行单独的特征重要性评估
#     rf_model = RandomForestClassifier()
#     rf_model.fit(X_train, y_train)
#
#     # 为每个最重要的特征生成 ICE 图
#     for i in feature_indices:
#         feature_name = feature_names[i]
#         print(f"Generating ICE Plot for {feature_name} in category {category}")
#
#         # 使用 PartialDependenceDisplay 生成 ICE 图
#         display = PartialDependenceDisplay.from_estimator(
#             rf_model, X_train, [i], kind="both", ax=axes[plot_index], grid_resolution=50,
#             feature_names=feature_names
#         )
#
#         # 设置X轴和Y轴标签字体大小和粗细
#         axes[plot_index].set_xlabel(feature_name, fontsize=14, fontweight='bold')
#         axes[plot_index].set_ylabel("Partial dependence", fontsize=14, fontweight='bold')
#
#         # 设置标题
#         axes[plot_index].set_title(f"ICE Plot for {feature_name} in {category}", fontsize=16, fontweight='bold')
#
#         plot_index += 1
#
# # 调整布局以适应所有子图
# plt.tight_layout()
# # 保存合并的图像
# plt.savefig(os.path.join(output_dir_ice, 'Combined_ICE_Plots.png'))



# from xgboost import XGBClassifier
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
# file_path = 'imputed_selected_features_W.csv'  # 替换为你原始数据集的路径
# data = pd.read_csv(file_path)
#
# # 提取SMILES和标签
# smiles_list = data['SMILES'].tolist()
# labels = data['W'].values
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
# importance_data = pd.read_excel('Top_4_Features_Per_Category_WR.xlsx')
#
# # 获取类别名称
# categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别
#
# # 创建输出目录
# output_dir_ice = 'Out_result_W_3D_PDP'
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
#     xgb_model = XGBClassifier()
#     xgb_model.fit(X_train, y_train)
#
#     # 生成3D部分依赖图
#     pd_results = partial_dependence(xgb_model, X_train, features=feature_indices, grid_resolution=50)
#     XX, YY = np.meshgrid(pd_results['values'][0], pd_results['values'][1])
#     Z = pd_results['average'].reshape(XX.shape)
#
#     # 绘制3D图
#     ax_3d = axes[plot_index]
#     surf = ax_3d.plot_surface(XX, YY, Z, cmap='coolwarm')
#     ax_3d.contour(XX, YY, Z)  # 在表面底部绘制
#     #ax_3d.clabel(cset, fontsize=10, fmt="%.2f")  # 在等高线处标注数值
#     ax_3d.set_xlabel(f'{feature_names[feature_indices[0]]}')
#     ax_3d.set_ylabel(f'{feature_names[feature_indices[1]]}')
#     #ax_3d.set_zlabel('Partial Dependence')
#     cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, pad=0.1)  # 使用 pad 参数增加间距
#     cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')  # 设置颜色条标签
#     ax_3d.set_title(f'{category} in RF\nfor WR')
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
#     ax_2d.set_title(f'{category} in RF\nfor WR')
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
# from xgboost import XGBClassifier
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
# file_path = 'imputed_selected_features_W.csv'  # 替换为你原始数据集的路径
# data = pd.read_csv(file_path)
#
# # 提取SMILES和标签
# smiles_list = data['SMILES'].tolist()
# labels = data['W'].values
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
# importance_data = pd.read_excel('Top_4_Features_Per_Category_WR.xlsx')
#
# # 获取类别名称
# categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别
#
# # 创建输出目录
# output_dir_ice = 'Out_result_W_3D_PDP'
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
#     xgb_model = XGBClassifier()
#     xgb_model.fit(X_train, y_train)
#
#     # 生成3D部分依赖图
#     pd_results = partial_dependence(xgb_model, X_train, features=feature_indices, grid_resolution=50)
#     XX, YY = np.meshgrid(pd_results['values'][0], pd_results['values'][1])
#     Z = pd_results['average'].reshape(XX.shape)
#
#     # 创建并保存3D图
#     fig_3d = plt.figure(figsize=(8, 6))
#     ax_3d = fig_3d.add_subplot(111, projection='3d')
#     surf = ax_3d.plot_surface(XX, YY, Z, cmap='coolwarm')
#     ax_3d.contour(XX, YY, Z)  # 在表面底部绘制等高线
#     ax_3d.set_xlabel(f'{feature_names[feature_indices[0]]}')
#     ax_3d.set_ylabel(f'{feature_names[feature_indices[1]]}')
#     ax_3d.set_title(f'{category} in XGBoost for RW')
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
#     ax_2d.set_title(f'{category} in XGBoost for RW')
#     cbar = fig_2d.colorbar(c, shrink=0.4)
#     cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')
#
#     # 保存2D图
#     file_2d_path = os.path.join(output_dir_ice, f'{category}_2D_Partial_Dependence.png')
#     ensure_directory_exists(file_2d_path)
#     plt.savefig(file_2d_path)
#     plt.close(fig_2d)


#RF
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

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
file_path = 'imputed_selected_features_W.csv'  # 替换为你原始数据集的路径
data = pd.read_csv(file_path)

# 提取SMILES和标签
smiles_list = data['SMILES'].tolist()
labels = data['W'].values

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

# 加载Excel文档中各类别的特征重要性
importance_data = pd.read_excel('Top_4_Features_Per_Category_WR.xlsx')

# 获取类别名称
categories = importance_data.columns[1:]  # 第一列是特征名称，后面的列是各个类别

# 创建输出目录
output_dir_ice = 'Out_result_W_3D_PDP'
os.makedirs(output_dir_ice, exist_ok=True)

# 确保目录存在的函数
def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 针对每个类别生成3D和2D图
for category in categories:
    # 找到当前类别中最重要的两个特征
    top_features = importance_data.nlargest(2, category)['Feature'].values  # 获取两个最重要的特征

    # 获取特征在原始特征集中的索引
    feature_indices = [feature_names.index(feature) for feature in top_features]

    # 重新训练模型，针对每个类别进行单独的特征重要性评估
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # 生成3D部分依赖图
    pd_results = partial_dependence(rf_model, X_train, features=feature_indices, grid_resolution=50)
    XX, YY = np.meshgrid(pd_results['values'][0], pd_results['values'][1])
    Z = pd_results['average'].reshape(XX.shape)

    # 创建并保存3D图
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    surf = ax_3d.plot_surface(XX, YY, Z, cmap='coolwarm')
    ax_3d.contour(XX, YY, Z)  # 在表面底部绘制等高线
    ax_3d.set_xlabel(f'{feature_names[feature_indices[0]]}')
    ax_3d.set_ylabel(f'{feature_names[feature_indices[1]]}')
    ax_3d.set_title(f'{category} in RF for RW')
    cbar = fig_3d.colorbar(surf, shrink=0.5, pad=0.1)
    cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')

    # 保存3D图
    file_3d_path = os.path.join(output_dir_ice, f'{category}_3D_Partial_Dependence.png')
    ensure_directory_exists(file_3d_path)
    plt.savefig(file_3d_path)
    plt.close(fig_3d)

    # 创建并保存2D图
    fig_2d = plt.figure(figsize=(8, 6))
    ax_2d = fig_2d.add_subplot(111)
    c = ax_2d.contourf(XX, YY, Z, cmap='coolwarm')
    cset2 = ax_2d.contour(XX, YY, Z, colors='black', linewidths=0.5)  # 添加黑色等高线
    ax_2d.clabel(cset2, fontsize=10, fmt="%.2f")  # 在等高线处标注数值
    ax_2d.set_xlabel(f'{feature_names[feature_indices[0]]}')
    ax_2d.set_ylabel(f'{feature_names[feature_indices[1]]}')
    ax_2d.set_title(f'{category} in RF for RW')
    cbar = fig_2d.colorbar(c, shrink=0.4)
    cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')

    # 保存2D图
    file_2d_path = os.path.join(output_dir_ice, f'{category}_2D_Partial_Dependence.png')
    ensure_directory_exists(file_2d_path)
    plt.savefig(file_2d_path)
    plt.close(fig_2d)
