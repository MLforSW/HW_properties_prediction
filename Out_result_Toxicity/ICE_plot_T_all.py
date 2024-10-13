import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os
from sklearn.model_selection import train_test_split
# 加载原始数据
file_path = 'imputed_selected_features_Toxcity.csv'  # 替换为你原始数据集的路径
data = pd.read_csv(file_path)

# 提取SMILES和标签
smiles_list = data['SMILES'].tolist()
labels = data['Toxicity'].values

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

# 你提供的前四个最重要特征（假设是前四个）
top_features = ["MorganFingerprints", "SpMax5_Bhi", "ATS8v", "maxdsCH"]  # 替换为实际的特征名称
# 重新训练模型以获取特征重要性
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# 获取模型的特征重要性
importances = xgb_model.feature_importances_

# 如果MorganFingerprints在列表中，找出最重要的一个指纹特征
if "MorganFingerprints" in top_features:
    top_features.remove("MorganFingerprints")
    fingerprint_importances = importances[4:2052]  # 前四个特征是描述符，后面2048个是指纹
    most_important_fingerprint_index = np.argmax(fingerprint_importances)
    top_features.append(f'Fingerprint_{most_important_fingerprint_index}')

# 获取特征在原始特征集中的索引
feature_indices = [feature_names.index(feature) for feature in top_features]

# 创建输出目录
output_dir_ice = 'Out_result_Toxicity_all'
os.makedirs(output_dir_ice, exist_ok=True)

# 计算总的图像数量和行数
total_plots = len(feature_indices)
n_cols = 4  # 每行2个图
n_rows = (total_plots + n_cols - 1) // n_cols  # 计算需要的行数

# 创建一个大的图像面板，用于合并所有子图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 6))  # 每行2个图，总共n_rows行
axes = axes.flatten()

plot_index = 0
# 添加全局字体设置，确保字体大小和粗细应用于整个图形
plt.rcParams['axes.labelsize'] = 14  # 设置x轴和y轴标签的字体大小
plt.rcParams['axes.labelweight'] = 'bold'  # 设置x轴和y轴标签的字体粗细
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
# 全局设置线条粗细
plt.rcParams['lines.linewidth'] = 2  # 设置线条粗细，默认值为1.5，你可以根据需要调整这个值

# 针对每个最重要的特征生成 ICE 图
for i in feature_indices:
    feature_name = feature_names[i]
    print(f"Generating ICE Plot for {feature_name}")

    # 使用 PartialDependenceDisplay 生成 ICE 图
    display = PartialDependenceDisplay.from_estimator(
        xgb_model, X_train, [i], kind="both", ax=axes[plot_index], grid_resolution=50,
        feature_names=feature_names
    )

    # 设置X轴和Y轴标签字体大小和粗细
    axes[plot_index].set_xlabel(feature_name, fontsize=14, fontweight='bold')
    axes[plot_index].set_ylabel("Partial dependence", fontsize=14, fontweight='bold')

    # 设置标题
    axes[plot_index].set_title(f"Toxicity in XGBoost", fontsize=14, fontweight='bold')

    plot_index += 1

# 调整布局以适应所有子图
plt.tight_layout()
# 保存合并的图像
plt.savefig(os.path.join(output_dir_ice, 'Combined_ICE_Plots_T_all.png'))