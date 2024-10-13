import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# # 文件路径列表
# file_paths = [
#     'SHAP_values_train_Amine_F.csv',
#     'SHAP_values_train_Aromatic_F.csv',
#     'SHAP_values_train_Ether_F.csv',
#     'SHAP_values_train_Halogen_F.csv',
#     'SHAP_values_train_Hydroxyl_F.csv',
#     'SHAP_values_train_Ketone_Aldehyde_F.csv',
#     'SHAP_values_train_Nitrile_F.csv',
#     'SHAP_values_train_Other_F.csv',
#     'SHAP_values_train_Thioether_F.csv',
#     'SHAP_values_train_Thiol_F.csv'
# ]
#
# # 类别标签
# labels = [
#     'Amine', 'Aromatic', 'Ether', 'Halogen', 'Hydroxyl',
#     'Ketone_Aldehyde', 'Nitrile', 'Other', 'Thioether', 'Thiol'
# ]
#
# # 创建一个数据字典来存储所有类别的数据
# shap_data = {}
# for label, file_path in zip(labels, file_paths):
#     shap_data[label] = pd.read_csv(file_path)
#
# # 计算每个特征的绝对平均SHAP值
# feature_means = {label: data.abs().mean() for label, data in shap_data.items()}
#
# # 转换为DataFrame
# feature_means_df = pd.DataFrame(feature_means)
#
# # 删除所有类别中SHAP值均为0的特征
# feature_means_df = feature_means_df.loc[(feature_means_df != 0).any(axis=1)]
#
# # 设置输出目录
# output_dir = 'SHAP_output_plots'
# os.makedirs(output_dir, exist_ok=True)
#
# # 保存合并后的数据到Excel
# excel_output_path = os.path.join(output_dir, 'Combined_SHAP_Values.xlsx')
# feature_means_df.to_excel(excel_output_path, index=True)
#
# # 绘制热图并保存
# plt.figure(figsize=(12, 8))
# sns.heatmap(feature_means_df, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Average SHAP Values for Each Feature Across Different Categories')
# plt.xlabel('Category')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'Average_SHAP_Values_Heatmap.png'))
# plt.close()
#
# # 生成并保存每个类别的特征重要性条形图
# for label, data in shap_data.items():
#     # 计算特征重要性
#     feature_importance = data.abs().mean()
#
#     # 删除所有类别中SHAP值均为0的特征
#     feature_importance = feature_importance[feature_importance.index.isin(feature_means_df.index)]
#
#     feature_importance = feature_importance.sort_values(ascending=False)
#
#     plt.figure(figsize=(10, 6))
#     feature_importance.plot(kind='bar')
#     plt.title(f'SHAP Feature Importance for {label}')
#     plt.xlabel('Feature')
#     plt.ylabel('Mean |SHAP Value|')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'SHAP_Feature_Importance_{label}.png'))
#     plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 文件路径列表
file_paths = [
    'SHAP_values_train_Amine_WR.csv',
    'SHAP_values_train_Aromatic_WR.csv',
    'SHAP_values_train_Ether_WR.csv',
    'SHAP_values_train_Halogen_WR.csv',
    'SHAP_values_train_Hydroxyl_WR.csv',
    'SHAP_values_train_Ketone_Aldehyde_WR.csv',
    'SHAP_values_train_Nitrile_WR.csv',
    'SHAP_values_train_Other_WR.csv',
    'SHAP_values_train_Thioether_WR.csv',
    'SHAP_values_train_Thiol_WR.csv'
]

# 类别标签
labels = [
    'Amine', 'Aromatic', 'Ether', 'Halogen', 'Hydroxyl',
    'Ketone_Aldehyde', 'Nitrile', 'Other', 'Thioether', 'Thiol'
]

# 创建一个数据字典来存储所有类别的数据
shap_data = {}
for label, file_path in zip(labels, file_paths):
    shap_data[label] = pd.read_csv(file_path)

# 计算每个特征的绝对平均SHAP值
feature_means = {label: data.abs().mean() for label, data in shap_data.items()}

# 转换为DataFrame
feature_means_df = pd.DataFrame(feature_means)

# 删除所有类别中SHAP值均为0的特征
feature_means_df = feature_means_df.loc[(feature_means_df != 0).any(axis=1)]

# 设置输出目录
output_dir = 'SHAP_output_plots'
os.makedirs(output_dir, exist_ok=True)

# 保存合并后的数据到Excel
excel_output_path = os.path.join(output_dir, 'Combined_SHAP_Values.xlsx')
feature_means_df.to_excel(excel_output_path, index=True)

# 创建一个DataFrame来存储每个类别数值最大的前四个特征
top_features_df = pd.DataFrame()

# 找出每个类别中绝对平均SHAP值最大的前四个特征
for label in labels:
    top_features = feature_means_df[label].nlargest(4).reset_index()
    top_features.columns = ['Feature', f'Top 4 SHAP Values for {label}']
    if top_features_df.empty:
        top_features_df = top_features
    else:
        top_features_df = top_features_df.merge(top_features, on='Feature', how='outer')

# 保存前四个特征到Excel
top_features_excel_path = os.path.join(output_dir, 'Top_4_Features_Per_Category_WR.xlsx')
top_features_df.to_excel(top_features_excel_path, index=False)

# 绘制热图并保存
plt.figure(figsize=(12, 8))
sns.heatmap(feature_means_df, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Average SHAP Values for Each Feature Across Different Categories')
plt.xlabel('Category')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Average_SHAP_Values_Heatmap_4.png'))
plt.close()

# 生成并保存每个类别的特征重要性条形图
for label, data in shap_data.items():
    # 计算特征重要性
    feature_importance = data.abs().mean()

    # 删除所有类别中SHAP值均为0的特征
    feature_importance = feature_importance[feature_importance.index.isin(feature_means_df.index)]

    feature_importance = feature_importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title(f'SHAP Feature Importance for {label}')
    plt.xlabel('Feature')
    plt.ylabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'SHAP_Feature_Importance_{label}_4_WR.png'))
    plt.close()
