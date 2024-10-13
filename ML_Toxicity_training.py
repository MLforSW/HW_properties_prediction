# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 加载数据
# file_path = 'imputed_selected_features_Toxcity.csv'
# data = pd.read_csv(file_path)
#
# # 提取SMILES和标签
# smiles_list = data['SMILES'].tolist()
# labels = data['Toxicity'].values
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
# # 获取CSV文件中的其他特征（从第二列到倒数第二列）
# additional_features = data.iloc[:, 1:-1].values
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
# # 定义模型字典
# models = {
#     'RandomForest': RandomForestClassifier(),
#     'KNN': KNeighborsClassifier(),
#     'SVM': SVC(probability=True),
#     'DecisionTree': DecisionTreeClassifier(),
#     'XGBoost': XGBClassifier(eval_metric='logloss')
# }
#
# # 存储评估指标的字典
# results = {
#     'Model': [],
#     'Accuracy': [],
#     'Precision': [],
#     'Recall': [],
#     'F1-Score': [],
#     'ROC-AUC': []
# }
#
# # 遍历模型字典，训练并评估每个模型
# for model_name, model in models.items():
#     # 训练模型
#     model.fit(X_train, y_train)
#
#     # 预测概率
#     y_prob = model.predict_proba(X_test)[:, 1]
#
#     # 预测标签
#     y_pred = model.predict(X_test)
#
#     # 计算评估指标
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob)
#
#     # 存储结果
#     results['Model'].append(model_name)
#     results['Accuracy'].append(accuracy)
#     results['Precision'].append(precision)
#     results['Recall'].append(recall)
#     results['F1-Score'].append(f1)
#     results['ROC-AUC'].append(roc_auc)
#
#     # 打印模型评估结果
#     print(f"Model: {model_name}")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#     print(f"ROC-AUC: {roc_auc:.4f}")
#     print()
#
#     # 绘制混淆矩阵
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix for {model_name}')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.show()
#
# # 将评估结果保存到Excel文件
# results_df = pd.DataFrame(results)
# results_df.to_excel('model_comparison_results_ml.xlsx', index=False)
# print("Model evaluation metrics saved to model_comparison_results.xlsx")

# # 获取特征名称
# molecular_descriptor_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']
# fingerprint_names = [f'Fingerprint_{i}' for i in range(2048)]
# additional_feature_names = data.columns[1:-1]
# feature_names = molecular_descriptor_names + fingerprint_names + list(additional_feature_names)
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 从训练好的 XGBoost 模型中获取特征重要性
# xgb_model = models['XGBoost']
# feature_importances = xgb_model.feature_importances_
#
# # 获取特征名称
# molecular_descriptor_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']
# fingerprint_names = [f'Fingerprint_{i}' for i in range(2048)]
# additional_feature_names = data.columns[1:-1]
#
# # 合并特征名称，将所有 Fingerprint 合并为一个
# feature_names = molecular_descriptor_names + ['MorganFingerprints'] + list(additional_feature_names)
#
# # 合并 Morgan 指纹特征的重要性为一个整体特征
# molecular_descriptor_importance = feature_importances[:4]
# fingerprint_importance = np.sum(feature_importances[4:2052])
# additional_feature_importance = feature_importances[2052:]
#
# # 将合并后的重要性和其他特征的重要性整合为一个列表
# combined_feature_importances = np.concatenate((molecular_descriptor_importance, [fingerprint_importance], additional_feature_importance))
#
# # 对特征重要性进行排序
# sorted_indices = np.argsort(combined_feature_importances)
# sorted_feature_importances = combined_feature_importances[sorted_indices]
# sorted_feature_names = np.array(feature_names)[sorted_indices]
#
# # 保存排序后的特征重要性到Excel文件
# importance_df = pd.DataFrame({
#     'Feature': sorted_feature_names,
#     'Importance': sorted_feature_importances
# })
# importance_df.to_excel('sorted_feature_importances_toxicity.xlsx', index=False)
# print("Sorted feature importances saved to sorted_feature_importances.xlsx")
#
# # 定义马卡龙配色
# macaron_colors = ['#FFB6C1', '#FFDAB9', '#E6E6FA', '#FFFACD', '#E0FFFF', '#D8BFD8', '#FFDEAD', '#F5DEB3', '#AFEEEE', '#D3FFCE']
#
# # 绘制合并后的特征重要性图
# plt.figure(figsize=(12, 15))  # 增加图形的高度
# plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, color=macaron_colors[:len(sorted_feature_importances)], tick_label=sorted_feature_names)
# plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
# plt.ylabel('Feature', fontsize=14, fontweight='bold')
# plt.title('Feature Importance for toxicity in XGBoost', fontsize=16, fontweight='bold')
#
# # 调整Y轴标签，放大字体并旋转标签
# plt.yticks(fontsize=10, fontweight='bold')
# plt.xticks(fontsize=10, fontweight='bold')
# # 调整图的左边距，确保所有标签都在图形范围内
# plt.gca().margins(y=0.01)
# plt.subplots_adjust(left=0.25)
#
# # 先保存图形
# plt.savefig("feature_importance_in_XGBoost_combined_sorted.png", bbox_inches='tight')
#
# # 再显示图形
# plt.show()

#Hazardou materials test
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import joblib
from sklearn.preprocessing import StandardScaler

# 加载 HW_list 数据
hw_list_data = pd.read_excel('HW_list.xlsx')

# 加载训练集以获取训练时使用的特征
training_data = pd.read_csv('imputed_selected_features_Toxcity.csv')

# 提取除去 'SMILES'、'Toxicity' 和 'Classification' 的特征
training_features = training_data.columns.difference(['SMILES', 'Toxicity', 'Classification'])

# 提取 HW_list 中的匹配特征
matching_features = [feature for feature in training_features if feature in hw_list_data.columns]
hw_matching_features = hw_list_data[matching_features]

# 检查是否有缺失的特征
missing_features = [feature for feature in training_features if feature not in hw_list_data.columns]
if missing_features:
    print(f"以下特征在 HW_list.xlsx 中缺失: {missing_features}")

# 函数：将 SMILES 转换为分子描述符和指纹
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 提取分子描述符
    descriptors = [
        Descriptors.MolWt(mol),  # 分子量
        Descriptors.MolLogP(mol),  # LogP
        Descriptors.NumHDonors(mol),  # 氢键供体数量
        Descriptors.NumHAcceptors(mol)  # 氢键受体数量
    ]
    # 生成 Morgan 指纹
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint_array = np.zeros((2048,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    # 合并描述符和指纹
    features = np.concatenate([descriptors, fingerprint_array])
    return features

# 处理 HW_list 中的 SMILES 并生成对应的分子描述符和指纹
hw_smiles_list = hw_list_data['SMILES'].tolist()
hw_smiles_features = []
for smiles in hw_smiles_list:
    feature = smiles_to_features(smiles)
    if feature is not None:
        hw_smiles_features.append(feature)

# 转换为 numpy 数组
hw_smiles_features = np.array(hw_smiles_features)

# 检查 SMILES 特征是否正确生成
print(f"SMILES 特征数量: {hw_smiles_features.shape[1]} (应为 2052，包含分子描述符和指纹)")

# 合并分子特征和 HW_list.xlsx 中的匹配特征
hw_all_features = np.hstack((hw_smiles_features, hw_matching_features.values))

# 检查合并后特征数量
print(f"合并后特征数量: {hw_all_features.shape[1]} (应为 2113)")

# 加载保存的 StandardScaler 和模型
scaler = joblib.load('scaler.pkl')
best_model = joblib.load('XGBoost_best_model.pkl')

# 对所有特征进行标准化
try:
    hw_features_scaled = scaler.transform(hw_all_features)
except ValueError as e:
    print(f"标准化错误: {e}")
    print(f"当前特征数: {hw_all_features.shape[1]}，期望特征数: {scaler.n_features_in_}")

# 使用保存的模型进行毒性预测
hw_toxicity_predictions = best_model.predict(hw_features_scaled)

# 输出预测结果
print("毒性预测结果：")
print(hw_toxicity_predictions)

# 如果需要将预测结果添加到原数据中并保存
hw_list_data['Toxicity_Prediction'] = hw_toxicity_predictions
hw_list_data.to_excel('HW_list_with_predictions.xlsx', index=False)
print("预测结果已保存到 HW_list_with_predictions.xlsx")
