import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'imputed_selected_features_Toxcity.csv'
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
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
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
additional_features = data.iloc[:, 1:-1].values

# 合并所有特征
all_features = np.hstack((features, additional_features))

# 标准化特征
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

# 定义模型字典
models = {
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# 存储评估指标的字典
results = {
    'Model': [],
    'ED Threshold': [],
    'MFS Threshold': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': []
}

# 计算训练集的质心
centroid = X_train.mean(axis=0)

# 计算训练集和测试集的分子指纹
train_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2) for smiles in
                      data['SMILES'].iloc[X_train.index]]
test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2) for smiles in
                     data['SMILES'].iloc[X_test.index]]

# 遍历不同的ED和MFS阈值
for ed_threshold in np.linspace(0, 1, 11):  # ED阈值在0到1之间，步长为0.1
    for similarity_threshold in np.linspace(0, 1, 11):  # MFS阈值在0到1之间，步长为0.1
        # 计算每个测试样本与训练集质心的欧氏距离
        distances = cdist(X_test, centroid.reshape(1, -1), metric='euclidean').flatten()
        # 判断样本是否在ED适用域内
        in_ad_ed = distances < ed_threshold

        # 计算每个测试样本与训练集中样本的Tanimoto相似性
        similarities = []
        for test_fp in test_fingerprints:
            sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fingerprints)
            similarities.append(max(sims))
        # 判断样本是否在MFS适用域内
        in_ad_similarity = np.array(similarities) > similarity_threshold

        # 综合判断
        final_in_ad = in_ad_ed & in_ad_similarity

        # 遍历模型字典，训练并评估每个模型
        for model_name, model in models.items():
            # 训练模型
            model.fit(X_train, y_train)

            # 预测概率
            y_prob = model.predict_proba(X_test[final_in_ad])[:, 1]

            # 预测标签
            y_pred = model.predict(X_test[final_in_ad])

            # 计算评估指标
            accuracy = accuracy_score(y_test[final_in_ad], y_pred)
            precision = precision_score(y_test[final_in_ad], y_pred)
            recall = recall_score(y_test[final_in_ad], y_pred)
            f1 = f1_score(y_test[final_in_ad], y_pred)
            roc_auc = roc_auc_score(y_test[final_in_ad], y_prob)

            # 存储结果
            results['Model'].append(model_name)
            results['ED Threshold'].append(ed_threshold)
            results['MFS Threshold'].append(similarity_threshold)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1-Score'].append(f1)
            results['ROC-AUC'].append(roc_auc)

# 将评估结果保存到DataFrame
results_df = pd.DataFrame(results)

# 保存结果到Excel文件
results_df.to_excel('model_performance_across_thresholds.xlsx', index=False)
print("Model performance across different thresholds saved to model_performance_across_thresholds.xlsx")

# 可视化模型性能随阈值变化的情况
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.lineplot(data=results_df, x='ED Threshold', y='Accuracy', hue='Model', marker='o')
plt.title('Model Accuracy Across Different ED Thresholds')
plt.show()

plt.figure(figsize=(12, 8))
sns.lineplot(data=results_df, x='MFS Threshold', y='Accuracy', hue='Model', marker='o')
plt.title('Model Accuracy Across Different MFS Thresholds')
plt.show()
