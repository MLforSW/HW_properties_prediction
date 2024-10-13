import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
#
# # 数据集路径列表
# datasets = {
#     'Reactivity': ('imputed_selected_features_Reactivity.csv', XGBClassifier(eval_metric='logloss')),
#     'Toxicity': ('imputed_selected_features_Toxcity.csv', XGBClassifier(eval_metric='logloss')),
#     'Flammability': ('imputed_selected_features_Flam.csv', RandomForestClassifier()),
#     'WR': ('imputed_selected_features_W_1.csv', RandomForestClassifier())
# }
#
# # 存储所有数据集的分析结果
# all_results = []
#
# # 初始化LabelEncoder
# label_encoder = LabelEncoder()
#
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
#
# # 处理每个数据集
# for name, (file_path, model) in datasets.items():
#     # 加载数据
#     data = pd.read_csv(file_path)
#
#     # 提取目标标签 (例如反应性或毒性)
#     labels = data['Reactivity'].values if 'Reactivity' in data.columns else data.iloc[:, -1].values
#
#     # 对标签进行编码，将字符串转换为数值
#     labels = label_encoder.fit_transform(labels)
#
#     # 提取并转换SMILES为分子特征
#     smiles_list = data['SMILES'].tolist()
#     features = []
#     for smiles in smiles_list:
#         feature = smiles_to_features(smiles)
#         if feature is not None:
#             features.append(feature)
#
#     features = np.array(features)
#
#     # 获取CSV文件中的其他特征（从第二列到倒数第二列）
#     additional_features = data.iloc[:, 1:-1].values
#
#     # 合并所有特征
#     all_features = np.hstack((features, additional_features))
#
#     # 数据集拆分为训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)
#
#     # 处理NaN值
#     imputer = SimpleImputer(strategy='mean')
#     X_train = imputer.fit_transform(X_train)
#     X_test = imputer.transform(X_test)
#
#     # 检查y_train和y_test是否有多个类别
#     if len(np.unique(y_train)) == 1 or len(np.unique(y_test)) == 1:
#         print(f"Skipping dataset {name} due to only one class present in y_train or y_test.")
#         continue
#
#     # 存储当前数据集的分析结果
#     results = []
#
#     # 1. Data Missing (数据缺失处理)
#     missing_methods = {
#         'Mean Imputation': SimpleImputer(strategy='mean'),
#         'Median Imputation': SimpleImputer(strategy='median'),
#         'KNN Imputation': KNNImputer()
#     }
#
#     for method_name, imputer in missing_methods.items():
#         # 使用填充策略处理缺失值
#         X_train_imputed = imputer.fit_transform(X_train)
#         X_test_imputed = imputer.transform(X_test)
#
#         # 训练模型并进行预测
#         model.fit(X_train_imputed, y_train)
#         y_pred = model.predict(X_test_imputed)
#
#         # 混淆矩阵来计算高估和低估
#         cm = confusion_matrix(y_test, y_pred)
#         if cm.shape == (2, 2):
#             tn, fp, fn, tp = cm.ravel()  # True Negative, False Positive, False Negative, True Positive
#
#             # 计算高估和低估
#             overestimation = fp  # False Positive
#             underestimation = fn  # False Negative
#
#             # 记录结果
#             results.append({
#                 'Dataset': name,
#                 'Error_Type': 'Data Missing',  # 添加 Error_Type
#                 'Method': method_name,
#                 'Overestimation': overestimation,
#                 'Underestimation': underestimation
#             })
#
#     # 2. Feature Limitation (特征集受限)
#     feature_reduction_ratios = [0.01, 0.1, 0.3, 0.5]  # 减少1%，10%，30%，50%的特征
#
#     for ratio in feature_reduction_ratios:
#         num_features_to_keep = int(X_train.shape[1] * (1 - ratio))
#         X_train_limited = X_train[:, :num_features_to_keep]
#         X_test_limited = X_test[:, :num_features_to_keep]
#
#         # 训练模型并进行预测
#         model.fit(X_train_limited, y_train)
#         y_pred = model.predict(X_test_limited)
#
#         # 混淆矩阵来计算高估和低估
#         cm = confusion_matrix(y_test, y_pred)
#         if cm.shape == (2, 2):
#             tn, fp, fn, tp = cm.ravel()
#
#             # 计算高估和低估
#             overestimation = fp  # False Positive
#             underestimation = fn  # False Negative
#
#             # 记录结果
#             results.append({
#                 'Dataset': name,
#                 'Error_Type': 'Feature Limitation',
#                 'Feature_Reduction_Ratio': f'{int(ratio * 100)}%',
#                 'Overestimation': overestimation,
#                 'Underestimation': underestimation
#             })
#
#     # 3. Overfitting (交叉验证)
#     folds = [2, 5, 10, 20]
#
#     for fold in folds:
#         kfold = KFold(n_splits=fold, shuffle=True, random_state=42)
#
#         # 检查样本数是否足够进行交叉验证
#         if len(y_train) > fold:
#             cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#
#             # 记录结果
#             results.append({
#                 'Dataset': name,
#                 'Error_Type': 'Overfitting',
#                 'Cross_Validation_Folds': fold,
#                 'Accuracy_Mean': np.mean(cv_results),
#                 'Accuracy_Std': np.std(cv_results)
#             })
#
#     # 4. Data Heterogeneity (数据异质性)
#     normalization_methods = {
#         'No Processing': None,  # 不处理数据
#         'Standardization': StandardScaler(),
#         'Normalization': MinMaxScaler()
#     }
#
#     for method_name, scaler in normalization_methods.items():
#         if method_name == 'No Processing':
#             # 如果不处理数据，直接使用原始的训练和测试集
#             X_train_processed = X_train
#             X_test_processed = X_test
#         else:
#             # 处理数据
#             X_train_processed = scaler.fit_transform(X_train)
#             X_test_processed = scaler.transform(X_test)
#
#         # 训练模型并进行预测
#         model.fit(X_train_processed, y_train)
#         y_pred = model.predict(X_test_processed)
#
#         # 混淆矩阵来计算高估和低估
#         cm = confusion_matrix(y_test, y_pred)
#         if cm.shape == (2, 2):
#             tn, fp, fn, tp = cm.ravel()
#
#             # 计算高估和低估
#             overestimation = fp  # False Positive
#             underestimation = fn  # False Negative
#
#             # 记录结果
#             results.append({
#                 'Dataset': name,
#                 'Error_Type': 'Data Heterogeneity',
#                 'Method': method_name,
#                 'Overestimation': overestimation,
#                 'Underestimation': underestimation
#             })
#
#         # 保存当前数据集的分析结果
#         all_results.extend(results)
#
# # 将结果保存到CSV文件
# results_df = pd.DataFrame(all_results)
# results_df.to_csv('detailed_error_analysis_results.csv', index=False)
# print("Detailed error analysis results saved to detailed_error_analysis_results.csv")
#
# # 绘制Sankey图
# nodes = ['Overestimation', 'Underestimation'] + list(datasets.keys()) + list(results_df['Error_Type'].unique())
# source = []
# target = []
# values = []
#
# # 为每个数据集连接错误类型和高估/低估分类
# for dataset in datasets.keys():
#     for error_type in results_df['Error_Type'].unique():
#         sub_df = results_df[(results_df['Dataset'] == dataset) & (results_df['Error_Type'] == error_type)]
#         if not sub_df.empty:
#             overestimation_sum = sub_df['Overestimation'].sum()
#             underestimation_sum = sub_df['Underestimation'].sum()
#
#             # 添加高估的连接
#             if overestimation_sum > 0:
#                 source.append(nodes.index(error_type))
#                 target.append(nodes.index('Overestimation'))
#                 values.append(overestimation_sum)
#
#             # 添加低估的连接
#             if underestimation_sum > 0:
#                 source.append(nodes.index(error_type))
#                 target.append(nodes.index('Underestimation'))
#                 values.append(underestimation_sum)
# print(results_df['Error_Type'].unique())
# # 绘制并保存Sankey图
# fig = go.Figure(go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=nodes
#     ),
#     link=dict(
#         source=source,  # 起点节点的索引
#         target=target,  # 终点节点的索引
#         value=values    # 流动的权重
#     )
# ))
#
# fig.update_layout(title_text="Detailed Error Types and Their Impact on Model Estimations by Dataset", font_size=12)
# fig.write_image("detailed_combined_sankey_diagram_1.png")
# fig.show()

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# 数据集路径列表
datasets = {
    'Reactivity': ('imputed_selected_features_Reactivity.csv', XGBClassifier(eval_metric='logloss')),
    'Toxicity': ('imputed_selected_features_Toxcity.csv', XGBClassifier(eval_metric='logloss')),
    'Flammability': ('imputed_selected_features_Flam.csv', RandomForestClassifier()),
    'WR': ('imputed_selected_features_W_1.csv', RandomForestClassifier())
}

# 存储所有数据集的分析结果
all_results = []

# 初始化LabelEncoder
label_encoder = LabelEncoder()


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


# 处理每个数据集
for name, (file_path, model) in datasets.items():
    # 加载数据
    data = pd.read_csv(file_path)

    # 提取目标标签 (例如反应性或毒性)
    labels = data['Reactivity'].values if 'Reactivity' in data.columns else data.iloc[:, -1].values

    # 对标签进行编码，将字符串转换为数值
    labels = label_encoder.fit_transform(labels)

    # 提取并转换SMILES为分子特征
    smiles_list = data['SMILES'].tolist()
    features = []
    for smiles in smiles_list:
        feature = smiles_to_features(smiles)
        if feature is not None:
            features.append(feature)

    features = np.array(features)

    # 获取CSV文件中的其他特征（从第二列到倒数第二列）
    additional_features = data.iloc[:, 1:-1].values

    # 合并所有特征
    all_features = np.hstack((features, additional_features))

    # 数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

    # 处理NaN值
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 检查y_train和y_test是否有多个类别
    if len(np.unique(y_train)) == 1 or len(np.unique(y_test)) == 1:
        print(f"Skipping dataset {name} due to only one class present in y_train or y_test.")
        continue

    # 存储当前数据集的分析结果
    results = []

    # 1. Data Missing (数据缺失处理)
    missing_methods = {
        'Mean Imputation': SimpleImputer(strategy='mean'),
        'Median Imputation': SimpleImputer(strategy='median'),
        'KNN Imputation': KNNImputer()
    }

    for method_name, imputer in missing_methods.items():
        # 使用填充策略处理缺失值
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # 训练模型并进行预测
        model.fit(X_train_imputed, y_train)
        y_pred = model.predict(X_test_imputed)

        # 混淆矩阵来计算高估和低估
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()  # True Negative, False Positive, False Negative, True Positive

            # 计算高估和低估
            overestimation = fp  # False Positive
            underestimation = fn  # False Negative

            # 记录结果
            results.append({
                'Dataset': name,
                'Error_Type': 'Data Missing',  # 添加 Error_Type
                'Method': method_name,
                'Overestimation': overestimation,
                'Underestimation': underestimation
            })

    # 2. Feature Limitation (特征集受限)
    feature_reduction_ratios = [0.01, 0.1, 0.3, 0.5]  # 减少1%，10%，30%，50%的特征

    for ratio in feature_reduction_ratios:
        num_features_to_keep = int(X_train.shape[1] * (1 - ratio))
        X_train_limited = X_train[:, :num_features_to_keep]
        X_test_limited = X_test[:, :num_features_to_keep]

        # 训练模型并进行预测
        model.fit(X_train_limited, y_train)
        y_pred = model.predict(X_test_limited)

        # 混淆矩阵来计算高估和低估
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            # 计算高估和低估
            overestimation = fp  # False Positive
            underestimation = fn  # False Negative

            # 记录结果
            results.append({
                'Dataset': name,
                'Error_Type': 'Feature Limitation',
                'Feature_Reduction_Ratio': f'{int(ratio * 100)}%',
                'Overestimation': overestimation,
                'Underestimation': underestimation
            })

    # 3. Dataset Size (数据集大小)
    dataset_size_ratios = [0.2, 0.4, 0.6, 0.8, 0.9]  # 数据集大小比例

    for size_ratio in dataset_size_ratios:
        # 按比例选择训练数据
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=size_ratio, random_state=42)

        # 训练模型并进行预测
        model.fit(X_train_small, y_train_small)
        y_pred = model.predict(X_test)

        # 混淆矩阵来计算高估和低估
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            # 计算高估和低估
            overestimation = fp  # False Positive
            underestimation = fn  # False Negative

            # 记录结果
            results.append({
                'Dataset': name,
                'Error_Type': 'Dataset Size',
                'Dataset_Size_Ratio': f'{int(size_ratio * 100)}%',
                'Overestimation': overestimation,
                'Underestimation': underestimation
            })

    # 4. Data Heterogeneity (数据异质性)
    normalization_methods = {
        'No Processing': None,  # 不处理数据
        'Standardization': StandardScaler(),
        'Normalization': MinMaxScaler()
    }

    for method_name, scaler in normalization_methods.items():
        if method_name == 'No Processing':
            # 如果不处理数据，直接使用原始的训练和测试集
            X_train_processed = X_train
            X_test_processed = X_test
        else:
            # 处理数据
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)

        # 训练模型并进行预测
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        # 混淆矩阵来计算高估和低估
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            # 计算高估和低估
            overestimation = fp  # False Positive
            underestimation = fn  # False Negative

            # 记录结果
            results.append({
                'Dataset': name,
                'Error_Type': 'Data Heterogeneity',
                'Method': method_name,
                'Overestimation': overestimation,
                'Underestimation': underestimation
            })

     # 保存当前数据集的分析结果
    all_results.extend(results)

# 将结果保存到CSV文件
results_df = pd.DataFrame(all_results)
results_df.to_csv('detailed_error_analysis_results_2.csv', index=False)
print("Detailed error analysis results saved to detailed_error_analysis_results_2.csv")

# 绘制Sankey图
nodes = ['Overestimation', 'Underestimation'] + list(datasets.keys()) + list(results_df['Error_Type'].unique())
source = []
target = []
values = []

# 为每个数据集连接错误类型和高估/低估分类
for dataset in datasets.keys():
    for error_type in results_df['Error_Type'].unique():
        sub_df = results_df[(results_df['Dataset'] == dataset) & (results_df['Error_Type'] == error_type)]
        if not sub_df.empty:
            overestimation_sum = sub_df['Overestimation'].sum()
            underestimation_sum = sub_df['Underestimation'].sum()

            # 添加高估的连接
            if overestimation_sum > 0:
                source.append(nodes.index(error_type))
                target.append(nodes.index('Overestimation'))
                values.append(overestimation_sum)

            # 添加低估的连接
            if underestimation_sum > 0:
                source.append(nodes.index(error_type))
                target.append(nodes.index('Underestimation'))
                values.append(underestimation_sum)

# 绘制并保存Sankey图
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes
    ),
    link=dict(
        source=source,  # 起点节点的索引
        target=target,  # 终点节点的索引
        value=values    # 流动的权重
    )
))

fig.update_layout(title_text="Error Types and Their Impact on Model Estimations by Dataset", font_size=12)
fig.write_image("detailed_combined_sankey_diagram_2.png")
fig.show()

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
# from sklearn.impute import SimpleImputer  # 用于填充NaN值
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib  # 用于保存模型
#
# # 加载数据
# file_path = 'imputed_selected_features_Reactivity.csv'
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
# # 获取CSV文件中的其他特征（从第二列到倒数第二列）
# additional_features = data.iloc[:, 1:-2].values
#
# # 合并所有特征
# all_features = np.hstack((features, additional_features))
#
# # 使用SimpleImputer来填充缺失值
# imputer = SimpleImputer(strategy='mean')  # 可以选择'mean', 'median', 或'most_frequent'
# all_features = imputer.fit_transform(all_features)
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
# # 定义变量存储最佳模型及其ROC-AUC
# best_model = None
# best_roc_auc = 0
# best_model_name = ""
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
#     # 检查是否为最佳模型
#     if roc_auc > best_roc_auc:
#         best_roc_auc = roc_auc
#         best_model = model
#         best_model_name = model_name
#
# # 打印最佳模型信息
# print(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
#
# # 保存最佳模型
# joblib.dump(best_model, f'{best_model_name}_best_model_4_R.pkl')
# print(f"Best model saved as {best_model_name}_best_model.pkl")
#
# # 将评估结果保存到Excel文件
# results_df = pd.DataFrame(results)
# results_df.to_excel('model_comparison_results_R.xlsx', index=False)
# print("Model evaluation metrics saved to model_comparison_results_ml.xlsx")




# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
# import joblib
# from sklearn.preprocessing import StandardScaler
#
# # 加载 HW_list 数据
# hw_list_data = pd.read_excel('HW_list.xlsx')
#
# # 加载训练集以获取训练时使用的特征
# training_data = pd.read_csv('imputed_selected_features_Reactivity.csv')
#
# # 提取除去 'SMILES'、'Toxicity' 和 'Classification' 的特征
# training_features = training_data.columns.difference(['SMILES', 'Reactivity', 'Classification'])
#
# # 提取 HW_list 中的匹配特征
# matching_features = [feature for feature in training_features if feature in hw_list_data.columns]
# hw_matching_features = hw_list_data[matching_features]
#
# # 检查是否有缺失的特征
# missing_features = [feature for feature in training_features if feature not in hw_list_data.columns]
# if missing_features:
#     print(f"以下特征在 HW_list.xlsx 中缺失: {missing_features}")
#
# # 函数：将 SMILES 转换为分子描述符和指纹
# def smiles_to_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     # 提取分子描述符
#     descriptors = [
#         Descriptors.MolWt(mol),  # 分子量
#         Descriptors.MolLogP(mol),  # LogP
#         Descriptors.NumHDonors(mol),  # 氢键供体数量
#         Descriptors.NumHAcceptors(mol)  # 氢键受体数量
#     ]
#     # 生成 Morgan 指纹
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#     fingerprint_array = np.zeros((2048,))
#     Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
#     # 合并描述符和指纹
#     features = np.concatenate([descriptors, fingerprint_array])
#     return features
#
# # 处理 HW_list 中的 SMILES 并生成对应的分子描述符和指纹
# hw_smiles_list = hw_list_data['SMILES'].tolist()
# hw_smiles_features = []
# for smiles in hw_smiles_list:
#     feature = smiles_to_features(smiles)
#     if feature is not None:
#         hw_smiles_features.append(feature)
#
# # 转换为 numpy 数组
# hw_smiles_features = np.array(hw_smiles_features)
#
# # 检查 SMILES 特征是否正确生成
# print(f"SMILES 特征数量: {hw_smiles_features.shape[1]} (应为 2052，包含分子描述符和指纹)")
#
# # 合并分子特征和 HW_list.xlsx 中的匹配特征
# hw_all_features = np.hstack((hw_smiles_features, hw_matching_features.values))
#
# # 检查合并后特征数量
# print(f"合并后特征数量: {hw_all_features.shape[1]} (应为 2113)")
#
# # 加载保存的 StandardScaler 和模型
# scaler = joblib.load('scaler.pkl')
# best_model = joblib.load('XGBoost_best_model_4_R.pkl')
#
# # 对所有特征进行标准化
# try:
#     hw_features_scaled = scaler.transform(hw_all_features)
# except ValueError as e:
#     print(f"标准化错误: {e}")
#     print(f"当前特征数: {hw_all_features.shape[1]}，期望特征数: {scaler.n_features_in_}")
#
# # 使用保存的模型进行毒性预测
# hw_reactivity_predictions = best_model.predict(hw_features_scaled)
#
# # 输出预测结果
# print("反应性性预测结果：")
# print(hw_reactivity_predictions)
#
# # 如果需要将预测结果添加到原数据中并保存
# hw_list_data['Reactivity_Prediction'] = hw_reactivity_predictions
# hw_list_data.to_excel('HW_list_with_predictions_R.xlsx', index=False)
# print("预测结果已保存到 HW_list_with_predictions.xlsx")
