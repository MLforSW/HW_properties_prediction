import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# #Toxicity
# # 加载数据
# file_path = 'selected_features_Toxcity.csv'
# data = pd.read_csv(file_path)
#
# # 去除非数值特征并分离数值特征和目标变量
# numerical_data = data.drop(['SMILES', 'Toxicity'], axis=1)
# target = data['Toxicity']
#
# # 初始化KNN填充器
# imputer = KNNImputer(n_neighbors=5)
#
# # 对数值数据进行KNN填充
# imputed_numerical_data = imputer.fit_transform(numerical_data)
#
# # 转换回DataFrame
# imputed_numerical_df = pd.DataFrame(imputed_numerical_data, columns=numerical_data.columns)
#
# # 将填充后的数值数据与目标变量合并
# imputed_df = pd.concat([imputed_numerical_df, target], axis=1)
#
# # 分离特征和目标用于分类
# X = imputed_df.drop('Toxicity', axis=1)
# y = imputed_df['Toxicity']
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 设置参数网格用于选择最佳的k值
# param_grid = {'n_neighbors': np.arange(1, 21)}
#
# # 初始化KNN分类器
# knn = KNeighborsClassifier()
#
# # 使用GridSearchCV寻找最佳的k值
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=make_scorer(accuracy_score))
# grid_search.fit(X_train, y_train)
#
# # 提取结果
# results = grid_search.cv_results_
#
# # 绘制不同k值对应的准确率图表
# plt.figure(figsize=(10, 6))
# plt.plot(param_grid['n_neighbors'], results['mean_test_score'], marker='o')
# plt.title('Accuracy Score for Different Values of k')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy Score')
# plt.grid(True)
# plt.show()
# plt.savefig("K VS accuracy.png")
# # 最佳的k值
# best_k = grid_search.best_params_['n_neighbors']
# print(best_k)
# # 使用最佳的k值初始化KNN填充器
# imputer_best_k = KNNImputer(n_neighbors=best_k)
#
# # 使用最佳的k值对数值数据进行KNN填充
# imputed_numerical_data_best_k = imputer_best_k.fit_transform(numerical_data)
#
# # 转换回DataFrame
# imputed_numerical_df_best_k = pd.DataFrame(imputed_numerical_data_best_k, columns=numerical_data.columns)
#
# # 将填充后的数值数据与SMILES和目标变量合并
# imputed_df_best_k = pd.concat([data['SMILES'], imputed_numerical_df_best_k, target], axis=1)
#
# # 将填充后的数据保存到新的CSV文件中
# output_path = 'imputed_selected_features_Toxcity_best_k.csv'
# imputed_df_best_k.to_csv(output_path, index=False)

# #Flam
# # 加载数据
# import pandas as pd
# from sklearn.impute import KNNImputer
#
# import pandas as pd
# from sklearn.impute import KNNImputer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import make_scorer, accuracy_score
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 1. 加载数据集
# new_file_path = 'selected_features_Flam.csv'
# new_data = pd.read_csv(new_file_path)
#
# # 2. 清理数据
# # 将非数值占位符（如'#NAME?'）替换为NaN
# cleaned_data = new_data.replace('#NAME?', pd.NA)
#
# # 检查是否有其他需要替换的非数值字符串
# non_numeric_columns = cleaned_data.select_dtypes(include=['object']).columns
# cleaned_data[non_numeric_columns] = cleaned_data[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
#
# # 3. 分离SMILES，数值特征和目标变量（Flammability）
# smiles = cleaned_data['SMILES']
# numerical_data_cleaned = cleaned_data.drop(['SMILES', 'Flammability'], axis=1)
# target = cleaned_data['Flammability']
#
# # 4. 使用初始的K值（如5）初始化KNN填充器进行初次填充
# imputer = KNNImputer(n_neighbors=5)
# imputed_numerical_data = imputer.fit_transform(numerical_data_cleaned)
# imputed_numerical_df = pd.DataFrame(imputed_numerical_data, columns=numerical_data_cleaned.columns)
#
# # 5. 准备数据进行K值选择
# X = imputed_numerical_df
# y = target
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 6. 选择最佳K值
# param_grid = {'n_neighbors': np.arange(1, 21)}
# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=make_scorer(accuracy_score))
# grid_search.fit(X_train, y_train)
#
# # 提取结果并绘制图表
# results = grid_search.cv_results_
# plt.figure(figsize=(10, 6))
# plt.plot(param_grid['n_neighbors'], results['mean_test_score'], marker='o')
# plt.title('Accuracy Score for Different Values of k')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy Score')
# plt.grid(True)
# plt.show()
# plt.savefig("K VS Flam.png")
# # 最佳的K值
# best_k = grid_search.best_params_['n_neighbors']
# print(f'最佳的K值是: {best_k}')
#
# # 7. 使用最佳的K值初始化KNN填充器
# imputer_best_k = KNNImputer(n_neighbors=best_k)
#
# # 对清理后的数值数据进行KNN填充
# imputed_numerical_data_best_k = imputer_best_k.fit_transform(numerical_data_cleaned)
# imputed_numerical_df_best_k = pd.DataFrame(imputed_numerical_data_best_k, columns=numerical_data_cleaned.columns)
#
# # 8. 合并填充后的数值数据、SMILES和目标变量
# imputed_df_best_k = pd.concat([smiles, imputed_numerical_df_best_k, target], axis=1)
#
# # 9. 将填充后的数据保存到新的CSV文件中
# output_path = 'imputed_selected_features_Flam_best_k.csv'
# imputed_df_best_k.to_csv(output_path, index=False)
#
# # 输出保存的文件路径
# print("填充后的数据已保存至:", output_path)

#Reactivity
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.impute import KNNImputer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, make_scorer
#
# # 加载数据
# file_path = 'selected_features_Reactivity.csv'
# data = pd.read_csv(file_path)
#
# # 去除非数值特征并分离数值特征和目标变量
# numerical_data = data.drop(['SMILES', 'Reactivity'], axis=1)
# target = data['Reactivity']
#
# # 处理数据中的无限值
# numerical_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#
# # 初始化KNN填充器
# imputer = KNNImputer(n_neighbors=5)
#
# # 对数值数据进行KNN填充
# imputed_numerical_data = imputer.fit_transform(numerical_data)
#
# # 转换回DataFrame
# imputed_numerical_df = pd.DataFrame(imputed_numerical_data, columns=numerical_data.columns)
#
# # 将填充后的数值数据与目标变量合并
# imputed_df = pd.concat([imputed_numerical_df, target], axis=1)
#
# # 分离特征和目标用于分类
# X = imputed_df.drop('Reactivity', axis=1)
# y = imputed_df['Reactivity']
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 设置参数网格用于选择最佳的k值
# param_grid = {'n_neighbors': np.arange(1, 21)}
#
# # 初始化KNN分类器
# knn = KNeighborsClassifier()
#
# # 使用GridSearchCV寻找最佳的k值
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=make_scorer(accuracy_score))
# grid_search.fit(X_train, y_train)
#
# # 提取结果
# results = grid_search.cv_results_
#
# # 绘制不同k值对应的准确率图表
# plt.figure(figsize=(10, 6))
# plt.plot(param_grid['n_neighbors'], results['mean_test_score'], marker='o')
# plt.title('Accuracy Score for Different Values of k')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy Score')
# plt.grid(True)
# plt.savefig("K_vs_accuracy_Re.png")  # 保存图表
#
# # 最佳的k值
# best_k = grid_search.best_params_['n_neighbors']
# print(f"Best k value: {best_k}")
#
# # 使用最佳的k值初始化KNN填充器
# imputer_best_k = KNNImputer(n_neighbors=best_k)
#
# # 使用最佳的k值对数值数据进行KNN填充
# imputed_numerical_data_best_k = imputer_best_k.fit_transform(numerical_data)
#
# # 转换回DataFrame
# imputed_numerical_df_best_k = pd.DataFrame(imputed_numerical_data_best_k, columns=numerical_data.columns)
#
# # 将填充后的数值数据与SMILES和目标变量合并
# imputed_df_best_k = pd.concat([data['SMILES'], imputed_numerical_df_best_k, target], axis=1)
#
# # 将填充后的数据保存到新的CSV文件中
# output_path = 'imputed_selected_features_Reactivity_best_k.csv'
# imputed_df_best_k.to_csv(output_path, index=False)
#
# print(f"Imputed dataset saved to: {output_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, make_scorer

# 加载数据
file_path_new = 'selected_features_W.csv'
data_new = pd.read_csv(file_path_new)

# 去除非数值特征并分离数值特征和目标变量
numerical_data_new = data_new.drop(['SMILES', 'W'], axis=1)
target_new = data_new['W']

# 处理数据中的无限值
numerical_data_new.replace([np.inf, -np.inf], np.nan, inplace=True)

# 初始化KNN填充器（默认k=5）
imputer = KNNImputer(n_neighbors=5)

# 对数值数据进行KNN填充
imputed_numerical_data_new = imputer.fit_transform(numerical_data_new)

# 转换回DataFrame
imputed_numerical_df_new = pd.DataFrame(imputed_numerical_data_new, columns=numerical_data_new.columns)

# 将填充后的数值数据与目标变量合并
imputed_df_new = pd.concat([imputed_numerical_df_new, target_new], axis=1)

# 分离特征和目标用于分类
X_new = imputed_df_new.drop('W', axis=1)
y_new = imputed_df_new['W']

# 划分训练集和测试集
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# 设置参数网格用于选择最佳的k值
param_grid_new = {'n_neighbors': np.arange(1, 21)}

# 初始化KNN分类器
knn_new = KNeighborsClassifier()

# 使用GridSearchCV寻找最佳的k值
grid_search_new = GridSearchCV(knn_new, param_grid_new, cv=5, scoring=make_scorer(accuracy_score))
grid_search_new.fit(X_train_new, y_train_new)

# 提取结果
results_new = grid_search_new.cv_results_

# 绘制不同k值对应的准确率图表
plt.figure(figsize=(10, 6))
plt.plot(param_grid_new['n_neighbors'], results_new['mean_test_score'], marker='o')
plt.title('Accuracy Score for Different Values of k (New Dataset)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.savefig("K_vs_accuracy_W.png")  # 保存图表

# 最佳的k值
best_k_new = grid_search_new.best_params_['n_neighbors']
print(f"Best k value for the new dataset: {best_k_new}")

# 使用最佳的k值初始化KNN填充器
imputer_best_k_new = KNNImputer(n_neighbors=best_k_new)

# 使用最佳的k值对数值数据进行KNN填充
imputed_numerical_data_best_k_new = imputer_best_k_new.fit_transform(numerical_data_new)

# 转换回DataFrame
imputed_numerical_df_best_k_new = pd.DataFrame(imputed_numerical_data_best_k_new, columns=numerical_data_new.columns)

# 将填充后的数值数据与SMILES和目标变量合并
imputed_df_best_k_new = pd.concat([data_new['SMILES'], imputed_numerical_df_best_k_new, target_new], axis=1)

# 将填充后的数据保存到新的CSV文件中
output_path_new = 'imputed_selected_features_W_best_k.csv'
imputed_df_best_k_new.to_csv(output_path_new, index=False)

print(f"Imputed dataset saved to: {output_path_new}")
