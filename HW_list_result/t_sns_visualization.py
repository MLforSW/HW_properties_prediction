# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
# from rdkit import DataStructs
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# #t-SNE
# # 加载Excel文件
# file_path = 'HW_list_data.xlsx'
# data = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 转换 SMILES 为特征的函数
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
#     # 生成 Morgan 指纹
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#     fingerprint_array = np.zeros((2048,))
#     DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
#     # 合并描述符和指纹
#     features = np.concatenate([descriptors, fingerprint_array])
#     return features
#
# # 应用 SMILES 到特征的转换
# data['SMILES_Features'] = data['SMILES'].apply(smiles_to_features)
#
# # 移除特征为 None 的行
# data = data.dropna(subset=['SMILES_Features'])
#
# # 提取 SMILES 转换后的特征
# smiles_features = np.array(data['SMILES_Features'].tolist())
#
# # 提取文件中的其他特征列，除了 'SMILES'、'Toxicity' 和 'Classification'
# feature_columns = ['Toxicity', 'Flammability', 'Reactivity', 'WR']
# other_features = data[feature_columns].to_numpy()
#
# # 组合 SMILES 特征和其他特征
# combined_features = np.hstack((smiles_features, other_features))
#
# # 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(combined_features)
#
# # 将 t-SNE 结果添加到数据框中
# data['TSNE_D1'] = tsne_results[:, 0]
# data['TSNE_D2'] = tsne_results[:, 1]
#
# # 创建 t-SNE 散点图的函数
# def plot_tsne(data, property_column, filename):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(data[data[property_column] == 1]['TSNE_D1'], data[data[property_column] == 1]['TSNE_D2'],
#                 color='red', alpha=0.6, label='Active')
#     plt.scatter(data[data[property_column] == 0]['TSNE_D1'], data[data[property_column] == 0]['TSNE_D2'],
#                 color='skyblue', alpha=0.6, label='Inactive')
#     plt.title(f'{property_column} Distribution', fontsize=20, fontweight='bold')
#     plt.xlabel('t-SNE D1', fontsize=20, fontweight='bold')
#     plt.ylabel('t-SNE D2', fontsize=20, fontweight='bold')
#     plt.xticks(fontsize=18, fontweight='bold')  # 加粗 x 轴数字
#     plt.yticks(fontsize=18, fontweight='bold')  # 加粗 y 轴数字
#     plt.legend(fontsize=20, frameon=True, loc='best', prop={'weight': 'bold', 'size': 18})  # 更新：加粗和放大图例
#     plt.gca().spines['top'].set_linewidth(1.5)  # 加粗上边框
#     plt.gca().spines['right'].set_linewidth(1.5)  # 加粗右边框
#     plt.gca().spines['left'].set_linewidth(1.5)  # 加粗左边框
#     plt.gca().spines['bottom'].set_linewidth(1.5)  # 加粗下边框
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()
#
# # 绘制基于每个性质的 t-SNE 图
# plot_tsne(data, 'Toxicity', 'toxicity_tsne.png')
# plot_tsne(data, 'Flammability', 'flammability_tsne.png')
# plot_tsne(data, 'Reactivity', 'reactivity_tsne.png')
# plot_tsne(data, 'WR', 'wr_tsne.png')




# pie chart plot
import pandas as pd
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = 'HW_list_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 创建饼状图的函数
def plot_pie_chart(data, property_column, filename):
    # 计算 Active 和 Inactive 的数量
    counts = data[property_column].value_counts()
    active_count = counts.get(1, 0)  # 获取活性 (1)，若不存在则为 0
    inactive_count = counts.get(0, 0)  # 获取非活性 (0)，若不存在则为 0

    labels = ['Inactive', 'Active']
    sizes = [inactive_count, active_count]
    # 使用与 t-SNE 相同的填充颜色：天蓝色和红色
    colors = ['skyblue', 'red']  # Inactive: 天蓝色, Active: 红色

    # 绘制饼状图
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 20, 'fontweight': 'bold'})
    #plt.title(f'{property_column} Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')  # 保证饼状图是圆形
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 绘制每个属性的饼状图
plot_pie_chart(data, 'Toxicity', 'toxicity_pie_chart.png')
plot_pie_chart(data, 'Flammability', 'flammability_pie_chart.png')
plot_pie_chart(data, 'Reactivity', 'reactivity_pie_chart.png')
plot_pie_chart(data, 'RW', 'wr_pie_chart.png')

#Venn diagram
import pandas as pd
from upsetplot import UpSet
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = 'HW_list_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 创建四个属性的布尔值
data['Toxicity_active'] = data['Toxicity'] == 1
data['Flammability_active'] = data['Flammability'] == 1
data['Reactivity_active'] = data['Reactivity'] == 1
data['RW_active'] = data['RW'] == 1

# 创建一个组合属性列，用于反映交集
data['combination'] = list(zip(data['Toxicity_active'], data['Flammability_active'],
                               data['Reactivity_active'], data['RW_active']))

# 生成交集的计数
combination_counts = data['combination'].value_counts()

# 将组合转换为 MultiIndex 格式，适用于 UpSet
multi_index = pd.MultiIndex.from_tuples(combination_counts.index, names=['Toxicity', 'Flammability', 'Reactivity', 'RW'])
upset_data = pd.Series(combination_counts.values, index=multi_index)

# 使用 UpSetPlot 绘制交集，显示数值
upset = UpSet(upset_data, subset_size='count', show_counts='%d', element_size=40, intersection_plot_elements=10)

# 绘制图表
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
upset.plot(fig=fig)

# 手动更改条形图的颜色
for container in ax.containers:
    if isinstance(container, plt.BarContainer):
        for bar in container.patches:
            bar.set_facecolor('#ff69b4')  # 设置为粉红色

plt.title('UpSet Diagram of Toxicity, Flammability, Reactivity, RW')
plt.tight_layout()

# 保存图像
plt.savefig('upset_diagram_properties_pink_fixed_update.png')

#Venn Digram
import pandas as pd
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = 'HW_list_data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 将属性转换为集合，表示活性 (1) 的索引
toxicity_set = set(data[data['Toxicity'] == 1].index)
flammability_set = set(data[data['Flammability'] == 1].index)
reactivity_set = set(data[data['Reactivity'] == 1].index)
wr_set = set(data[data['RW'] == 1].index)

# 定义要绘制的三集合组合
venn_combinations = [
    ('Toxicity', 'Flammability', 'Reactivity', toxicity_set, flammability_set, reactivity_set),
    ('Toxicity', 'Flammability', 'RW', toxicity_set, flammability_set, wr_set),
    ('Toxicity', 'Reactivity', 'RW', toxicity_set, reactivity_set, wr_set),
    ('Flammability', 'Reactivity', 'RW', flammability_set, reactivity_set, wr_set)
]

# 依次绘制每个三集合的韦恩图
for combo in venn_combinations:
    plt.figure(figsize=(8, 8))
    venn = venn3([combo[3], combo[4], combo[5]], set_labels=(combo[0], combo[1], combo[2]))

    # 放大和加粗标签
    for text in venn.set_labels:
        text.set_fontsize(24)  # 放大标签字体
        text.set_fontweight('bold')  # 加粗标签字体
    for text in venn.subset_labels:
        if text:  # 防止有的交集为空没有标签
            text.set_fontsize(24)  # 放大交集数字的字体
            text.set_fontweight('bold')  # 加粗交集数字的字体

    # 添加图例，放在左下角，并加粗
    legend_labels = [combo[0], combo[1], combo[2]]
    handles = [plt.Line2D([0], [0], color='red', lw=4),
               plt.Line2D([0], [0], color='green', lw=4),
               plt.Line2D([0], [0], color='blue', lw=4)]

    # 设置图例位置为左下角，并加粗字体
    plt.legend(handles, legend_labels, loc='lower left', fontsize=28, frameon=True, fancybox=True,
              title_fontsize=28, prop={'size': 20,'weight': 'bold'})  # 设置图例字体大小和加粗

    # 设置标题
    # plt.title(f'Venn Diagram of {combo[0]}, {combo[1]}, {combo[2]}', fontsize=20, fontweight='bold')

    # 保存图像
    plt.tight_layout()
    plt.savefig(f'venn_diagram_{combo[0]}_{combo[1]}_{combo[2]}.png')


# # bar plot with intersection
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载Excel文件
# file_path = 'HW_list_data.xlsx'
# data = pd.read_excel(file_path, sheet_name='Sheet1')
#
# # 定义四个属性列
# attributes = ['Toxicity', 'Flammability', 'Reactivity', 'WR']
#
# # 计算具有 n 种属性的SMILES数量
# one_property_smiles = data[data[attributes].sum(axis=1) == 1]  # 具有一种属性的
# two_properties_smiles = data[data[attributes].sum(axis=1) == 2]  # 具有两种属性的
# three_properties_smiles = data[data[attributes].sum(axis=1) == 3]  # 具有三种属性的
# four_properties_smiles = data[data[attributes].sum(axis=1) == 4]  # 具有四种属性的
#
# # 保存SMILES到CSV文件
# one_property_smiles.to_csv('one_property_smiles.csv', columns=['SMILES'], index=False)
# two_properties_smiles.to_csv('two_property_smiles.csv', columns=['SMILES'], index=False)
# three_properties_smiles.to_csv('three_property_smiles.csv', columns=['SMILES'], index=False)
# four_properties_smiles.to_csv('four_property_smiles.csv', columns=['SMILES'], index=False)
#
# # 计算数量
# counts = {
#     'One Property': len(one_property_smiles),
#     'Two Properties': len(two_properties_smiles),
#     'Three Properties': len(three_properties_smiles),
#     'Four Properties': len(four_properties_smiles)
# }
#
# # 使用马卡龙色系颜色填充（粉红色和淡蓝色）
# macaron_colors = ['#FFC0CB', '#ADD8E6', '#98FB98', '#FFC0CB']  # 粉红色, 淡蓝色, 浅绿色, 粉红色
#
# # 创建柱状图
# plt.figure(figsize=(10, 4))
# bars = plt.bar(counts.keys(), counts.values(), color=macaron_colors, width=0.8)
#
# # 添加数值显示
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', fontsize=18, fontweight='bold')
#
# # 设置图表的标签和标题
# # plt.xlabel('Number of Properties', fontsize=14, fontweight='bold')
# plt.ylabel('Number of SMILES', fontsize=20, fontweight='bold')
# plt.title('SMILES with 1, 2, 3, and 4 Properties', fontsize=20, fontweight='bold')
# plt.xticks(fontsize=18, fontweight='bold')
# plt.yticks(fontsize=18, fontweight='bold')
# plt.tight_layout()
#
# # 保存柱状图
# plt.savefig('smiles_property_bar_chart_macaron_custom.png')
#
#
#
