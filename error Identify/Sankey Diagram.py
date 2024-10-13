import pandas as pd
import plotly.graph_objects as go

# 读取CSV数据
df = pd.read_csv('detailed_error_analysis_results_2.csv')

# 创建唯一的标签列表：错误类型、数据集、以及过高和过低估计
error_types = df['Error_Type'].unique().tolist()
datasets = df['Dataset'].unique().tolist()

# 定义节点（错误类型、数据集、过高/过低估计）
labels = error_types + datasets + ['Overestimations', 'Underestimations']

# 定义source和target连接
source = []
target = []
value = []

# 用于保存到 CSV 的数据
csv_data = {
    'source': [],
    'target': [],
    'value': []
}

# 连接错误类型到数据集
for i, row in df.iterrows():
    src_idx = error_types.index(row['Error_Type'])  # 错误类型作为source
    tgt_idx = len(error_types) + datasets.index(row['Dataset'])  # 数据集作为target
    val = row['Overestimation'] + row['Underestimation']  # 总错误数量（过高 + 过低）

    source.append(src_idx)
    target.append(tgt_idx)
    value.append(val)

    # 添加到 CSV 数据
    csv_data['source'].append(labels[src_idx])
    csv_data['target'].append(labels[tgt_idx])
    csv_data['value'].append(val)

# 连接数据集到过高估计和过低估计
for i, row in df.iterrows():
    # 从数据集到过高估计
    src_idx = len(error_types) + datasets.index(row['Dataset'])
    tgt_idx = len(error_types) + len(datasets)  # 过高估计的索引
    val = row['Overestimation']

    source.append(src_idx)
    target.append(tgt_idx)
    value.append(val)

    # 添加到 CSV 数据
    csv_data['source'].append(labels[src_idx])
    csv_data['target'].append('Overestimations')
    csv_data['value'].append(val)

    # 从数据集到过低估计
    tgt_idx = len(error_types) + len(datasets) + 1  # 过低估计的索引
    val = row['Underestimation']

    source.append(src_idx)
    target.append(tgt_idx)
    value.append(val)

    # 添加到 CSV 数据
    csv_data['source'].append(labels[src_idx])
    csv_data['target'].append('Underestimations')
    csv_data['value'].append(val)

# 定义柔和的色系（符合Nature期刊风格的配色）
nature_colors = [
    "#1f77b4",  # 柔和的蓝色
    "#ff7f0e",  # 柔和的橙色
    "#2ca02c",  # 柔和的绿色
    "#d62728",  # 柔和的红色
    "#9467bd",  # 柔和的紫色
    "#8c564b",  # 柔和的棕色
    "#e377c2",  # 柔和的粉红色
    "#7f7f7f",  # 柔和的灰色
    "#bcbd22",  # 柔和的黄绿色
    "#17becf"   # 柔和的青色
]
# 将数据保存到CSV文件
df_sankey = pd.DataFrame(csv_data)
df_sankey.to_csv('sankey_diagram_data.csv', index=False)

# 创建桑基图，并使用自然风格颜色
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=nature_colors[:len(labels)],  # 应用Nature风格的颜色
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=[nature_colors[i % len(nature_colors)] for i in source],  # 让流线颜色与节点颜色匹配
    )
))

# 更新布局：调整字体大小并设置加粗
fig.update_layout(
    title_text="Error Type to Dataset Flow with Overestimations and Underestimations",
    font=dict(size=14, color='black', family='Arial'),
    title_font=dict(size=16, color='black', family='Arial', weight='bold')  # 使用weight参数来设置加粗
)

# 保存Sankey图
fig.write_image("sankey_diagram_5.png")
fig.show()



print("Sankey Diagram 数据已保存到 sankey_diagram_data.csv")
