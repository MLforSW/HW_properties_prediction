import pandas as pd
import holoviews as hv
from holoviews import opts
from bokeh.io.export import export_png

hv.extension('bokeh')

# 读取CSV数据
df = pd.read_csv('detailed_error_analysis_results_2.csv')

# 准备数据：错误类型 -> 数据集 -> 过高估计和过低估计的流动
data = []

for i, row in df.iterrows():
    # 连接错误类型和数据集
    data.append((row['Error_Type'], row['Dataset'], row['Overestimation'] + row['Underestimation']))

    # 连接数据集到过高估计
    data.append((row['Dataset'], 'Overestimations', row['Overestimation']))

    # 连接数据集到过低估计
    data.append((row['Dataset'], 'Underestimations', row['Underestimation']))

# 将数据转换为DataFrame格式
df_chord = pd.DataFrame(data, columns=['source', 'target', 'value'])

# 保存数据到CSV文件
output_path = 'chord_diagram_data.csv'
df_chord.to_csv(output_path, index=False)

# 创建节点数据集（source 和 target 的所有唯一值）
nodes = pd.DataFrame(list(set(df_chord['source']).union(set(df_chord['target']))), columns=['name'])
nodes['index'] = nodes.index

# 创建边数据集，将source和target替换为索引值
edges = df_chord.copy()
edges = edges.merge(nodes, how='left', left_on='source', right_on='name').rename(columns={'index': 'source_index'})
edges = edges.merge(nodes, how='left', left_on='target', right_on='name').rename(columns={'index': 'target_index'})

# 创建Chord图
chord = hv.Chord((edges[['source_index', 'target_index', 'value']], hv.Dataset(nodes, 'index', 'name')))

# 应用自定义样式和配置
chord.opts(
    opts.Chord(
        cmap='Category20',  # 使用柔和的颜色
        edge_color='source',  # 边缘的颜色使用来源节点的颜色
        labels='name',  # 显示节点标签
        node_color='index',  # 节点颜色
        edge_cmap='Category20',  # 边缘颜色映射
        width=600,
        height=600
    )
)

# 渲染为bokeh对象
plot = hv.render(chord, backend='bokeh')

# 保存为PNG图片
export_png(plot, filename="chord_diagram.png")
