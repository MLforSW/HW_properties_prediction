import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdmolops
import os

# 加载数据
file_path = 'imputed_selected_features_Toxcity.csv'
data = pd.read_csv(file_path)

# 提取SMILES和标签
smiles_list = data['SMILES'].tolist()
labels = data['Toxicity'].values

# 提取附加特征（从第二列到倒数第二列）
additional_features = data.iloc[:, 1:-1].values

# 标准化附加特征
scaler = StandardScaler()
additional_features = scaler.fit_transform(additional_features)

# 创建图数据列表
graph_data_list = []

for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    # 获取分子图的邻接矩阵
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)

    # 将原子特征和附加特征组合
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feature = [atom.GetAtomicNum()]  # 获取原子特征（这里使用原子的种类作为示例，可以根据需要使用更多特征）
        atom_features.append(atom_feature)

    atom_features = np.array(atom_features)
    node_features = np.hstack(
        (atom_features, np.tile(additional_features[i], (atom_features.shape[0], 1))))  # 将附加特征添加为每个节点的属性

    # 转换为PyTorch张量
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels[i], dtype=torch.long)

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    graph_data_list.append(data)

# 将数据集拆分为训练集和测试集
train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# 定义模型类
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 16, heads=8, dropout=0.6)
        self.conv2 = GATConv(16 * 8, 2, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 16)
        self.conv2 = SAGEConv(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


# 模型列表
models = {
    'GCN': GCN,
    'GAT': GAT,
    'GraphSAGE': GraphSAGE
}

# 存储评估指标的字典
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': []
}


# 训练和评估模型的函数
def train_model(model_class, model_name):
    # 初始化模型和优化器
    num_node_features = graph_data_list[0].num_node_features
    model = model_class(num_node_features=num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    for epoch in range(1, 201):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

    # 保存模型
    model_path = f'{model_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 测试模型
    model.eval()
    y_true = []
    y_prob = []
    for data in test_loader:
        out = model(data.x, data.edge_index, data.batch)
        prob = torch.exp(out)[:, 1]
        y_true.extend(data.y.cpu().numpy())
        y_prob.extend(prob.cpu().detach().numpy())

    # 计算评估指标
    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    # 存储结果
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1-Score'].append(f1)
    results['ROC-AUC'].append(roc_auc)

    # 打印评估结果
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print()

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    plt.savefig(f"Confusion Matrix for {model_name}.png")


# 运行所有模型
for model_name, model_class in models.items():
    train_model(model_class, model_name)

# 保存评估结果到Excel文件
results_df = pd.DataFrame(results)
results_df.to_excel('gnn_model_comparison_results.xlsx', index=False)
print("Model evaluation metrics saved to gnn_model_comparison_results.xlsx")
