import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdmolops

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

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
additional_features_resampled, labels_resampled = smote.fit_resample(additional_features, labels)

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
        # 获取原子特征（这里使用原子的种类作为示例，可以根据需要使用更多特征）
        atom_feature = [atom.GetAtomicNum()]
        atom_features.append(atom_feature)

    atom_features = np.array(atom_features)
    # 将附加特征添加为每个节点的属性
    node_features = np.hstack((atom_features, np.tile(additional_features_resampled[i], (atom_features.shape[0], 1))))

    # 转换为PyTorch张量
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels_resampled[i], dtype=torch.long)

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    graph_data_list.append(data)

# 将数据集拆分为训练集和测试集
train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # 池化
        return F.log_softmax(x, dim=1)


# 获取节点特征数
num_node_features = graph_data_list[0].num_node_features

# 初始化模型和优化器
model = GCN(num_node_features=num_node_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练模型
def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()


# 测试模型
def test(loader):
    model.eval()
    y_true = []
    y_prob = []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        prob = torch.exp(out)[:, 1]  # 获取第二类的概率
        y_true.extend(data.y.cpu().numpy())
        y_prob.extend(prob.cpu().detach().numpy())
    return y_true, y_prob


# 存储训练过程中的准确率
train_accuracies = []
test_accuracies = []

# 训练和评估模型
for epoch in range(1, 201):
    train()
    train_true, train_prob = test(train_loader)
    test_true, test_prob = test(test_loader)

    train_pred = [1 if p > 0.5 else 0 for p in train_prob]
    test_pred = [1 if p > 0.5 else 0 for p in test_prob]

    train_acc = accuracy_score(train_true, train_pred)
    test_acc = accuracy_score(test_true, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# 保存模型
model_path = 'gcn_model_Toxicity.pth.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 可视化训练过程
plt.figure(figsize=(10, 6))
plt.plot(range(1, 201), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 201), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy Over Epochs')
plt.legend()
plt.show()
plt.savefig("toxicity_training.png")
# 加载模型
loaded_model = GCN(num_node_features=num_node_features)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print("Model loaded successfully")

# 计算混淆矩阵和其他评估指标
conf_matrix = confusion_matrix(test_true, test_pred)
accuracy = accuracy_score(test_true, test_pred)
precision = precision_score(test_true, test_pred)
recall = recall_score(test_true, test_pred)
f1 = f1_score(test_true, test_pred)
roc_auc = roc_auc_score(test_true, test_prob)

# 绘制混淆矩阵的热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 打印评估指标
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# 保存评估指标到Excel
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, roc_auc]
})

metrics_df.to_excel('model_evaluation_metrics_toxicity.xlsx', index=False)
print("Evaluation metrics saved to model_evaluation_metrics.xlsx")