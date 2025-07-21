import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

###############计算患者和每个患者的序列长度
# # 文件路径
# file_path = '/home/yuanhongxu/mimic3/Processed_Patient_Data_with_Numeric_Labels_Final.csv'
#
# # 读取数据
# data = pd.read_csv(file_path, float_precision='round_trip')
#
# # 统计每个 PatientID 的就诊次数
# visit_count_df = data["PatientID"].value_counts().reset_index()
# visit_count_df.columns = ["PatientID", "Visit_Count"]
# total_patients = data["PatientID"].nunique()
#
# # 计算每个就诊次数的患者人数
# visit_count_distribution = visit_count_df["Visit_Count"].value_counts().reset_index()
# visit_count_distribution.columns = ["Visit_Count", "Patient_Count"]
#
# # 排序方便查看
# visit_count_distribution = visit_count_distribution.sort_values(by="Visit_Count")
#
# # 打印结果
# print(visit_count_distribution.to_string(index=False))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
data = pd.read_csv('/home/yuanhongxu/CarpeDiem_dataset.csv')
data = data.fillna(0)
# 2. 数据筛选：只保留床段长度大于等于15的患者数据
label_column = 'Patient_category'  # 替换为实际标签列名称
patient_groups = data.groupby('Patient_id')

filtered_data = patient_groups.filter(lambda x: (len(x) >= 10 and len(x) <= 1000))

# 3. 定义特征列
features = [
    'SOFA_score', 'Temperature', 'Heart_rate', 'Systolic_blood_pressure',
    'Diastolic_blood_pressure', 'Mean_arterial_pressure', 'Respiratory_rate',
    'Oxygen_saturation', 'Urine_output', 'PEEP', 'FiO2', 'Plateau_Pressure',
    'Lung_Compliance', 'PEEP_changes', 'Respiratory_rate_changes', 'FiO2_changes',
    'WBC_count', 'Lymphocytes', 'Neutrophils', 'Hemoglobin', 'Platelets',
    'Bicarbonate', 'Creatinine', 'Albumin', 'Bilirubin', 'Procalcitonin'
]

# 4. 对特征数据进行归一化
def normalize_features(data, features):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

# 5. 对 filtered_data 的特征进行归一化
filtered_data_normalized = normalize_features(filtered_data, features)
# 6. 定义滑动窗口函数
def create_sliding_window_dataset(df, features, sequence_length=16):
    X, labelneed = [], []
    patients_data = []
    for patient_id, group in df.groupby('Patient_id'):
        patient_windows = []
        labelneed.append(group[label_column].values[0])  # 取窗口末尾的标签
        # 窗口滑动逻辑
        for start in range(0, len(group), sequence_length):  # 每次滑动整个窗口长度
            if start + sequence_length <= len(group):  # 保证窗口有足够长度
                sequence = group.iloc[start:start + sequence_length][features].values
                patient_windows.append(sequence)
                X.append(sequence)
            else:
                # 如果剩余部分不足以构成一个完整窗口，则停止
                break
        patients_data.append(np.array(patient_windows))

    return np.array(X), np.array(labelneed), patients_data

# 7. 调用滑动窗口函数
X, Y, patients_data = create_sliding_window_dataset(filtered_data_normalized, features)




# scaler = StandardScaler()
# scaler.fit((X.reshape(-1, X.shape[-1])))
#
# X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(-1, 15, len(features))
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)



import torch
import torch.nn as nn

class GENC(nn.Module):
    def __init__(self, in_channels=1, num_features=26, hidden_dim=256):
        super(GENC, self).__init__()

        # 2D 卷积：提取多特征信息
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(1, num_features)),  # 核宽度为特征数量

            nn.ReLU()

        )

        # 第一层 1D 卷积：局部特征提取
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=1),  # 小卷积核
            nn.ReLU()

        )

        # 第二层 1D 卷积：全局特征提取
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # 较大卷积核

            nn.ReLU(),

        )

    def forward(self, x):
        """
        输入: x (batch_size, in_channels, time_steps, num_features)
        输出: (batch_size, time_steps, hidden_dim)
        """
        # 2D 卷积提取多特征信息
        x = self.conv2d(x)  # (batch_size, 128, time_steps, 1)

        # 去掉最后一个维度，使其适配 1D 卷积
        x = x.squeeze(3)  # (batch_size, 128, time_steps)

        # 第一层 1D 卷积提取局部特征
        x = self.conv1d_1(x)  # (batch_size, hidden_dim, time_steps)

        # 第二层 1D 卷积提取全局特征
        x = self.conv1d_2(x)  # (batch_size, hidden_dim, time_steps)

        # 转置维度，使时间维度为第 2 维
        x = x.transpose(1, 2)  # (batch_size, time_steps, hidden_dim)

        return x



class GAR(nn.Module):
    def __init__(self):
        super(GAR, self).__init__()
        self.gru = nn.GRU(256, 256, num_layers=1,batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

# CPC模型
class CPCModel(nn.Module):
    def __init__(self):
        super(CPCModel, self).__init__()
        self.genc = GENC()
        self.gar = GAR()
        # self.temperature = temperature  # 温度参数

    def forward(self, x):
        features = self.genc(x)  # 形状: (batch_size, 5, 256)
        context_features = features[:, :10, :]  # 前三次记录
        pos_features = features[:, 10:, :]  # 后两次记录
        context = self.gar(context_features)  # 通过 GAR 获取上下文表示
        return context, pos_features  # 返回上下文特征和正样本特征

# InfoNCE 损失
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05, dynamic_negative_sampling=False):
        """
        :param temperature: 温度参数，控制对比强度
        :param dynamic_negative_sampling: 是否动态采样负样本
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.dynamic_negative_sampling = dynamic_negative_sampling

    def forward(self, context_vector, positive_samples, negative_samples):
        """
        :param context_vector: (batch_size, hidden_size)
        :param positive_samples: (batch_size, num_positive, hidden_size)
        :param negative_samples: (batch_size, num_negative, hidden_size)
        """
        batch_size = context_vector.size(0)
        num_positive = positive_samples.size(1)
        num_negative = negative_samples.size(1)



        # 初始化总损失
        total_loss = 0

        # 遍历每个正样本
        for i in range(5):
            # 当前正样本
            pos_sample = positive_samples[:, i, :]  # (batch_size, hidden_size)
            # 动态采样负样本
            if self.dynamic_negative_sampling:
                neg_sample_indices = torch.randint(0, num_negative, (batch_size, num_negative)).to(context_vector.device)
                sampled_negative_samples = torch.gather(negative_samples, 1, neg_sample_indices.unsqueeze(-1).expand(-1, -1, negative_samples.size(2)))
            else:
                sampled_negative_samples = negative_samples  # 使用所有负样本

            # 2. 计算正样本相似度
            pos_similarity = torch.sum(context_vector * pos_sample, dim=1, keepdim=True)  # (batch_size, 1)

            # 3. 计算负样本相似度
            neg_similarity = torch.bmm(sampled_negative_samples, context_vector.unsqueeze(2)).squeeze(2)  # (batch_size, num_negative)

            # 4. 合并 logits
            logits = torch.cat([pos_similarity, neg_similarity], dim=1)  # (batch_size, 1 + num_negative)

            # 5. 创建标签 (正样本总是在第 0 个位置)
            labels = torch.zeros(batch_size, dtype=torch.long).to(context_vector.device)

            # 6. 计算损失并累加
            total_loss += F.cross_entropy(logits / self.temperature, labels)

        # 7. 平均损失
        total_loss /= num_positive

        return total_loss




# 训练过程

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=200, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0  # 用于计数没有改善的周期

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0  # 用于累计训练损失
        for X_batch in train_loader:
            optimizer.zero_grad()  # 清空梯度

            # 确保数据类型正确并移动到相应设备
            X_batch = X_batch.float().to(device)

            # 前向传播
            context_features, pos_features = model(X_batch.unsqueeze(1))
            batch_size, num_positive, hidden_size = pos_features.size()

            all_pos_features = pos_features.reshape(-1, hidden_size)

            negative_samples = []
            for i in range(batch_size):
                # 当前样本的正样本索引范围
                pos_indices = list(range(i * num_positive, (i + 1) * num_positive))
                # 负样本索引为除去当前样本的正样本
                neg_indices = list(set(range(batch_size * num_positive)) - set(pos_indices))
                # 从负样本索引中随机抽取 num_negative 个样本
                num_negative = 5  # 可以根据需要设置
                neg_sample_indices = np.random.choice(neg_indices, num_negative, replace=False)
                neg_samples = all_pos_features[neg_sample_indices]
                negative_samples.append(neg_samples.unsqueeze(0))

            negative_samples = torch.cat(negative_samples, dim=0)

            # 计算 InfoNCE 损失
            loss = criterion(context_features, pos_features, negative_samples)
            total_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}')

        # 验证过程
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():
            for X_val_batch in val_loader:
                X_val_batch = X_val_batch.float().to(device)
                context_features, pos_features = model(X_val_batch.unsqueeze(1))
                batch_size, num_positive, hidden_size = pos_features.size()
                all_pos_features = pos_features.reshape(-1, hidden_size)
                negative_samples = []
                for i in range(batch_size):
                    # 当前样本的正样本索引范围
                    pos_indices = list(range(i * num_positive, (i + 1) * num_positive))
                    # 负样本索引为除去当前样本的正样本
                    neg_indices = list(set(range(batch_size * num_positive)) - set(pos_indices))
                    # 从负样本索引中随机抽取 num_negative 个样本
                    num_negative = 5  # 可以根据需要设置
                    neg_sample_indices = np.random.choice(neg_indices, num_negative, replace=False)
                    neg_samples = all_pos_features[neg_sample_indices]
                    negative_samples.append(neg_samples.unsqueeze(0))

                negative_samples = torch.cat(negative_samples, dim=0)
                loss = criterion(context_features, pos_features, negative_samples)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss}')

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 重置计数器
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model_carpe.pth')
            print('保存')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型、优化器和损失函数
model_main = CPCModel().to(device)
optimizer = optim.Adam(model_main.parameters(),lr=0.001)  # 调整学习率
criterion = InfoNCELoss()

# 训练集和验证集的加载器
train_loader = torch.utils.data.DataLoader(X_train, batch_size=32)
val_loader = torch.utils.data.DataLoader(X_test, batch_size=32)

#开始训练
train(model_main, train_loader, val_loader, optimizer, criterion, num_epochs=100)


######分类器




# 加载最佳CPC模型
model_pretrain = CPCModel()  # 创建模型实例
model_pretrain.load_state_dict(torch.load('best_model_carpe.pth'))  # 加载模型参数
model_pretrain.to(device)
model_pretrain.eval()  # 设置为评估模式

# 提取特征
features_extracted = []
window_counts = []  # 用于存储每个患者的窗口数量

with torch.no_grad():
    for patient_windows in patients_data:
        print(patient_windows.shape)
        # 将患者的所有窗口特征转换为张量，并增加通道维度
        patient_tensor = torch.tensor(patient_windows, dtype=torch.float32).unsqueeze(1).to(device)
        # 输入到模型中提取特征
        context_features, _ = model_pretrain(patient_tensor)  # 根据你的模型结构
        # 将提取的特征转换为 numpy 数组，保留二维结构
        patient_features = context_features.cpu().numpy()  # 保留每个窗口的特征
        window_counts.append(patient_features.shape[0])

        # 将提取的特征添加到列表中
        features_extracted.append(patient_features)
# 将提取的特征转换为 numpy 数组，形状为 (患者数量, 窗口数量, 特征维度)
X_features = np.array(features_extracted, dtype=object)  # 使用 dtype=object 以保持不规则形状

from sklearn.preprocessing import LabelEncoder

# 假设 y 是每个患者的标签列表
# 创建 LabelEncoder 实例
label_encoder = LabelEncoder()

# 拟合标签并转换为整数编码
y_encoded = label_encoder.fit_transform(Y)
unique_labels, counts = np.unique(y_encoded, return_counts=True)

# 打印结果
for label, count in zip(unique_labels, counts):
    print(f"标签 {label} 的数量: {count}")
from torch.utils.data import DataLoader, TensorDataset
max_windows = 100

# 准备 padding 和 mask
def pad_and_mask(X_normalized_features, max_windows):
    padded_features = []
    masks = []
    for patient_features in X_normalized_features:
        # 进行 padding
        if len(patient_features) < max_windows:
            padding = np.zeros((max_windows - len(patient_features), patient_features.shape[1]))
            padded_patient = np.vstack((patient_features, padding))
            mask = np.concatenate((np.ones(len(patient_features)), np.zeros(max_windows - len(patient_features))))
        else:
            padded_patient = patient_features[:max_windows]
            mask = np.ones(max_windows)

        padded_features.append(padded_patient)
        masks.append(mask)

    return np.array(padded_features), np.array(masks)

# 进行 padding 和 mask
X_padded, masks = pad_and_mask(X_features, max_windows)
# 将数据转换为 PyTorch 张量
X_tensor = torch.tensor(X_padded, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# # 统计每个标签的数量
# unique_labels, counts = np.unique(y_encoded, return_counts=True)
#
# # 打印每个标签及其对应的数量
# label_counts = dict(zip(unique_labels, counts))
# print("标签数量统计:", label_counts)


# 创建自定义的 Dataset
class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, masks):
        self.features = features
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.masks[idx]

# 创建数据集
dataset = PatientDataset(X_tensor, y_tensor, masks)


# LSTM 分类器
class RNNClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassificationModel, self).__init__()

        # 第一层 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=1, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self, x, mask):
        lengths = mask.sum(dim=1).long().cpu()
        # 第一层 LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        lstm_out, (h_n, c_n) = self.lstm(packed_input)

        # 取最后时间步的输出
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        last_hidden = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1, :]  # 取每个序列的最后一个有效输出

        x = F.relu(self.fc(last_hidden))  # 隐藏层
        output = self.fc2(x)

        return output  # 输出概率


# 使用示例
input_size = X_tensor.shape[2]  # 特征维度
hidden_size1 = 256  # 第一层 LSTM 隐藏层维度
output_size = len(np.unique(y_encoded))  # 类别数
model_rnn = RNNClassificationModel(input_size, hidden_size1, output_size).to(device)

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# 训练过程
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support,average_precision_score
from sklearn.model_selection import StratifiedKFold


def train_with_early_stopping(model_class, dataset, optimizer_class, criterion, num_epochs=100, patience=10, k_folds=5):
    """
    :param model_class: 模型类，用于每折初始化新的模型实例
    :param dataset: 数据集
    :param optimizer_class: 优化器类，用于每折初始化新的优化器
    :param criterion: 损失函数
    :param num_epochs: 最大训练轮数
    :param patience: 早停的容忍轮数
    :param k_folds: 交叉验证的折数
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_auc = []
    all_f1 = []
    all_auprc = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(X=np.arange(len(dataset)), y=dataset.labels)):
        print(f'\nFold {fold + 1}/{k_folds}')

        # 初始化模型和优化器
        model = model_class().to(device)  # 每一折初始化一个新的模型实例
        optimizer = optimizer_class(model.parameters())  # 每一折重新初始化优化器

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for features, labels, masks in train_loader:
                optimizer.zero_grad()
                features, labels, masks = features.to(device), labels.to(device), masks.to(device)
                outputs = model(features, masks)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()  # 更新参数

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

            # 验证过程
            model.eval()
            with torch.no_grad():
                val_loss = 0
                all_outputs = []
                all_labels = []

                for features, labels, masks in val_loader:
                    features, labels, masks = features.to(device), labels.to(device), masks.to(device)
                    outputs = model(features, masks)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    probabilities = F.softmax(outputs, dim=1)
                    all_outputs.append(probabilities.cpu())
                    all_labels.append(labels.cpu())

                avg_val_loss = val_loss / len(val_loader)
                print(f'Validation Loss: {avg_val_loss:.4f}')

                # 合并输出和标签
                all_outputs = torch.cat(all_outputs).numpy()
                all_labels = torch.cat(all_labels).numpy()

                # 计算AUC
                auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')
                print(f'AUC: {auc:.4f}')
                all_auc.append(auc)

                auprc = average_precision_score(all_labels, all_outputs, average='macro')
                all_auprc.append(auprc)
                # 计算F1分数
                predicted_labels = np.argmax(all_outputs, axis=1)
                f1 = f1_score(all_labels, predicted_labels, average='weighted')
                print(f'F1 Score: {f1:.4f}')
                all_f1.append(f1)

                # 混淆矩阵
                cm = confusion_matrix(all_labels, predicted_labels)
                print("Confusion Matrix:")
                print(cm)

                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    print(f'\nAverage AUC: {np.mean(all_auc):.4f}, Average F1 Score: {np.mean(all_f1):.4f}, Average auprc: {np.mean(all_auprc):.4f}')




# 创建模型、优化器和损失函数
# optimizer = optim.Adam(model_rnn.parameters())  # 只在这里定义一次
# criterion = nn.CrossEntropyLoss()

class_counts = np.array([136,32,102,19])
total_samples = class_counts.sum()

# 计算每个类别的权重
class_frequencies = class_counts / total_samples
class_weights = 1.0 / class_frequencies  # 与类别频率成反比

# 将权重转换为 PyTorch 张量
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 定义交叉熵损失函数并设置权重
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

train_with_early_stopping(
    model_class=lambda: RNNClassificationModel(input_size, hidden_size1, output_size),
    dataset=dataset,
    optimizer_class=lambda params: optim.Adam(params, lr=0.001),
    criterion=criterion,
    num_epochs=100,
    patience=10,
    k_folds=5
)





