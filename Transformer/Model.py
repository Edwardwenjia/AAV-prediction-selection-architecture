import torch.nn as nn

class RNN_Model(nn.Module):
    """
    vocab_size : 词汇表大小，即输入数据的不同字符数目
    embed_dim : 嵌入层的维度。在这里我们把每个字符嵌入到一个 embed_dim 维的向量中
    hidden_size : 循环神经网络的隐含状态的大小
    num_layers : 循环神经网络的层数
    dropout : Dropout正则化的概率
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(RNN_Model, self).__init__()

        # 嵌入层(embedding layer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 循环神经网络(RNN)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 线性层(linear layer)将输出转换为一个float值
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 嵌入输入(input)并进行维度转换
        embedded = self.embedding(x)

        # 将数据送入RNN中
        rnn_output, _ = self.rnn(embedded)

        # 线性层(linear layer)将输出转换为一个float值
        output = self.linear(rnn_output[:, -1, :].squeeze())

        return output

class Transformer_Model_1(nn.Module):
    """
    vocab_size : 词汇表大小，即输入数据的不同字符数目
    embed_dim : 嵌入层的维度。在这里我们把每个字符嵌入到一个 embed_dim 维的向量中
    hidden_size : Transformer中注意力机制的隐含状态的大小
    num_layers : Transformer中注意力机制的层数
    dropout : Dropout正则化的概率
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(Transformer_Model_1, self).__init__()

        # 嵌入层(embedding layer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer模块
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )

        # 线性层(linear layer)将输出转换为一个float值
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # 嵌入输入(input)并进行维度转换
        embedded = self.embedding(x).permute(1, 0, 2)  # Shape: (seq_len, batch_size, embed_dim)

        # 使用Transformer模块进行特征提取
        transformer_output = self.transformer(embedded)

        # 取最后一个位置的输出作为序列表示
        sequence_repr = transformer_output[-1, :, :]

        # 线性层(linear layer)将输出转换为一个float值
        output = self.linear(sequence_repr)

        return output

# _____________
import torch
import torch.nn as nn


import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, _ = self.gru(embedded)
        gru_output = self.dropout(gru_output)
        return gru_output

class Transformer_Model_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, dropout):
        super(Transformer_Model_2, self).__init__()
        # 20 128 256 1 0.2
        self.gru_model = GRUModel(vocab_size, embed_dim, hidden_size, num_layers_1, dropout)
        self.transformer = nn.TransformerEncoder(
            #
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers_2
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_output = self.gru_model(x).permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_size)
        transformer_output = self.transformer(gru_output)

        # 添加残差连接
        output = gru_output + transformer_output

        sequence_repr = output[-1, :, :]
        output = self.linear(sequence_repr)
        return output


import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, _ = self.gru(embedded)
        gru_output = self.dropout(gru_output)
        return gru_output

class Transformer_Model_3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, dropout):
        super(Transformer_Model_3, self).__init__()
        self.gru_model = GRUModel(vocab_size, embed_dim, hidden_size, num_layers_1, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers_2,
            norm=nn.LayerNorm(hidden_size)  # 添加Layer Normalization
        )
        self.relu = nn.ReLU()  # 添加ReLU激活函数
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_output = self.gru_model(x).permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_size)
        transformer_output = self.transformer(gru_output)

        # 添加残差连接
        output = gru_output + transformer_output

        # 添加ReLU激活函数
        output = self.relu(output)

        sequence_repr = output[-1, :, :]
        output = self.linear(sequence_repr)
        return output


class Transformer_Model_4(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, dropout):
        super(Transformer_Model_4, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear_1 = nn.Linear(embed_dim, hidden_size)
        self.transformer_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers_1,
            norm=nn.LayerNorm(hidden_size)  # 添加Layer Normalization
        )
        self.gru_model_1 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers_2, dropout=dropout, batch_first=True, bidirectional=True)
        self.gru_model_2 = nn.GRU(64, hidden_size, num_layers=1, dropout=dropout, batch_first=True)
        self.relu = nn.ReLU()  # 添加ReLU激活函数
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded_x = self.embedding(x).permute(1, 0, 2)  # 进行嵌入维度转换 [7, 128]
        trans_x = self.linear_1(embedded_x)
        transformer_output = self.transformer_1(trans_x)
        gru_output, _ = self.gru_model_1(transformer_output)

        # 残差连接
        combined_output = embedded_x + gru_output

        output,_ = self.gru_model_2(combined_output)

        sequence_repr = output[-1, :, :]
        output = self.linear_2(sequence_repr)

        return output

import torch.nn as nn

class RNN_Model(nn.Module):
    """
    vocab_size : 词汇表大小，即输入数据的不同字符数目
    embed_dim : 嵌入层的维度。在这里我们把每个字符嵌入到一个 embed_dim 维的向量中
    hidden_size : 循环神经网络的隐含状态的大小
    num_layers : 循环神经网络的层数
    dropout : Dropout正则化的概率
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(RNN_Model, self).__init__()

        # 嵌入层(embedding layer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 循环神经网络(RNN)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 线性层(linear layer)将输出转换为一个float值
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 嵌入输入(input)并进行维度转换
        embedded = self.embedding(x)

        # 将数据送入RNN中
        rnn_output, _ = self.rnn(embedded)

        # 线性层(linear layer)将输出转换为一个float值
        output = self.linear(rnn_output[:, -1, :].squeeze())

        return output

class Transformer_Model_1(nn.Module):
    """
    vocab_size : 词汇表大小，即输入数据的不同字符数目
    embed_dim : 嵌入层的维度。在这里我们把每个字符嵌入到一个 embed_dim 维的向量中
    hidden_size : Transformer中注意力机制的隐含状态的大小
    num_layers : Transformer中注意力机制的层数
    dropout : Dropout正则化的概率
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(Transformer_Model_1, self).__init__()

        # 嵌入层(embedding layer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer模块
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )

        # 线性层(linear layer)将输出转换为一个float值
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # 嵌入输入(input)并进行维度转换
        embedded = self.embedding(x).permute(1, 0, 2)  # Shape: (seq_len, batch_size, embed_dim)

        # 使用Transformer模块进行特征提取
        transformer_output = self.transformer(embedded)

        # 取最后一个位置的输出作为序列表示
        sequence_repr = transformer_output[-1, :, :]

        # 线性层(linear layer)将输出转换为一个float值
        output = self.linear(sequence_repr)

        return output

# ------------------------ 模型 1 先 Trans 再 GRU -------------------------------
import torch
import torch.nn as nn

class TransModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(TransModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer(embedded)
        transformer_output = self.dropout(transformer_output)
        return transformer_output

class TransGRU_1(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, dropout):
        super(TransGRU_1, self).__init__()

        self.trans_model = TransModel(vocab_size, embed_dim, hidden_size, num_layers_2, dropout)

        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers_1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        transformer_output = self.trans_model(x)

        gru_output, _ = self.gru(transformer_output)
        sequence_repr = gru_output[:, -1, :]

        output = self.linear(sequence_repr)

        return output

# ------------------------ 模型 2 先 单向 GRU 再 Trans -------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, _ = self.gru(embedded)
        gru_output = self.dropout(gru_output)
        return gru_output


class Transformer_Model_3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, dropout):
        super(Transformer_Model_3, self).__init__()
        self.gru_model = GRUModel(vocab_size, embed_dim, hidden_size, num_layers_1, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers_2,
            norm=nn.LayerNorm(hidden_size)  # 添加Layer Normalization
        )
        self.relu = nn.ReLU()  # 添加ReLU激活函数
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_output = self.gru_model(x).permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_size)
        transformer_output = self.transformer(gru_output)

        # 添加残差连接
        output = gru_output + transformer_output

        # 添加ReLU激活函数
        output = self.relu(output)

        sequence_repr = output[-1, :, :]
        output = self.linear(sequence_repr)
        return output

# ------------------------ 模型 3 先 双向 GRU 再 Trans -------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, _ = self.gru(embedded)
        gru_output = self.dropout(gru_output)
        return gru_output

class Transformer_Model_5(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, dropout):
        super(Transformer_Model_5, self).__init__()
        self.gru_model = GRUModel(vocab_size, embed_dim, hidden_size, num_layers_1, dropout)
        self.transformer= nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size * 2, nhead=4, dim_feedforward=hidden_size * 2, dropout=dropout),
            num_layers=num_layers_2,
            norm=nn.LayerNorm(hidden_size * 2)  # 添加Layer Normalization
        )
        self.relu = nn.ReLU()  # 添加ReLU激活函数
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        gru_output = self.gru_model(x).permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_size)
        transformer_output = self.transformer(gru_output)

        # 添加残差连接
        output = gru_output + transformer_output

        # 添加ReLU激活函数
        output = self.relu(output)

        sequence_repr = output[-1, :, :]
        output = self.linear(sequence_repr)
        return output


# ------------------------ 模型 3 先 双向 GRU 再 Trans -------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, _ = self.gru(embedded)
        gru_output = self.dropout(gru_output)
        return gru_output

class Transformer_Model_6(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers_1, num_layers_2, num_head, dropout):
        super(Transformer_Model_6, self).__init__()
        self.gru_model = GRUModel(vocab_size, embed_dim, hidden_size, num_layers_1, dropout)
        self.transformer_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size * 2, nhead=num_head, dim_feedforward=hidden_size * 2, dropout=dropout),
            num_layers=num_layers_2,
            norm=nn.LayerNorm(hidden_size * 2)  # 添加Layer Normalization
        )
        # self.relu = nn.ReLU()  # 添加ReLU激活函数
        # self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size * 4, 1)

    def forward(self, x):
        gru_output = self.gru_model(x).permute(1, 0, 2)  #[len, bs, flen] # Shape: (seq_len, batch_size, hidden_size)
        transformer_output = self.transformer_1(gru_output)  #[len, bs, flen]

        # 添加残差连接
        output = gru_output + transformer_output #[len, bs, flen]

        # 拼接初始时间步和末尾时间步
        initial_time_step = output[0, :, :]  # [bs, flen]
        final_time_step = output[-1, :, :]  # [bs, flen]
        concatenated_repr = torch.cat((initial_time_step, final_time_step), dim=1)  # [bs, 2*flen] 0.95--0.90

        #combine_repr = initial_time_step + final_time_step


        # 使用线性层降维
        output = self.linear(concatenated_repr)  # [bs, 1]

        #output = self.linear(combine_repr)  # [bs, 1]

        # output = self.relu(output) #[len, bs, flen]
        #
        # sequence_repr = output[-1, :, :]  #[bs, flen]
        # output = self.linear(sequence_repr) #[bs, 1]
        return output












