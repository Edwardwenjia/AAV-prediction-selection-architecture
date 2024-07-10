import math
import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from Model import RNN_Model, Transformer_Model_5, Transformer_Model_6, Transformer_Model_3, TransGRU_1
import random
import pickle
from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split
import json
import random
import numpy as np
import torch.nn.functional as F


class NucleotideDataset(Dataset):
    '''
    核苷酸序列的Dataset
    '''

    def __init__(self, datas):
        super(NucleotideDataset, self).__init__()
        # 读取 datas 的数据
        self.datas = datas
        # 创建序列上核苷酸和数字对应的字典
        self.d_dict = {'A': 0, 'G': 1, 'C': 2, 'T': 3}

    def __getitem__(self, idx):
        data_used = self.datas[idx]  # 根据某个 id 获取 data_used
        seq_x = list(data_used['seq'])  # 根据某个 seq 的字符串数据，将其转为列表数据
        seq_x = [self.d_dict[i] for i in seq_x]  # 将字符列表转换为数字列表
        nor_pac_y = data_used['nor_package']  # 取出 nor_package 的值
        # 为了避免出现负数或零，我们将对每个值取对数，并将结果存储在新的列表中
        # log_nor_pac_y = [math.log(x) for x in nor_pac_y]
        return torch.LongTensor(seq_x), nor_pac_y  # 返回训练数据

    def __len__(self):
        return len(self.datas)


class Amino(Dataset):
    '''
    核苷酸序列的Dataset
    '''

    def __init__(self, datas):
        super(Amino, self).__init__()
        # 读取 datas 的数据
        self.datas = datas
        # 创建序列上核苷酸和数字对应的字典
        self.d_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                       'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    def __getitem__(self, idx):
        data_used = self.datas[idx]  # 根据某个 id 获取 data_used
        seq_x = list(data_used['aa'])  # 根据某个 seq 的字符串数据，将其转为列表数据
        seq_x = [self.d_dict[i] for i in seq_x]  # 将字符列表转换为数字列表
        nor_pac_y = data_used['nor_package']  # 取出 nor_package 的值
        # 为了避免出现负数或零，我们将对每个值取对数，并将结果存储在新的列表中
        # log_nor_pac_y = [math.log(x) for x in nor_pac_y]
        return torch.LongTensor(seq_x), nor_pac_y  # 返回训练数据

    def __len__(self):
        return len(self.datas)


class ModelConfig:
    def __init__(self, vs, ed, hs, nl, dp, ler, batch_size, train_rate, val_rate, device, epochs):
        self.vs = vs
        self.ed = ed
        self.hs = hs
        self.nl = nl
        self.dp = dp
        self.ler = ler
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.device = device
        self.epochs = epochs

def calculate_mape(pre_value, targets):
    absolute_error = torch.abs(targets - pre_value)
    mape = torch.mean(absolute_error / targets) * 100
    return mape.item()

def rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))

def get_data(filename, train_rate_used=0, val_rate_used=0, batch=0, datatype='N'):
    '''

    :param filename: 文件路径
    :param train_rate_used: 训练集数据占比
    :param val_rate_used: 验证集数据占比
    :param batch: 每个batch大小
    :param type: 选取哪种dataset。type = 0，选择核苷酸的数据
    :return:
    '''
    all_data_used = json.load(open(filename, 'r'))  # 打开数据
    random.shuffle(all_data_used)

    # 获取所有数据的labels
    labels = [item['label'] for item in all_data_used]

    # 根据labels进行分层采样
    train_data, temp_data = train_test_split(all_data_used, train_size=train_rate_used, stratify=labels)
    val_data, test_data = train_test_split(temp_data, train_size=val_rate_used / (1 - train_rate_used),
                                           stratify=[item['label'] for item in temp_data])

    if datatype == 'N':
        # 使用 核苷酸序列 的 DataSet 类
        train_dataset = NucleotideDataset(train_data)
        train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch)  # 每个 epoch 再打乱

        val_dataset = NucleotideDataset(val_data)
        val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch)

        test_dataset = NucleotideDataset(test_data)
        test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch)

    if datatype == 'A':
        train_dataset = Amino(train_data)
        train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch)  # 每个 epoch 再打乱

        val_dataset = Amino(val_data)
        val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch)

        test_dataset = Amino(test_data)
        test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch)

    return train_data_loader, val_data_loader, test_data_loader

def train_and_validate_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs,
                             save_model_as_filename):
    '''
    :param model: 进行训练的模型
    :param train_dataloader: 将之前得到的 训练数据加载器 载入
    :param val_dataloader: 将之前得到的 验证数据加载器 载入
    :param criterion: 选取合适的损失函数
    :param optimizer: 选取合适的优化器
    :param device: 选取合适的设备
    :param epochs: 指定 epoch
    :return:
    train_loss_res: 存储每一个 epoch 中的 criterion 误差均值，用来观察变化趋势
    train_mape_res: 存储每一个 epoch 中的 mape 误差均值，用来观察变化趋势
    val_loss_res:
    val_mape_res:
    save_model_as_filename: 存储模型
    '''


    # 存储训练误差和验证误差
    train_loss_res = []
    val_loss_res = []

    train_loss2_res = []
    val_loss2_res = []

    val_mape_res = []
    train_mape_res = []

    best_loss = np.inf

    # 定义学习率调度器
    min_learning_rate = 0.00001  # 设置最低学习率
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3,
                                               verbose=True, min_lr=min_learning_rate)

    loss_weight = 1.0  # 可调整，控制模型的训练误差损失
    kl_div_weight = 0.8  # 可调整，控制训练数据的活性与预测标签的活性的分布吻合
    Rmse_weight = 1.5  #  可调整确保其中的相关性在0.6水平之上

    for epoch in range(epochs):
        # 存储每个 epoch 中的损失
        train_loss_loss = []
        val_loss = []

        train_loss2_loss = []
        val_loss2_loss = []

        train_mape_loss = []
        val_mape_loss = []



        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            x, y = (i.to(device) for i in data)  # 取出数据
            outputs = model(x)  # 得到预测值

            # 计算 KL 散度
            outputs_r = F.softmax(outputs.float().squeeze(), dim=-1)
            y_r =  F.softmax(y.float().squeeze(), dim=-1)
            kl_div = F.kl_div(y_r.log(), outputs_r, reduction='batchmean')

            # 计算 SL1
            outputs = outputs.float()  # 将模型输出转换为Float类型
            y = y.unsqueeze(1).float()  # 将y的形状从[64]转换为[64, 1]并转换为Float类型

            loss = criterion(outputs, y)  # 1 计算当次的损失
            #train_loss_loss.append(loss.item())  # 存储当次的损失

            loss2 = rmse_loss(outputs, y)  # 3 计算 RMSE
            train_loss2_loss.append(loss2.item())  # 存储当次的 RMSE 损失

            optimizer.zero_grad()
            #loss.backward()
            weighted_loss = loss_weight * loss + kl_div_weight * kl_div + Rmse_weight * loss2
            train_loss_loss.append(weighted_loss.item())  # 存储当次的损失

            weighted_loss.backward()
            optimizer.step()

        # 计算每个 epoch 中的平均损失
        avg_train_loss_loss = torch.mean(torch.tensor(train_loss_loss))
        train_loss_res.append(avg_train_loss_loss.item())  # 存储每个 epoch 的平均损失

        avg_train_mape_loss = torch.mean(torch.tensor(train_mape_loss))
        train_mape_res.append(avg_train_mape_loss.item())  # 存储每个 epoch 的平均 MAPE

        avg_train_loss2_loss = torch.mean(torch.tensor(train_loss2_loss))
        train_loss2_res.append(avg_train_loss2_loss.item())  # 存储每个 epoch 的平均 RMSE 损失

        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                x, y = (i.to(device) for i in data)
                outputs = model(x)

                # 计算 KL 散度
                outputs_r = F.softmax(outputs.float().squeeze(), dim=-1)
                y_r = F.softmax(y.float().squeeze(), dim=-1)
                kl_div = F.kl_div(y_r.log(), outputs_r, reduction='batchmean')

                # 计算 SL1
                outputs = outputs.float()  # 将模型输出转换为Float类型
                y = y.unsqueeze(1).float()  # 将y的形状从[64]转换为[64, 1]并转换为Float类型

                val_l = criterion(outputs, y)  # 1 计算当次的损失
                #val_loss.append(val_l.item())  # 存储这个 batch 的损失

                # val_mape = calculate_mape(outputs, y)  # 2 计算当次的 MAPE
                # val_mape_loss.append(val_mape)

                weighted_loss = loss_weight * val_l + kl_div_weight * kl_div + Rmse_weight * loss2
                val_loss.append(weighted_loss.item())

                val_l2 = rmse_loss(outputs, y)  # 3 计算当次的 RMSE
                val_loss2_loss.append(val_l2.item())  # 存储这个 batch 的 RMSE 损失


            #avg_val_loss = torch.mean(torch.tensor(val_loss))  # 计算验证集上的平均损失
            avg_val_loss = torch.mean(torch.tensor(val_loss))  # 计算验证集上的平均损失
            val_loss_res.append(avg_val_loss.item())  # 存储验证集上的平均损失

            avg_val_mape_loss = torch.mean(torch.tensor(val_mape_loss))
            val_mape_res.append(avg_val_mape_loss.item())  # 存储验证集上的平均 MAPE

            avg_val_loss2 = torch.mean(torch.tensor(val_loss2_loss))  # 计算验证集上的平均 RMSE
            val_loss2_res.append(avg_val_loss2.item())  # 存储验证集上的平均 RMSE 损失

        # 调用学习率调度器
        scheduler.step(avg_val_loss)  # 这里可以使用验证集上的损失进行调度

        # 保存最佳模型
        if avg_val_loss < best_loss:
            print(f'best loss {avg_val_loss:.5f}')
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_as_filename)

        # 打印结果，包含学习率信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}, '
              f'Learning Rate: {current_lr:.6f}, '
              f'Train SmoothL1: {avg_train_loss_loss:.5f}, '
              f'Train RMSE: {avg_train_loss2_loss:.3f}, '
              f'Validation SmoothL1: {avg_val_loss:.5f}, '
              f'Val RMSE: {avg_val_loss2:.3f}'
              )

        # 返回 loss 列表
    return train_loss_res, train_loss2_res, val_loss_res, val_loss2_res, save_model_as_filename

def test_model(model_path, model, test_dataloader, criterion, device):
    # 加载模型参数
    ckpt = torch.load(model_path, map_location=device)  # 添加map_location参数
    model.load_state_dict(ckpt)
    model.eval()

    test_loss = []
    test_rmse = []

    loss_weight = 1.0  # 权重可以根据需求进行调整
    kl_div_weight = 0.0  # 权重可以根据需求进行调整
    Rmse_weight = 0.2

    # 测试
    with torch.no_grad():
        for data in test_dataloader:
            x, y = (i.to(device) for i in data)
            # 前向传播
            outputs = model(x)

            # 计算 KL 散度
            outputs_r = F.softmax(outputs.float().squeeze(), dim=-1)
            y_r = F.softmax(y.float().squeeze(), dim=-1)
            kl_div = F.kl_div(y_r.log(), outputs_r, reduction='batchmean')

            outputs = outputs.float()  # 将模型输出转换为Float类型
            y = y.unsqueeze(1).float()  # 将y的形状从[64]转换为[64, 1]并转换为Float类型

            # 计算损失
            loss = criterion(outputs, y)
            loss2 = rmse_loss(outputs, y)
            weighted_loss = loss_weight * loss + kl_div_weight * kl_div + Rmse_weight * loss2

            #test_loss.append(loss.item())
            test_loss.append(weighted_loss.item())
            test_rmse.append(loss2.item())  # 使用loss2.item()获取RMSE的数值

    print(f'Test SmoothL1: {np.mean(test_loss):.3f}, '
          f'Test RMSE: {np.mean(test_rmse):.3f}'
          )

    return np.mean(test_loss), np.mean(test_rmse)


def save_data_to_pickle(save_path, train_loss_res, val_loss_res, train_loss2_res, val_loss2_res, test_loss, test_rmse):
    # 数据存储
    data = {
        "train_loss1_res": train_loss_res,
        "val_loss1_res": val_loss_res,
        "train_loss2_res": train_loss2_res,
        "val_loss2_res": val_loss2_res,
        "test_loss1": test_loss,
        "test_loss2": test_rmse
    }

    with open(save_path, "wb") as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    # ________________ Transfomer
    # 1 读取配置文件 trans
    config = json.load(open('Data/Params/T6.json', 'r'))  # 打开数据

    # 选择模型
    selected_model = "model-T6"

    # 从配置中获取选定模型的参数
    model_params_selected = config["model_params"][selected_model]
    vs = model_params_selected["vs"]
    ed = model_params_selected["ed"]
    hs = model_params_selected["hs"]
    nl1 = model_params_selected["nl1"]
    nl2 = model_params_selected["nl2"]
    nh = model_params_selected["nh"]
    dp = model_params_selected["dp"]
    ler = model_params_selected["ler"]
    batch_size = model_params_selected["batch_size"]
    train_rate = model_params_selected["train_rate"]
    val_rate = model_params_selected["val_rate"]
    device = model_params_selected["device"]
    epochs = model_params_selected["epochs"]
    folds = model_params_selected["folds"]
    #
    # 从配置中获取与路径相关的参数
    file_path = model_params_selected["file_path"]
    save_path = model_params_selected["save_path"]
    save_data_path = model_params_selected["save_data_path"]

    # 2 初始化模型
    # model = TransGRU_1(vocab_size=vs, embed_dim=ed, hidden_size=hs, num_layers_1=nl1, num_layers_2=nl2,
    #                             dropout=dp).to(device)#1.06--0.95
    model = Transformer_Model_6(vocab_size=vs, embed_dim=ed, hidden_size=hs, num_layers_1=nl1, num_layers_2=nl2,
                                num_head=nh, dropout=dp).to(device)  # 0.96--0.91
    # model = TransGRU_1(vocab_size=vs, embed_dim=ed, hidden_size=hs, num_layers_1=nl1, num_layers_2=nl2,
    #                    dropout=dp).to(device)

    criterion = nn.SmoothL1Loss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=ler)  # 优化器


    # # 3 获取训练、验证、测试的数据
    # for i in range(folds):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=ler)
    #     train_dataloader, val_dataloader, test_dataloader = get_data(file_path, train_rate, val_rate, batch_size, 'A')
    #
    #
    #
    #     if i == 0:
    #         # 4 进行训练和验证
    #         train_loss_res, train_loss2_res, val_loss_res, val_loss2_res, save_path = train_and_validate_model(model,
    #                                                                                                            train_dataloader,
    #                                                                                                            val_dataloader,
    #                                                                                                            criterion,
    #                                                                                                            optimizer,                                                                                                           device, epochs,
    #                                                                                                            save_path)
    #     else:
    #         ckpt = torch.load(save_path, map_location=device)  # 添加map_location参数
    #         model.load_state_dict(ckpt)
    #         # 4 进行训练和验证
    #         train_loss_res_, train_loss2_res_, val_loss_res_, val_loss2_res_, save_path = train_and_validate_model(model,
    #                                                                                                            train_dataloader,
    #                                                                                                            val_dataloader,
    #                                                                                                            criterion,
    #                                                                                                            optimizer,
    #                                                                                                            device, epochs,
    #                                                                                                            save_path)
    #         train_loss_res.extend(train_loss_res_)
    #         train_loss2_res.extend(train_loss2_res_)
    #         val_loss_res.extend(val_loss_res_)
    #         val_loss2_res.extend(val_loss2_res_)
    #
    #
    # # 5进行测试
    # test_loss, test_loss2 = test_model(save_path, model, test_dataloader, criterion, device)
    #
    # # 6 储存训练后的结果
    # save_data_to_pickle(save_data_path, train_loss_res, val_loss_res, train_loss2_res, val_loss2_res, test_loss,
    #                     test_loss2)

    # ———————————————————————————————————————————————————生成数据预测—————————————————————————————————————————————————————

    # 在随机生成的数据上进行预测
    # 导入数据
    get_pre_data = json.load(open('PreData&Res/Data/1000wData.json', 'r'))
    pre_dataset = Amino(get_pre_data)
    pre_dataloader = DataLoader(pre_dataset, shuffle=False, batch_size=256)  # 每个 epoch 再打乱

    # 导入模型
    ckpt = torch.load("Data/Result/TransGru/2GRU+Trans/A/5-fold/model.pt", map_location=device)  # 添加map_location参数
    model.load_state_dict(ckpt)
    model.eval()

    predicted_output = []
    with torch.no_grad():
        for data in pre_dataloader:
            x, y = (i.to(device) for i in data)
            # 前向传播
            outputs = model(x)
            predicted_output.extend(outputs.squeeze().tolist())  # 将预测的输出扁平化并添加到列表中

    # print(get_data)
    for i in range(len(get_pre_data)):
        get_pre_data[i]['nor_package'] = predicted_output[i]

    # 储存结果
    # df = pd.DataFrame(get_pre_data)
    # df.to_excel('PreData&Res/T3/1000wresult.xlsx', index=False)
    with open('Data/Result/TransGru/2GRU+Trans/A/5-fold/1000w-result.json', 'w') as f:
        json.dump(get_pre_data, f)

    # ———————————————————————————————————————————————————测试数据预测—————————————————————————————————————————————————————
    # 在测试集上进行预测
    # 导入模型
    train_dataloader, val_dataloader, test_dataloader = get_data(file_path, train_rate, val_rate, batch_size, 'A')
    ckpt = torch.load("Data/Result/TransGru/2GRU+Trans/6/model.pt", map_location=device)  # 添加map_location参数

    model.load_state_dict(ckpt)
    model.eval()

    # 测试集上预测
    x_all = []  # 存储x的每一行数据
    y_all = []  # 存储y的每一行数据

    # 创建一个空列表来存储输出值
    output_list = []

    # 使用 torch.no_grad() 来禁用梯度计算
    with torch.no_grad():
        for data in test_dataloader:
            x, y = (i.to(device) for i in data)
            outputs = model(x)

            x = x.tolist()  # 将x转换为Python列表
            y = y.tolist()  # 将y转换为Python列表

            # 将x和y的每一行数据依次添加到x_all和y_all中
            for x_row, y_val in zip(x, y):
                x_all.append(x_row)
                y_all.append(y_val)

            # 将输出张量转换为 Python 列表，并将其添加到 output_list 中
            output_list.extend(outputs.squeeze().tolist())

    # 现在，output_list 包含了来自 'outputs' 张量的值，以 Python 列表的形式呈现，其中每个元素对应一个输出值。

    import pandas as pd

    # 将列表转换为 DataFrame
    df = pd.DataFrame({
        'aa': x_all,
        'true_y': y_all,
        'pre_y': output_list
    })

    # 将 DataFrame 保存为 Excel 文件
    df.to_excel('test_result.xlsx', index=False)

    # 计算相关性
    correlation = df['true_y'].corr(df['pre_y'])
    print("相关性：", correlation)


