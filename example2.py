# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter


from moe import MoE
from mmoe import MmoE
from datasets import MyDataset

def initiate_datasets(file_name,input_size,division=0):
    df = pd.read_csv(file_name + ".csv", encoding="utf-8")
    #如果要split trian 和test，就在这里把df分成df_train和df_test，再传入MyDataset
    train_dataset = MyDataset(df,input_size,style="train")
    ori_dataset = MyDataset(df,input_size,style="ori")
    return train_dataset, ori_dataset

def train4epoch(model, train_loader, loss_fn1,loss_fn2,optim, loss1_fraction=0.5):
    model.train()
    batch_loss1 = batch_loss2 = 0
    for i, (data, y1, y2) in enumerate(train_loader):
        data, y1, y2 = data.to(device).float(), y1.to(device),y2.to(device)
        # model = train(data.to(device), y1.to(device),y2.to(device), model, loss_fn1,loss_fn2, optim, i)
        y_hat1, y_hat2 = model(data)
        y_hat1, y_hat2 = y_hat1.to(torch.float16), y_hat2.to(torch.float16)
        # print(y_hat1,y1)
        # print("y_hat1, y_hat2", y_hat1, y_hat2)
        y1, y2 = y1.flatten(), y2.flatten()
        # print("y1, y2", y1, y2)
        loss1, loss2 = loss_fn1(y_hat1, y1), loss_fn2(y_hat2, y2)
        batch_loss1, batch_loss2 = loss1.item()+batch_loss1, loss2.item()+batch_loss2
        total_loss = loss1_fraction*loss1 + (1-loss1_fraction)*loss2
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        if i % 1000 == 0:
            print("{} batch Training Results - batch_loss1: {:.2f}, batch_loss2: {:.3f}"
                  .format(i, batch_loss1, batch_loss2))
            # writer.add_scalar("train_loss",loss1.item(),loss2.item(),batch_No)
            batch_loss1 = batch_loss2 = 0
        return model

now = datetime.now()
time_str = "_" + now.strftime("(%d-%m-%Y-%H)")
print("test for time_str: ", time_str)
# with open("./log{0}.txt".format(time_str), 'w') as f:
#     oldstdout = sys.stdout
#     sys.stdout = f

# arguments
epoch = 12
input_size = 42
num_classes = 2
num_experts = 10
moe_output_size = 30
moe_hidden_size = 70
mlp_hidden_size = 45
batch_size = 16
k = 4
loss1_fraction = 0.5
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# determine device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# instantiate the MoE layer
model = MmoE(input_size,moe_output_size,num_classes,num_experts,moe_hidden_size,mlp_hidden_size)
model = model.to(device)
optim = Adam(model.parameters())

file_name = "../LongHu/Data_Preprocess/data/424data_1w_output_expanded_cleaned_prepared"
# file_name = "../LongHu/Data_Preprocess/data/424data_1w_output_expanded_cleaned_prepared"
train_dataset, ori_dataset = initiate_datasets(file_name,input_size)
loss_fn1 = nn.CrossEntropyLoss()
loss_fn2 = nn.CrossEntropyLoss() #nn.NLLLoss()
#weight=torch.tensor([1,dataset.neg_pos_ratio[0]],device=device,dtype=torch.float16)
len_train_dataset, len_test_dataset = len(train_dataset),len(ori_dataset)
print("length of train_dataset: {}".format(len_train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=True)
# writer = SummaryWriter("./logs")
print('there are total {0}:{1} batches for train'.format(len_train_dataset,len(train_loader)))

# train
for j in range(epoch):
    print("-------epoch  {} -------".format(j+1))
    train4epoch(model, train_loader,loss_fn1,loss_fn2,optim,loss1_fraction)

    # evaluate
    model.eval()     #实际调用model.train(False)
    total_test_loss = [0,0]
    total_accuracy = [0,0]
    TF_counter = [[[0,0],[0,0]],[[0,0],[0,0]]]
    with torch.no_grad():    #禁止梯度的计算
        for i, (data, y1, y2) in enumerate(test_loader):
            y_hat1, y_hat2 = model(data.float().to(device))
            y_hat1, y_hat2 = y_hat1.to(torch.float16), y_hat2.to(torch.float16)
            y1, y2 = y1.flatten().to(device), y2.flatten().to(device)
            loss1, loss2 = loss_fn1(y_hat1, y1), loss_fn2(y_hat2, y2)
            total_test_loss[0] += loss1
            total_test_loss[1] += loss2
            y1, y2, y_hat1, y_hat2= y1.cpu(), y2.cpu(),y_hat1.cpu(),y_hat2.cpu()
            correct1, correct2 = (y_hat1.argmax(1) == y1).sum(),(y_hat2.argmax(1) == y2).sum()
            preT_actT1 = np.logical_and(y_hat1.argmax(1) == 1,y1==1).sum()
            preT_actT2 = np.logical_and(y_hat2.argmax(1) == 1,y2==1).sum()
            preT_actF1 = np.logical_and(y_hat1.argmax(1) == 1,y1==0).sum()
            preT_actF2 = np.logical_and(y_hat2.argmax(1) == 1,y2==0).sum()
            preF_actT1 = np.logical_and(y_hat1.argmax(1) == 0, y1 == 1).sum()
            preF_actT2 = np.logical_and(y_hat2.argmax(1) == 0, y2 == 1).sum()
            TF_counter[0][1][1] += preT_actT1
            TF_counter[1][1][1] += preT_actT2
            TF_counter[0][1][0] += preT_actF1
            TF_counter[1][1][0] += preT_actF2
            TF_counter[0][0][1] += preF_actT1
            TF_counter[1][0][1] += preF_actT2
            TF_counter[0][0][0] += correct1-preT_actT1
            TF_counter[1][0][0] += correct2-preT_actT2
            total_accuracy[0] += correct1
            total_accuracy[1] += correct2
    print("test set Loss: label 1:{0[0]:.3f}, label 2:{0[1]:.3f}".format(total_test_loss))
    print("test set accuracy: \nlabel 1: {0[0]:d}/{1} = {2[0]:.3f}\n"
          "label 2: {0[1]:d}/{1:d} = {2[1]:.3f}".format(total_accuracy, len_test_dataset, [acc / len_test_dataset for acc in total_accuracy]))
    print("test set TF_counter of label 1:\n{0:d}\t{1:d}\n{2:d}\t{3:d}".format(TF_counter[0][0][0],TF_counter[0][0][1],
                                                                       TF_counter[0][1][0],TF_counter[0][1][1]))
    print("test set TF_counter of label 2:\n{0:d}\t{1:d}\n{2:d}\t{3:d}".format(TF_counter[1][0][0], TF_counter[1][0][1],
                                                                       TF_counter[1][1][0], TF_counter[1][1][1]))
    print("first label recall:{0:.3f}, precision:{1:.3f}".format(
        TF_counter[0][1][1]/(TF_counter[0][1][1]+TF_counter[0][0][1]),
        TF_counter[0][1][1]/(TF_counter[0][1][1]+TF_counter[0][1][0])))
    print("second label recall:{0:.3f}, precision:{1:.3f}".format(
        TF_counter[1][1][1] / (TF_counter[1][1][1] + TF_counter[1][0][1]),
        TF_counter[1][1][1] / (TF_counter[1][1][1] + TF_counter[1][1][0])))
    # writer.add_scalar("test_loss", total_test_loss)
    # writer.add_scalar("test set accuracy: {}/{} = {}".format(total_accuracy,length_dataset,[acc/length_dataset for acc in total_accuracy]))
    torch.save(model, "./saved_model/model{0}_{1}.pth".format(time_str, j+1))
# writer.close()