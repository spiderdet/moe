# -*- coding: utf-8 -*-
import torch.utils.data as data
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler
import torch

class MyDataset(data.Dataset):
    def __init__(self, df, input_size = 42, labels=None, style = "ori"):
        # self.file_name = file_name
        if labels is None:
            labels = ['expanded_vqr', 'view_qyrg'] # ['expanded_vqr','view_ornot','qyrg']
        self.df = df
        self.rows = self.df.shape[0]
        self.neg_pos_ratio = [0]*len(labels)
        for i,label in enumerate(labels):
            neg_num = (self.df[label] == 0).sum()
            self.neg_pos_ratio[i] = neg_num/(self.rows-neg_num)
        self.X_num = input_size
        self.labels = labels
        # print(self.train_df.columns)
        # print([label in self.train_df.columns for label in self.labels])
        self.preprocess(style)

    def __getitem__(self, index):
        x = torch.tensor(self.df.iloc[index, :self.X_num].to_numpy(), dtype=torch.float16)
        # print(self.train_df.columns)
        # label = torch.tensor([label in self.train_df.columns for label in self.labels],dtype=torch.long)
        y = torch.tensor(self.df.loc[index, self.labels], dtype=torch.long)
        # print("index: ", index)
        # print(x, y)
        # print(list(x.size()), list(y.size()))
        if len(list(y.size())) >1:
            y1, y2 = torch.split(y, 1, dim=1)
            y1 = y1.flatten()
            y2 = y2.flatten()
        else:
            y1, y2 = torch.split(y,1,dim=0)
        # print(y1,y2)
        return x, y1, y2

    def __len__(self):
        return self.df.shape[0]

    def process_ID_variable(self):
        ID_col_names = ['age', 'nowbiz', 'knowway', 'purpose', 'workjob', 'waydivide',
                        'projectStatus', 'familyrealtion', 'lastreportchannel', 'firstreportchannel',
                        'lastreportsecondchannel', 'firstreportsecondchannel', 'projectId',
                        'internalId']
        infile = open("ID_encoder", 'rb')
        enc = pickle.load(infile)
        ms = pickle.load(infile)
        infile.close()
        self.df.iloc[:, :self.X_num] = enc.transform(self.df.iloc[:, :self.X_num])
        self.df[ID_col_names] = ms.transform(self.df[ID_col_names])
    def process_imbalance(self):
        ros = RandomOverSampler(sampling_strategy="auto", random_state=0)
        x, y = self.df, self.df[self.labels[0]]
        x, y = ros.fit_resample(x, y)
        self.df= pd.DataFrame(x)
        # print("after process_imbalance, self.train_df.rows: ", self.train_df.shape[0],(self.train_df[self.labels[0]]==1).sum())
    def preprocess(self,style):
        self.process_ID_variable()
        if style == "train":
            self.process_imbalance()
        elif style == "ori":
            return
        else:
            print("warning, not specified style")
if __name__ == "__main__":
    dataset = MyDataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    print('there are total {0}:{1} batches for train'.format(len(dataset),len(train_loader)))

    for i, (data, y1,y2) in enumerate(train_loader):
        if i>0:
            break
        print(data.size(), y1.size(),y2.size())