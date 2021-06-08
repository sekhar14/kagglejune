import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.0):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.output = nn.Linear(8, num_classes)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout_layer(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout_layer(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.dropout_layer(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.output(out)
        return out


class DatasetCustom(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.drop("id", axis=1, inplace=True)
        self.labels = pd.get_dummies(self.data.target, prefix='class')
        self.data.drop("target", axis=1, inplace=True)
        scaler = MinMaxScaler()
        self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns])
        print("Preprocessing min_max_scaler :::::: ")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        features = self.data.iloc[index, :].values
        label = self.labels.iloc[index, :].values
        return features, label


