import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 250)
        self.fc3 = nn.Linear(250, 9)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DatasetCustom(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.drop("id", axis=1, inplace=True)
        self.labels = pd.get_dummies(self.data.target, prefix='class')
        self.data.drop("target", axis=1, inplace=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        features = self.data.iloc[index, :].values.astype(np.float32)
        label = self.labels.iloc[index, :].values
        return features, label


