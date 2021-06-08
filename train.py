import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Classifier, DatasetCustom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = DatasetCustom(file_path='./data/train.csv')


dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

classifier = Classifier(input_size=75, num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(classifier.parameters(), lr=0.001)
EPOCH = 10

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(dataloader):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        output = classifier(x)
        loss = criterion(output,  torch.max(y, 1)[1])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (step+1) % 100 == 0:
            print(f"epoch : {epoch}, loss: {loss.item()}")


