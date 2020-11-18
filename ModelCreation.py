import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self,out1=16,out2=32,out3=64,out4=128,no_of_classes=10):
        super(CNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=out1,kernel_size=5,stride=1,padding=2)
        self.cnn1_batchNorm = nn.BatchNorm2d(out1)
        self.cnn1_maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn1_dropout = nn.Dropout(0.25)

        self.cnn2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=5, stride=1, padding=2)
        self.cnn2_batchNorm = nn.BatchNorm2d(out2)
        self.cnn2_maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn2_dropout = nn.Dropout(0.25)

        self.cnn3 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=5, stride=1, padding=2)
        self.cnn3_batchNorm = nn.BatchNorm2d(out3)
        self.cnn3_maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn3_dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(in_features=out3*4,out_features=out3*8)
        self.fc1_batchNorm = nn.BatchNorm1d(out3*8)
        self.cnn4_dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=out3 * 8, out_features=no_of_classes)
        self.fc2_batchNorm = nn.BatchNorm1d(no_of_classes)

    def forward(self,x):
        x = self.cnn1(x)
        x = self.cnn1_batchNorm(x)
        x = torch.relu(x)
        x = self.cnn1_maxpool(x)
        x = self.cnn1_dropout(x)

        x = self.cnn2(x)
        x = self.cnn2_batchNorm(x)
        x = torch.relu(x)
        x = self.cnn2_maxpool(x)
        x = self.cnn2_dropout(x)

        x = self.cnn3(x)
        x = self.cnn3_batchNorm(x)
        x = torch.relu(x)
        x = self.cnn3_maxpool(x)
        x = self.cnn3_dropout(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.fc1_batchNorm(x)
        x = self.cnn4_dropout(x)
        x = self.fc2(x)
        x = self.fc2_batchNorm(x)

        return x