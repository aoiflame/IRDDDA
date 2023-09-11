import torch
import torch.nn as nn
from torch.autograd import Function
from collections import Counter
import torch.nn.functional as F
import numpy as np

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.view(-1,64)
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(in_features=192, out_features=2, bias=True)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        return x


class Classifier(nn.Module):
    def __init__(self, classnum=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.fc3 = nn.Linear(in_features=16, out_features=classnum, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = self.fc3(x)
        return y


class DANN(nn.Module):

    def __init__(self, device, classnum):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier(classnum)
        self.domain_classifier = Discriminator(input_dim=64, hidden_dim=32)

    def forward(self, input_data, alpha=1, source=True, only_resampled=False):
        if (only_resampled == False):           
            feature = self.feature(input_data)
            tem_feature = feature
            feature = feature.view(-1, 64)
            class_output = self.classifier(feature)
            domain_output, domain_pred = self.get_adversarial_result(feature, source, alpha)
            return class_output, domain_output, tem_feature, None, None, domain_pred
        else:
            input_data = input_data.view(-1, 64)
            class_output = self.classifier(input_data)
            domain_output, domain_pred = self.get_adversarial_result(input_data, source, alpha)
            return class_output, domain_output, None, None, None, domain_pred
        

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.unsqueeze(1).float())
        return loss_adv, domain_pred

class CNN(nn.Module):
    def __init__(self, device, classnum):
        super(CNN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier(classnum)

    def forward(self, input_data):
        feature = self.feature(input_data)
        tem_feature = feature
        feature = feature.view(-1, 64)
        class_output = self.classifier(feature)
        return class_output, tem_feature, None, None
