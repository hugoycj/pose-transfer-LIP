import sys
sys.path.append("/home/hugoycj/Insync/118010378@link.cuhk.edu.cn/OneDrive Biz/06Workspace/02Internship/064Parsing/Implementation_Code/pose-transfer-LIP")
sys.path.append('../')

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from base import BaseModel
from model_factory.segmentation_head import deeplabv3_resnet50
from model_factory.segmentation_head.deeplabv3 import DeepLabHead


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DeeplabV3PlusModel(BaseModel):
    def __init__(self, num_classes=20):
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=True, progress=True)
        self.model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        return self.model(x)['out']

if __name__ == '__main__':
    im_size = 320
    model = DeeplabV3PlusModel()
    summary(model, input_size=(3, im_size, im_size))