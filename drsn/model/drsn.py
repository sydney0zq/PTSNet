import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

import os, sys, functools
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from inplace_ABN import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

import warnings
warnings.filterwarnings('ignore')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        endpoints = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        endpoints.append(x)
        x = self.layer2(x)
        endpoints.append(x)
        x = self.layer3(x)
        endpoints.append(x)
        x = self.layer4(x)
        endpoints.append(x)

        return endpoints

class ResidualBlock(nn.Module):
    def __init__(self, v):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(v, v, kernel_size=3, padding=1, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(v, v, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        x = x + self.res(x)
        return x

class GlobalConvolutionBlock(nn.Module):
    def __init__(self, in_dim, out_dim=256, k=7):
        super(GlobalConvolutionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1,k), padding=(0, k//2), bias=True), 
            nn.Conv2d(out_dim, out_dim, kernel_size=(k,1), padding=(k//2, 0), bias=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(k,1), padding=(0, k//2), bias=True), 
            nn.Conv2d(out_dim, out_dim, kernel_size=(1,k), padding=(k//2, 0), bias=True))

        self.RB = ResidualBlock(out_dim)

    def forward(self, x):
        out = self.branch1(x) + self.branch2(x)
        out = self.RB(out)
        return out


class RefinementModule(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super(RefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=True)
        self.RB1 = ResidualBlock(out_dim)
        self.RB2 = ResidualBlock(out_dim)

    def forward(self, x_top, x_low):
        _, _, h, w = x_low.size()

        x_top = F.upsample(x_top, size=(h, w), mode='bilinear')
        x_low = self.RB1(self.conv(x_low))
        x = x_top + x_low
        x = self.RB2(x)
        return x

class DRSN(nn.Module):

    def __init__(self):
        super(DRSN, self).__init__()

        self.features = ResNet(Bottleneck, [3, 4, 6, 3])
        self.visual_features = self.features #ResNet(Bottleneck, [3, 4, 6, 3])
        
        self.GCB = GlobalConvolutionBlock(8192)
        self.RM1 = RefinementModule(1024)
        self.RM2 = RefinementModule(512)
        self.RM3 = RefinementModule(256)

        self.classfier = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.dropout2d = nn.Dropout2d(p=0.5)

    def generate_visual_params(self, guide_image):
        visual_features = self.visual_features(guide_image)[-1]
        return visual_features

    def forward(self, ref_imask, p_imask, q_imask, x):
        _, _, h, w = x.size()
        visual_fea_list = []
        for g_imask in [ref_imask, p_imask, q_imask]:
            visual_fea_list.append(self.generate_visual_params(g_imask))
        visual_features = torch.cat(visual_fea_list, 1)

        feas = self.features(x)
        x = torch.cat((feas[-1],visual_features), 1)
        x = self.GCB(x)
        x = self.RM1(x, feas[-2])
        x = self.RM2(x, feas[-3])
        x = self.RM3(x, feas[-4])
        
        x = self.dropout2d(x)
        out = F.upsample(input=self.classfier(x), size=(h, w), mode='bilinear')

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, InPlaceABNSync):
                m.eval()
    
    def init(self, model_path, stage='yvos_train', mgpus=True):
        if stage == 'yvos_train':
            saved_state_dict = torch.load(model_path)
            new_params = self.features.state_dict().copy()
            
            for key1, key2 in zip(new_params, saved_state_dict):
                i_parts = key1.split('.')
                if i_parts[0] == 'conv1':
                    new_params[key1] = torch.cat((saved_state_dict[key1].data, torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)
                else:
                    new_params[key1] = saved_state_dict[key1]
            self.features.load_state_dict(new_params)
        elif stage == 'davis_train' or stage == 'test':
            saved_state_dict = torch.load(model_path)
            cur_state_dict = self.state_dict().copy()
            for k in cur_state_dict.keys():
                if mgpus:
                    cur_k = 'module.'+k
                else:
                    cur_k = k
                cur_state_dict[k] = saved_state_dict[cur_k]
            self.load_state_dict(cur_state_dict)
        else:
            assert False, 'Stage {} not recognized...'.format(stage)


if __name__ == "__main__":
    test = DRSN()
    test.init('../init_models/resnet50-19c8e357.pth')
    #x = torch.ones((1, 4, 256, 256)).type(torch.cuda.FloatTensor)
    #test.cuda()
    #test(x, x, x, x)



