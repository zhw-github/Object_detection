import torch
import torch.nn as nn
from collections import OrderedDict
import math
"""
残差结构
    利用1*1卷积降维度
    再利用3*3卷积升维
"""
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1,
                            stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual

        return out

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        layers = [1,2,8,8,4]
        self.in_channels = 32

        # 3,416,416 -> 32,416,416
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.LeakyReLU(0.1)

        # 32,416,416 -> 64,208,208
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 64,208,208, -> 128,104,104
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 128,104,104 -> 256,52,52
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 256,52,52-> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 512,26,26 -> 1024,13,13
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    """
    在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    然后进行残差结构的堆叠
    """
    def _make_layer(self, out_channels, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv",nn.Conv2d(self.in_channels,out_channels[1],3,2,1,bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(out_channels[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差
        self.in_channels = out_channels[1]
        for i in range(blocks):
            layers.append(("residual_{}".format(i),BasicBlock(self.in_channels, out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet()
    return model

def main():
    t  = torch.randn([4,3,416,416])
    model = DarkNet()
    out3, out4, out5 = model(t)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)




if __name__ == "__main__":
    main()