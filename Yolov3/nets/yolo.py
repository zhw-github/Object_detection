import sys
sys.path.append("../")

import torch
import math
import torch.nn as nn
from collections import OrderedDict
from nets.darknet import darknet53


def ConvBNLRelu(in_channels, out_channels, kernel_size):
    padding = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size,1,padding=padding,bias=False)),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.LeakyReLU(0.1))
    ]))


"""
maker_last_layers里面一共有7个卷积
前五个是 Conv set用来提取特征
后两个用来获得yolo网络的预测结果
"""
def make_last_layers(channels_list,in_channels,out_channels):
    m = nn.Sequential(
        ConvBNLRelu(in_channels, channels_list[0], 1),
        ConvBNLRelu(channels_list[0], channels_list[1], 3),
        ConvBNLRelu(channels_list[1], channels_list[0], 1),
        ConvBNLRelu(channels_list[0], channels_list[1], 3),
        ConvBNLRelu(channels_list[1], channels_list[0], 1),

        ConvBNLRelu(channels_list[0], channels_list[1], 3),
        ConvBNLRelu(channels_list[1], out_channels, 1)

    )
    
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        self.backbone = darknet53()
        # [64, 128, 256, 512, 1024]
        out_channels = self.backbone.layers_out_filters
        self.last_layer0 = make_last_layers([512,1024], out_channels[-1], len(anchors_mask)*(5+num_classes))
        
        self.last_layer1_conv = nn.Conv2d(512,256,1)
        self.last_layer1_upsampling = nn.Upsample(scale_factor=2,mode="nearest")
        self.last_layer1 = make_last_layers([256,512], out_channels[-2]+256, len(anchors_mask)*(5+num_classes))

        self.last_layer2_conv = nn.Conv2d(256,128,1)
        self.last_layer2_upsampling = nn.Upsample(scale_factor=2,mode="nearest")
        self.last_layer2 = make_last_layers([128,256], out_channels[-3]+128, len(anchors_mask)*(5+num_classes))




    def forward(self, x):
        x2,x1,x0 = self.backbone(x)
        # 1024,13,13->512,13,13->1024,13,13->512,13,13->1024,13,13->512,13,13
        out0_branch = self.last_layer0[:5](x0)
        # 512,13,13->1024,13,13->75,13,13
        out0 = self.last_layer0[5:](out0_branch)


        # 512,13,13 -> 256,13,13
        x1_in = self.last_layer1_conv(out0_branch)
        # 256,13,13->256,26,26
        x1_in = self.last_layer1_upsampling(x1_in)

        # 256,26,26->512+256,26,26
        x1_in = torch.cat([x1,x1_in],dim=1)
        # 768,26,26->256,26,26->512,26,26->256,26,26->512,26,26->256,26,26    
        out1_branch = self.last_layer1[:5](x1_in)
        # 256,26,26->512,26,26->75,26,26
        out1 = self.last_layer1[5:](out1_branch)

        # 256,26,26 -> 128,26,26
        x2_in = self.last_layer2_conv(out1_branch)
        # 128,26,26 -> 128,52,52
        x2_in = self.last_layer2_upsampling(x2_in)
        # 128,52,52 -> 128+256,52,52
        x2_in = torch.cat([x2,x2_in],dim=1)

        # 384,52,52->128,52,52->256,52,52->128,52,52->256,52,52->128,52,52
        # 128,52,52->256,52,52->75,52,52
        out2 = self.last_layer2(x2_in)




        return out2,out1,out0









def main():
    t = torch.randn([4,3,416,416])
    model = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 20)
    out2,out1,out0 = model(t)
    print("out0: ", out0.shape)
    print("out1: ", out1.shape)
    print("out2: ", out2.shape)




if __name__ == "__main__":
    main()