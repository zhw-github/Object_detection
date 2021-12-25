import sys
sys.path.append("../")
from collections import OrderedDict
import torch
import torch.nn as nn
from nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size- 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

"""
maker_last_layers里面一共有7个卷积
前五个是 Conv set用来提取特征
后两个用来获得yolo网络的预测结果
"""
def make_last_layers(filters_list,in_filters,out_filter):
    m = nn.Sequential(  
        conv2d(in_filters,filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        
        conv2d(filters_list[0],filters_list[1],3),
        nn.Conv2d(filters_list[1],out_filter,1,1,0,bias=True)
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        """
        生成darknet53backbone模型
        获得三个特征层，他们的reshape分别是:
        256,52,52 
        512,26,26
        1024,13,13        
        """
        self.backbone = darknet53()
        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters
        """
        计算yolo_head的输出通道数，对于voc数据集而言
        final_out_filter0 = final_out_filter1 = final_out_filter2 =(3*(20+5))=75
        """
        self.last_layer0 = make_last_layers([512, 1024], 
                            out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        
        self.last_layer1_conv = conv2d(512,256,1) 
        self.last_layer1_upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], 
                            out_filters[-2]+256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv  = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], 
                        out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))


    def forward(self, x):
        # 获得三个特征层，shape: 256,52,52 512,26,26 1024,13,13
        x2,x1,x0 = self.backbone(x)

        """
        第一个特征 out0 = (batch_size,75,13,13)
        """
        # 1024,13,13->512,13,13->1024,13,13->512,13,13->1024,13,13->512,13,13
        out0_branch = self.last_layer0[:5](x0)
        # 512,13,13->1024,13,13->75,13,13
        out0 = self.last_layer0[5:](out0_branch)
        
        # 512,13,13 -> 256,13,13
        x1_in = self.last_layer1_conv(out0_branch)
        # 256,13,13 -> 256,26,26
        x1_in = self.last_layer1_upsample(x1_in)

        # 256,26,26->512+256,26,26
        x1_in = torch.cat([x1_in,x1], 1)

        """
        第二个特征层
        out1 = (batch_size,75,26,26)
        """
        # 768,26,26->256,26,26->512,26,26->256,26,26->512,26,26->256,26,26   
        out1_branch = self.last_layer1[:5](x1_in)
        # 256,26,26->512,26,26->75,26,26
        out1        = self.last_layer1[5:](out1_branch)

         # 256,26,26 -> 128,26,26
        x2_in = self.last_layer2_conv(out1_branch)
        # 128,26,26 -> 128,52,52
        x2_in = self.last_layer2_upsample(x2_in)

       # 128,52,52 -> 128+256,52,52
        x2_in = torch.cat([x2_in, x2], 1)
        """
        第一个特征层
        out3 = (batch_size,75,52,52)
        """
        # 384,52,52->128,52,52->256,52,52->128,52,52->256,52,52->128,52,52
        # 128,52,52->256,52,52->75,52,52
        out2 = self.last_layer2(x2_in)
        
        return out0, out1, out2

        



def main():
    t = torch.randn([4,3,416,416])
    model = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 20)
    out0, out1, out2 = model(t)
    print("out0 shape: ", out0.shape)
    print("out1 shape: ", out1.shape)
    print("out2 shape: ", out2.shape)

if __name__ == "__main__":
    main()



