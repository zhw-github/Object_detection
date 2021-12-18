import torch
import torch.nn as nn
import sys 
sys.path.append("..") 
from nets.classifier import Resnet50RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork

class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                model="training",
                feat_stride=16,
                anchor_scales=[8,16,32],
                ratios=[0.5,1,2],
                backbone='resnet',
                pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        # 一共可以存在多个主干
        # 在这里我们只实现Resnet
        if backbone == 'resnet50':
            self.extractor,classifier = resnet50(pretrained)
            # 构建rpn网络
            self.rpn = RegionProposalNetwork(
                1024,512,
                ratios=ratios,
                anchor_scales = anchor_scales,
                feat_stride=self.feat_stride,
                model = model
            )
            # 构建head部分
            self.head = Resnet50RoIHead(
                n_class = num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier)

    def forward(self, x, scale=1.):
        # 计算输入图片大小
        img_size = x.shape[2:]
        # 利用主干网络提取特征
        base_feature = self.extractor.forward(x)
        # 获得建议框
        _,_,rois,roi_indices,_ = self.rpn.forward(base_feature, img_size,scale)
        # 获得head的分类和回归结果
        roi_cls_locs,roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs,roi_scores,rois,roi_indices
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()




def main():
    torch.manual_seed(111) 
    t = torch.randn([1,3,600,600]).cuda()
    model = FasterRCNN(20,model="predict",backbone="resnet50").cuda()
    roi_cls_locs,roi_scores,rois,roi_indices = model(t)
    print(roi_scores.shape)



if __name__ == "__main__":
    main()