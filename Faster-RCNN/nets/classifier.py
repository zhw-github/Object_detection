import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool
from rpn import RegionProposalNetwork
from nets.resnet50 import resnet50


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        # 对ROIPooling后的结果进行回归预测 
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # 对ROIPooling后的结果进行回归预测
        self.score = nn.Linear(2048, n_class)
        # 权值初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        B, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        print('Base_layers: ', x.size())
        print('roi_indices: ', roi_indices.size())
        print('rois: ', rois.size())
        # 将ROI映射到特征图中，才能进行RoIPooling
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]
        # 因为在训练时图片带有批次，所以需要将索引标号与roi相对应
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # 利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois)
       
        # 利用classifier网络进行特征提取
        fc7 = self.classifier(pool)
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        fc7 = fc7.view(fc7.size(0), -1)
        # 做最终预测的两个全连接
        roi_cls_locs = self.cls_loc(fc7) #[300, 21*4]
        roi_scores   = self.score(fc7)   #[300, 21]
        roi_cls_locs = roi_cls_locs.view(B, -1, roi_cls_locs.size(1)) # [1,300,21*4]
        roi_scores = roi_scores.view(B, -1, roi_scores.size(1))     # [1,300,21]
        return roi_cls_locs,roi_scores
       



def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()    




def main():
    torch.manual_seed(111) 
    t = torch.randn([1,1024,38,38]).cuda()
    model = RegionProposalNetwork(1024).cuda()
    rpn_locs, rpn_scores, rois, roi_indices, anchor = model(t,(600,600))
    base_features = torch.randn([1,1024,38,38]).cuda()
    _, classifier = resnet50()
    
    head = Resnet50RoIHead(21, 14, 1, classifier.cuda())
    head(base_features,rois,roi_indices,(600,600))
    

    



if __name__ == "__main__":
    main()