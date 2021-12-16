import sys 
sys.path.append("..") 
import torch
from torch import nn
from torch.nn import functional as F
from utils.anchors import _enumerate_shifted_anchor,generate_anchor_base
import numpy as np
from utils.utils_bbox import loc2bbox
from torchvision.ops import nms



"""
注意: 在生成建议框时，ProposalCreator处理单张图片，需要将B batch_size中的每一个B单独输入
"""
class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,
        min_size            = 16
    
    ):
        #   设置预测还是训练
        self.mode               = mode
        #   建议框非极大抑制的iou大小
        self.nms_iou            = nms_iou
        #   训练用到的建议框数量
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms
        #   预测用到的建议框数量
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms

        #   将先验框转换成tensor
        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()
        #   将RPN网络预测结果转化成建议框
        roi = loc2bbox(anchor, loc)
        """
        防止建议框超出图片边缘使用torch.clamp函数
        a = torch.randn(4)   tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        torch.clamp(a, min=-0.5, max=0.5) tensor([-0.5000,  0.1734, -0.0478, -0.0922])
        """
        roi[:,[0,2]] = torch.clamp(roi[:,[0,2]], min=0, max=img_size[1])
        roi[:,[1,3]] = torch.clamp(roi[:,[1,3]], min=0, max=img_size[0])
        # 建议框的宽高最小值不能小于16
        min_size = self.min_size * scale
        # torch.where 返回的是tuple类型，是框的索引,所以要取出第一个[0]
        keep = torch.where(   ((roi[:, 2] - roi[:, 0]) >= min_size) 
                            & ((roi[:, 3] - roi[:, 1]) >= min_size)  )[0]
        # 将对应的建议框保留下来
        roi = roi[keep, :]
        score = score[keep]
        """
        根据得分进行排序，取出建议框
        argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
        descending=True 加入后是从大到小
        """
        order = torch.argsort(score, descending=True)
        # 设定建议框多少需要再筛一下
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        """
        然后对建议框进行非极大值抑制(NMS)
        这里我们使用官方的会大大提高效率
        nms(boxes, scores, iou_threshold) 注意一定要从大到小排序好的
        boxes : Tensor[N, 4])
            boxes to perform NMS on. They
            are expected to be in (x1, y1, x2, y2) format
        scores : Tensor[N]
            scores for each one of the boxes
        iou_threshold : float
            discards all overlapping
            boxes with IoU > iou_threshold
        """
        keep = nms(roi, score, self.nms_iou)
        # 极大值抑制之后进一步去除一些框
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi
        
        

        






       
        



class RegionProposalNetwork(nn.Module):
    def __init__(self,
                    in_channels=512,
                    mid_channels=512,
                    ratios=[0.5,1,2],
                    anchor_scales=[8,16,32],
                    feat_stride=16,
                    model="training"):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchors = self.anchor_base.shape[0]

        # 先进行一个3*3的卷积(也就是滑动窗口)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 分类预测每一个先验框是否包含物体
        self.score = nn.Conv2d(mid_channels, n_anchors*2, 1, 1, 0)
        # 回归预测对先验框进行调整
        self.loc = nn.Conv2d(mid_channels, n_anchors*4, 1, 1, 0)

        # 网格中心点间的间距 
        self.feat_stride = feat_stride

        # 用于对建议区解码并进行非极大值抑制
        self.proposal_layer = ProposalCreator(model)




        # 对FPN的网络部分进行权值初始化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
    
    def forward(self, x,img_size, scale=1.):
        B, _, h, w = x.shape
        # 对特征层进行3*3卷积，也就是进行滑动窗口操作
        out = F.relu(self.conv1(x))
        # 分类预测
        # [B, 512, 38, 38] -> [B, 18, 38, 38]
        rpn_scores = self.score(out)
        # [B, 18, 38, 38] -> [B, 38, 38, 18] -> [B, 38*38*9, 2]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)

        # 预测回归 [B, 512, 38, 38] -> [B, 36, 38, 38]
        rpn_locs = self.loc(out)
        # [B, 36, 38, 38] -> [B, 38, 38, 36] -> [B, 38*38*9, 4]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        # 进行softmax运算
        rpn_softmax = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax[:, :, 1].contiguous() #获得前景的得分(包含物体的得分)
        rpn_fg_scores = rpn_fg_scores.view(B, -1)  # [B, 38*38*9*1]

        # 生成先验框,此时anchor是布满网格点的，当输入图片是600*600*3时，anchor: [12996,4] 38*38*9=12996
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        

        rois = list()
        roi_indices = list()
        for i in range(B):
            roi = self.proposal_layer(rpn_locs[i],rpn_fg_scores[i],anchor,img_size,scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = torch.cat(rois, dim=0) # [600*B, 4]
        roi_indices = torch.cat(roi_indices, dim=0) # [600*B]
         

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


# 对FPN网络进行权值初始化
def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()




def main():
    torch.manual_seed(111) 
    t = torch.randn([4,1024,38,38])
    model = RegionProposalNetwork(1024)
    out,_,_,_,_ = model(t,(600,600))
    print(out.shape)


if __name__ == "__main__":
    main()