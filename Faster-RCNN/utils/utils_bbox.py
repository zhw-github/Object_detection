import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms


def loc2bbox(src_bbox, loc):
    # 判断src_bbox是否为空 是否是没有必要的呢
    if src_bbox.size()[0] == 0:
        return torch.zeros((0,4), dtype=loc.dtype)
    
    """
    获取anchorw,h,x,y
    w = 右上角点的横坐标-左下角点的纵坐标
    h = 右上角点的纵坐标-左下角点的纵坐标
    x = 左下角的横坐标加上0.5*w
    y = 左下角的纵坐标加上0.5*h
    注意: torch.unsqueeze:扩展维度，[12996] -> [12996,1]
    """
    src_width = torch.unsqueeze(src_bbox[:,2] - src_bbox[:,0], -1)
    src_height = torch.unsqueeze(src_bbox[:,3] - src_bbox[:,1], -1)
    src_ctr_x = torch.unsqueeze(src_bbox[:,0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:,1], -1) + 0.5 * src_height
    """
    从loc (12996,4)中取出dx,dy,dh,dw
    注意如果直接使用loc[:,0]会改变维度(12996)
    使用loc[:,0::4]在列的维度上每隔四个去取，由于只有四个所以既不破坏取值也不破坏维度
    """
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]
    """
    使用论文中的公式由先验框(候选区域)生成建议框(调整后的anchors)，也就是由anchors -> proposal
    """
    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    """
    生成的建议框要与anchors一致用中心点的左下角点和右上角点表示一个矩形框
    """
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

    
class DecodeBox():
    def __init__(self, std, num_classes):
        self.std            = std
        self.num_classes    = num_classes + 1    

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # [600,600]
        input_shape = np.array(input_shape)
        # [1300,1300]
        image_shape = np.array(image_shape)
        
        # [所以类别一共剩下的框，2]
        box_mins    = box_yx - (box_hw / 2.)
        # [所以类别一共剩下的框，2]
        box_maxes   = box_yx + (box_hw / 2.)
        # [所以类别一共剩下的框，4]
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        # 由于前面做了归一化，所以这里面要成回来
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []
        # n_test_post_nms = num_rois = 100
        # [1,100,21*4] batch_size
        bs = len(roi_cls_locs)
        # batch_size, num_rois, 4
        rois = rois.view( (bs, -1, 4 ))
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以
        # for i in range(len(mbox_loc))只进行一次
        for i in range(bs):
            # 对回归参数进行reshape,起到改变数量级的作用
            # [100,21*4]
            roi_cls_loc = roi_cls_locs[i] * self.std
            # [100,21,4]
            roi_cls_loc = roi_cls_loc.view([-1,self.num_classes,4])
            # 利用classifier网络的预测结果对建议框进行调整获得预测框
            # [100,4] -> [100,1,4] -> [100,21,4]
            roi = rois[i].view((-1,1,4)).expand_as(roi_cls_loc)
            # [100*21,4]
            cls_bbox = loc2bbox(roi.contiguous().view((-1,4)), roi_cls_loc.contiguous().view((-1,4)))
            # [100,21,4]
            cls_bbox = cls_bbox.view(([-1, self.num_classes, 4]))
            """
            对预测框进行归一化，调整到0-1之间
            """
            cls_bbox[...,[0,2]] = (cls_bbox[...,[0,2]]) / input_shape[1]
            cls_bbox[...,[1,3]] = (cls_bbox[...,[1,3]]) / input_shape[0]

            roi_scores = roi_scores[i]
            prob = F.softmax(roi_scores, dim=1)

            results.append([])
            for c in range(1, self.num_classes):
                # 取出属于该类的所有框的置信度
                # 判断是否大于设定阈值
                # [100]
                c_confs = prob[:, c]
                # [100] True / False
                c_confs_m = c_confs > confidence
                

                if len(c_confs[c_confs_m]) > 0:
                    # 取出的分高于confidence的框
                    # [x,4]
                    boxes_to_process = cls_bbox[c_confs_m,c]
                    # [x]
                    confs_to_process = c_confs[c_confs_m]
                    
                    # [y]
                    keep = nms(boxes_to_process,confs_to_process,nms_iou)
                    # 取出极大值抑制之后剩余的框
                    good_boxes = boxes_to_process[keep]
                    # [:,None] : [y] -> [y,1]
                    confs = confs_to_process[keep][:,None]
                    # [y,1]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    # 将label、置信度、框的位置进行堆叠。
                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result[]里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy = (results[-1][:,0:2] + results[-1][:,2:4]) / 2
                box_wh = (results[-1][:,2:4] - results[-1][:,0:2])
                results[-1][:,:4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
            
        return results
                

                    
            



        