import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import cvtColor,resize_image,get_classes


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES、
#   model_path和classes_path参数的修改
#--------------------------------------------#
class FRCNN(object):
    _defaults={
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"    : 'model_data/voc_weights_resnet.pth',
        "classes_path"  : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   网络的主干特征提取网络，resnet50或者vgg
        #---------------------------------------------------------------------#
        "backbone"      : "resnet50",
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"    : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"       : 0.3,
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size'  : [8, 16, 32],
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"          : True,

    }
    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name" + n + ""
    

    # 初始化fasterRCNN
    def __init__(self, **kwargs):
        # self.a = a的简便写法
        self.__dict__.update(self._defaults)
        # 查看是否有其他的输入变量,items将初始化转成字典
        print(kwargs.items())
        for name,value in kwargs.items():
            # setattr设置属性值(self.name=value)
            setattr(self, name, value)
        # 获得种类名称和种类数量
        self.class_names,self.num_classes = get_classes(self.classes_path)
        print(self.num_classes)

        self.std = torch.Tensor([0.1,0.1,0.2,0.2]).repeat(self.num_classes+1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        """
        DecodeBox
        """
        










def main():
    FRCNN()



if __name__ == "__main__":
    main()