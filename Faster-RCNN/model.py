import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import cvtColor,resize_image,get_classes,get_new_img_size,preprocess_input
from utils.utils_bbox import DecodeBox


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES
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
        # 查看是否有其他的输入变量,items将初始化转成元组合,可遍历的形式
        for name,value in kwargs.items():
            # setattr设置属性值(self.name=value)
            setattr(self, name, value)
        # 获得种类名称和种类数量
        self.class_names,self.num_classes = get_classes(self.classes_path)

        self.std = torch.Tensor([0.1,0.1,0.2,0.2]).repeat(self.num_classes+1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        """
        DecodeBox
        """
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    # 载入模型
    def generate(self):
        # 载入模型和权值
        self.net = FasterRCNN(self.num_classes,"predict",anchor_scales=self.anchors_size,
                        backbone=self.backbone)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model,anchors,and classes loaded'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()
        

    # 检测图片
    def detect_image(self, image):
        # 计算输入图片宽高 [1330,1330]
        image_shape = np.array(np.shape(image)[0:2])
        # 计算resize后的图片大小,resize后图片的短边为600 [600,600]
        input_shape = get_new_img_size(image_shape[0], image_shape[1])  
        # 转换成RGB图像
        image = cvtColor(image)
        # 给原图像进行resize,resize到短边为600的大小上
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        # 添加上batch_size的维度 [1,3,600,600]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),
                                (2,0,1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            """
                n_test_post_nms = num_rois = 100
                roi_cls_locs 建议框调整参数 [1,100,21*4]
                roi_scores 建议框种类的得分 [1, 100, 21]
                rois 建议框坐标            [100, 4]
            """
            roi_cls_locs,roi_scores,rois,_ = self.net(images)
            print("roi_cls_locs: ",roi_cls_locs.shape)
            print("roi_scores: ",roi_scores.shape)
            print("rois: ",rois.shape)
            # 利用classifier的预测结果对建议框进行解码获得预测框
            #-------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            # 如果没有检测出物体返回原图
            if len(results[0]) <= 0:
                return image
            

            top_label = np.array(results[0][:, 5],dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            # 设置字体与边框厚度           
            font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

            # 图像绘制
            for i,c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box = top_boxes[i]
                score = top_conf[i]

                top,left,bottom,right = box

                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                # print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

            return image
                

def main():
    faster_rcnn = FRCNN()
    img = "./street.jpg"
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        faster_rcnn.detect_image(image)
        image.save("./result.jpg")
        image.show()
        
        


    



if __name__ == "__main__":
    main()