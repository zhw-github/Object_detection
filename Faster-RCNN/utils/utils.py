import numpy as np
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    w, h        = size
    new_image   = image.resize((w, h), Image.BICUBIC)
    return new_image

"""
    根据classes.txt获得种类名称和种类的数量
    f.readlines:
    ['aeroplane\n', 'bicycle\n', 'bird\n', 'boat\n', 'bottle\n', 'bus\n', 'car\n', 'cat\n', 'chair\n', 'cow\n', 'diningtable\n', 'dog\n', 'horse\n', 
    'motorbike\n', 'person\n', 'pottedplant\n', 'sheep\n', 'sofa\n', 'train\n', 'tvmonitor']
    c.strip()删除字符串的头尾信息

"""
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)





#---------------------------------------------------#
#   计算resize后的图片的大小，resize后的图片短边为600
#---------------------------------------------------#

def get_new_img_size(height,width,img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f*height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f*width)
        resized_height = int(img_min_side)
    
    return resized_height,resized_width

def preprocess_input(image):
    image /= 255.0
    return image


def main():
    resized_height,resized_width = get_new_img_size(1000, 800)
    print(resized_height,resized_width)


if __name__ == "__main__":
    main()