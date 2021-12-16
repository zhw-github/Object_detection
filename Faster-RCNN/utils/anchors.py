import numpy as np
import matplotlib.pyplot as plt
from time import *

def generate_anchor_base(base_size=16,ratios=[0.5,1,2],anchor_scales=[8,16,32]):
    anchor_base = np.zeros((len(ratios)*len(anchor_scales), 4),dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = -h / 2.
            anchor_base[index, 1] = -w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shrift_x = np.arange(0, width*feat_stride, feat_stride)
    
    shrift_y = np.arange(0, height * feat_stride, feat_stride)
    shrift_x,shrift_y = np.meshgrid(shrift_x,shrift_y)
    shrift = np.stack((shrift_x.ravel(),shrift_y.ravel(),shrift_x.ravel(),shrift_y.ravel()),axis=1)
    A = anchor_base.shape[0]
    K = shrift.shape[0]
   
    anchor = anchor_base.reshape(1,A,4) + shrift.reshape(K,1,4)
    anchor = anchor.reshape((K*A,4)).astype(np.float32)

    return anchor


def main():
    begin = time()
    anchor_base = generate_anchor_base()
    width, height, feat_stride = 38,38,16
    anchors_all = _enumerate_shifted_anchor(anchor_base, 16, 38, 38)
    print(np.array(anchors_all).shape)
    end = time()
    run_time = end - begin
    print(run_time)


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.ylim(-300,700)
    # plt.xlim(-300,700)
    # shift_x = np.arange(0, width * feat_stride , feat_stride)
    # shift_y = np.arange(0, height * feat_stride , feat_stride)
    # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # plt.scatter(shift_x, shift_y)
    # box_widths  = anchors_all[:,2]-anchors_all[:,0]
    # box_heights = anchors_all[:,3]-anchors_all[:,1]
    # # for i in range(anchors_all.shape[0]):
    # #     rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
    # #     ax.add_patch(rect)
    # for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
    #     rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
    #     ax.add_patch(rect)
    # plt.show()

if __name__ == "__main__":
    main()