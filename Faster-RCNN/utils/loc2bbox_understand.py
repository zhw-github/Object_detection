import torch


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




    


    
    




def main():
    pass

if __name__ == "__main__":
    main()