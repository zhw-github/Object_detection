import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                                    kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                                    kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsampling = downsample

    def forward(self, x):
        identity = x
        
        if self.downsampling is not None:
            identity = self.downsampling(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                                kernel_size=1, stride=1, bias=False) # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                        kernel_size=3, stride=stride, padding=1, bias=False) # change HW
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion 
                    , kernel_size=1, stride=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
       
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, 
                    block,
                    blocks_num,
                    num_classes=1000,
                    include_top=True):
        super(ResNet, self).__init__()
        """
            假设Input_size: [B,3,600,600]
        """
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True) # [B,3,600,600] -> [B,64, 300, 300]
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)#[B,64,300,300]->[B,64,150,150]

        # [B,64,150,150] -> [B,256,150,150]
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # [B,256,150,150] -> [B,512,75,75]
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # [B,512,75,75] -> [B,1024,38,38]
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # []B,1024,38,38]-> [B,2048,14,14]
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion , kernel_size=1,
                                    stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel,channel,stride=stride,downsample=downsample))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print("layer1: ", x.shape)
        x = self.layer2(x)
        # print("layer2: ", x.shape)
        x = self.layer3(x)
        # print("layer3: ", x.shape)
        x = self.layer4(x)
        # print("layer4: ", x.shape)

        if self.include_top:
            x = self.avgpool(x)
            print(x.shape)
            x = x.flatten(1)
            x = self.fc(x)
        return x



def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    
    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier 


    


if __name__ == "__main__":
    t = torch.randn([4,3,600,600])
    features, _ = resnet50()
    out = features[0:7](t)
    print(out.shape)
    # print(features[0:3])