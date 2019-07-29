import torch
from torch import nn
from layers import *

config = {}
config['anchors'] = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5  # 3 #6. #mm
config['sizelim2'] = 10  # 30
config['sizelim3'] = 20  # 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}

config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']


# config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','d92998a73d4654a442e6d6ba15bbb827','990fbe3f0a1b53878669967b9afd1441','820245d8b211808bd18e78ff5be16fdb','adc3bbc63d40f8761c59be10f1e504c3',
#                       '417','077','188','876','057','087','130','468']


class USNETres(nn.Module):
    def __init__(self, featureget=False):
        super(USNETres, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.featureget = featureget
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]

        for i in range(len(num_blocks_forw)):  ### 4
            blocks = []
            for j in range(num_blocks_forw[i]):  ##{2,2,3,3}
                if j == 0:  # conv
                    ###plus source connection
                    blocks.append(PostRes(self.featureNum_forw[i] + 1, self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        # Just for transfer
        num_blocks_back = [3, 3]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))
        ###########################

        self.avgpool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)

        self.postBlock = []
        self.classifiy = []

        # Just for transfer
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))


    #        self.path5 = nn.Sequential(
    #            nn.ConvTranspose3d(24, 2, kernel_size = 2, stride = 2),
    #            nn.BatchNorm3d(2),
    #            nn.ReLU(inplace = True))
    # self.drop = nn.Dropout3d(p = 0.5, inplace = False)
    # self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
    #                             nn.ReLU(),
    #                             #nn.Dropout3d(p = 0.3),
    #                            nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x):
        out = self.preBlock(x)  # 16     ### conv0: 1+1: conv3+conv3= conv5x5 [conv-bn-relu conv-bn-relu]
        out_pool, indices0 = self.maxpool1(out)

        source0 = self.avgpool(x)
        out_pool_s = torch.cat((out_pool, source0), 1)

        out1 = self.forw1(out_pool_s)  # 32+1             #### conv1: ([conv3+conv3] + x )*2
        out1_pool, indices1 = self.maxpool2(out1)

        source1 = self.avgpool(source0)
        out1_pool_s = torch.cat((out1_pool, source1), 1)

        out2 = self.forw2(out1_pool_s)  # 64 +1             #### conv2: ([conv3+conv3] + x )*2
        # out2 = self.drop(out2)
        out2_pool, indices2 = self.maxpool3(out2)

        source2 = self.avgpool(source1)
        out2_pool_s = torch.cat((out2_pool, source2), 1)

        out3 = self.forw3(out2_pool_s)  # 96  +1            #### conv3: ([conv3+conv3] + x )*3
        out3_pool, indices3 = self.maxpool4(out3)

        source3 = self.avgpool(source2)
        out3_pool_s = torch.cat((out3_pool, source3), 1)

        out4 = self.forw4(out3_pool_s)  # 96     +1         #### conv4: (([conv3+conv3] + x )*3    2*64*8*8
        # now: 64*1*16*16
        # out4 = self.drop(out4)

        # add4=out4+out3_pool
        # rev3 = self.path1(add4)

        # # comb3 = self.back3(torch.cat((rev3, out3), 1))#12+12
        # #comb3 = self.drop(comb3)
        # add3=rev3+out2_pool
        # rev2 = self.path2(add3)

        # # comb2 = self.back2(torch.cat((rev2, out2), 1))#24+24
        # add2=rev2+out1_pool
        # rev1=self.path3(add2)
        # # comb1 = self.back1(torch.cat((rev1, out1), 1))#48+48
        # #add1=rev1+out_pool
        # #rev=self.path4(add1)

        # # comb0 = self.back0(torch.cat((rev0,out),1))#96+96

        rev = self.postBlock(out4)
        features = torch.squeeze(rev)
        features = features.view(-1, 32)
        out = self.classifiy(features)

        # comb2 = self.drop(comb2)
        # out = self.output(comb2)
        # size = out.size()
        # out = out.view(out.size(0), out.size(1), -1)
        # #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        # out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        # out = out.view(-1, 5)
        if self.featureget is False:
            return out
        return out, features


class GroupFeatureNet(nn.Module):

    def __init__(self, featureget=False):
        super(GroupFeatureNet, self).__init__()
        self.linear1 = nn.Linear(39, 2)
        self.classify = nn.Softmax()
        # self.drop = nn.Dropout(p=0.5)
        self.featureget = featureget

    def forward(self, x):
        features = self.linear1(x)
        out = self.classify(features)
        if self.featureget is False:
            return x
        return out, features


class ClassifyNet(nn.Module):

    def __init__(self):
        """
        The final network used as the classifier for the four connected features
        """
        super(ClassifyNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(8, 2),
            nn.Softmax()
        )

    def forward(self, x1, x2, x3, gx):
        x = torch.cat((x1, x2, x3, gx), 1)
        x = self.classifier(x)
        return x


class MVIBigNet(nn.Module):

    def __init__(self):
        super(MVIBigNet, self).__init__()
        self.feature_a = get_USNE(featureget=True)
        self.feature_d = get_USNE(featureget=True)
        self.feature_p = get_USNE(featureget=True)
        self.feature_g = GroupFeatureNet(featureget=True)
        self.classifier = ClassifyNet()

    def forward(self, x1, x2, x3, gx):
        _, x1 = self.feature_a(x1)
        _, x2 = self.feature_a(x2)
        _, x3 = self.feature_a(x3)
        _, gx = self.feature_g(gx)
        x = self.classifier(x1, x2, x3, gx)
        return x


def get_USNE(featureget=False):
    net = USNETres(featureget=featureget)

    checkpoint = torch.load('weihua136.ckpt')
    net.load_state_dict(checkpoint['state_dict'])

    #######################
    for param in net.parameters():
        param.requires_grad = False

    #########################################
    # 64 4 8 8
    net.postBlock = nn.Sequential(
        nn.Conv3d(64, 32, kernel_size=3, stride=2, padding=1),  # 32 2 4 4
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),
        nn.Conv3d(32, 8, kernel_size=3, stride=2, padding=1),  # 8 1 2 2
        nn.BatchNorm3d(8),
        nn.ReLU(inplace=True),

        # nn.AvgPool3d(kernel_size=(1, 2, 2), stride=1)         # Global Average Pooling ==> -1 8 1 2 2 ==> -1 8
    )
    net.classifiy = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(32, 2),
        nn.Softmax()
    )
    return net


def test():
    debug = True
    # net = USNETres(True)
    # x = (torch.randn(1,1,96,96,96))
    # crd = (torch.randn(1,3,24,24,24))s
    # y = net(x, crd)
    # print(y.size())
    # print(y)

    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = USNETres(True)
    checkpoint = torch.load('136.ckpt')
    sd = checkpoint['state_dict']
    # pdb.set_trace()

    s = net.load_state_dict(checkpoint['state_dict'])

    #########################################
    # 64 4 8 8
    net.postBlock = nn.Sequential(
        nn.Conv3d(64, 32, kernel_size=3, stride=2, padding=1),  # 32 2 4 4
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),
        nn.Conv3d(32, 8, kernel_size=3, stride=2, padding=1),  # 8 1 2 2
        nn.BatchNorm3d(8),
        nn.ReLU(inplace=True),

        # nn.AvgPool3d(kernel_size=(1, 2, 2), stride=1)         # Global Average Pooling ==> -1 8 1 2 2 ==> -1 8
    )
    net.classifiy = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(32, 2),
        nn.Softmax()
    )

    net = net.to(device)

    summary(net, (1, 4, 128, 128))

    # net2 = MVIBigNet().to(device)
    # summary(net2, [(1, 16, 256, 256), (1, 16, 256, 256), (1, 16, 256, 256), (39,)])


if __name__ == '__main__':
    test()
