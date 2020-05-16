"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from network import Resnet
from network.PosEmbedding import PosEmbedding2D
from network.HANet import HANet_Conv
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights, RandomPosVal_Masking, RandomVal_Masking, Zero_Masking, RandomPosZero_Masking


import torchvision.models as models


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3PlusHANet(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV3PlusHANet, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.num_attention_layer = 0
        self.trunk = trunk
        
        for i in range(5):
            if args.hanet[i] > 0:
                self.num_attention_layer += 1

        print("#### HANet layers", self.num_attention_layer)
        

        if trunk == 'shufflenetv2':
            channel_1st = 3
            channel_2nd = 24
            channel_3rd = 116
            channel_4th = 232
            prev_final_channel = 464
            final_channel = 1024
            resnet = models.shufflenet_v2_x1_0(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.maxpool)
            self.layer1 = resnet.stage2
            self.layer2 = resnet.stage3
            self.layer3 = resnet.stage4
            self.layer4 = resnet.conv5

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        elif trunk == 'mnasnet_05' or trunk == 'mnasnet_10':

            if trunk == 'mnasnet_05':
                resnet = models.mnasnet0_5(pretrained=True)
                channel_1st = 3
                channel_2nd = 16
                channel_3rd = 24
                channel_4th = 48
                prev_final_channel = 160
                final_channel = 1280

                print("# of layers", len(resnet.layers))
                self.layer0 = nn.Sequential(resnet.layers[0],resnet.layers[1],resnet.layers[2],
                                            resnet.layers[3],resnet.layers[4],resnet.layers[5],resnet.layers[6],resnet.layers[7])   # 16
                self.layer1 = nn.Sequential(resnet.layers[8], resnet.layers[9]) # 24, 40
                self.layer2 = nn.Sequential(resnet.layers[10], resnet.layers[11])   # 48, 96
                self.layer3 = nn.Sequential(resnet.layers[12], resnet.layers[13]) # 160, 320
                self.layer4 = nn.Sequential(resnet.layers[14], resnet.layers[15], resnet.layers[16])  # 1280
            else:
                resnet = models.mnasnet1_0(pretrained=True)
                channel_1st = 3
                channel_2nd = 16
                channel_3rd = 40
                channel_4th = 96
                prev_final_channel = 320
                final_channel = 1280

                print("# of layers", len(resnet.layers))
                self.layer0 = nn.Sequential(resnet.layers[0],resnet.layers[1],resnet.layers[2],
                                            resnet.layers[3],resnet.layers[4],resnet.layers[5],resnet.layers[6],resnet.layers[7])   # 16
                self.layer1 = nn.Sequential(resnet.layers[8], resnet.layers[9]) # 24, 40
                self.layer2 = nn.Sequential(resnet.layers[10], resnet.layers[11])   # 48, 96
                self.layer3 = nn.Sequential(resnet.layers[12], resnet.layers[13]) # 160, 320
                self.layer4 = nn.Sequential(resnet.layers[14], resnet.layers[15], resnet.layers[16])  # 1280

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        elif trunk == 'mobilenetv2':
            channel_1st = 3
            channel_2nd = 16
            channel_3rd = 32
            channel_4th = 64

            # prev_final_channel = 160
            prev_final_channel = 320

            final_channel = 1280
            resnet = models.mobilenet_v2(pretrained=True)
            self.layer0 = nn.Sequential(resnet.features[0],
                                        resnet.features[1])
            self.layer1 = nn.Sequential(resnet.features[2], resnet.features[3],
                                        resnet.features[4], resnet.features[5], resnet.features[6])
            self.layer2 = nn.Sequential(resnet.features[7], resnet.features[8], resnet.features[9], resnet.features[10])

            # self.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13], resnet.features[14], resnet.features[15], resnet.features[16])
            # self.layer4 = nn.Sequential(resnet.features[17], resnet.features[18])

            self.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13],
                                        resnet.features[14], resnet.features[15], resnet.features[16],
                                        resnet.features[17])
            self.layer4 = nn.Sequential(resnet.features[18])

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        else:
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 256
            channel_4th = 512
            prev_final_channel = 1024
            final_channel = 2048

            if trunk == 'resnet-50':
                resnet = Resnet.resnet50()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-101': # three 3 X 3
                resnet = Resnet.resnet101()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                            resnet.conv2, resnet.bn2, resnet.relu2,
                                            resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            elif trunk == 'resnet-152':
                resnet = Resnet.resnet152()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnext-50':
                resnet = models.resnext50_32x4d(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnext-101':
                resnet = models.resnext101_32x8d(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'wide_resnet-50':
                resnet = models.wide_resnet50_2(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'wide_resnet-101':
                resnet = models.wide_resnet101_2(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            else:
                raise ValueError("Not a valid network arch")

            self.layer0 = resnet.layer0
            self.layer1, self.layer2, self.layer3, self.layer4 = \
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            if self.variant == 'D':
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D4':
                for n, m in self.layer2.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        if self.args.aux_loss is True:
            self.dsn = nn.Sequential(
                nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
                Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            initialize_weights(self.dsn)

        if self.args.hanet[0] == 1:
            self.hanet0 = HANet_Conv(prev_final_channel, final_channel,
                            self.args.hanet_set[0], self.args.hanet_set[1], self.args.hanet_set[2],
                            self.args.hanet_pos[0], self.args.hanet_pos[1],
                            pos_rfactor=self.args.pos_rfactor, pooling=self.args.pooling,
                            dropout_prob=self.args.dropout, pos_noise=self.args.pos_noise)
            initialize_weights(self.hanet0)

        if self.args.hanet[1] == 1:
            self.hanet1 = HANet_Conv(final_channel, 1280,
                            self.args.hanet_set[0], self.args.hanet_set[1], self.args.hanet_set[2], 
                            self.args.hanet_pos[0], self.args.hanet_pos[1],
                            pos_rfactor=self.args.pos_rfactor, pooling=self.args.pooling,
                            dropout_prob=self.args.dropout, pos_noise=self.args.pos_noise)
            initialize_weights(self.hanet1)
            
        if self.args.hanet[2] == 1:
            self.hanet2 = HANet_Conv(1280, 256,
                            self.args.hanet_set[0], self.args.hanet_set[1], self.args.hanet_set[2],
                            self.args.hanet_pos[0], self.args.hanet_pos[1],
                            pos_rfactor=self.args.pos_rfactor, pooling=self.args.pooling,
                            dropout_prob=self.args.dropout, pos_noise=self.args.pos_noise)
            initialize_weights(self.hanet2)

        if self.args.hanet[3] == 1:
            self.hanet3 = HANet_Conv(304, 256,
                            self.args.hanet_set[0], self.args.hanet_set[1], self.args.hanet_set[2],
                            self.args.hanet_pos[0], self.args.hanet_pos[1],
                            pos_rfactor=self.args.pos_rfactor, pooling=self.args.pooling,
                            dropout_prob=self.args.dropout, pos_noise=self.args.pos_noise)
            initialize_weights(self.hanet3)

        if self.args.hanet[4] == 1:
            self.hanet4 = HANet_Conv(256, num_classes,
                            self.args.hanet_set[0], self.args.hanet_set[1], self.args.hanet_set[2],
                            self.args.hanet_pos[0], self.args.hanet_pos[1],
                            pos_rfactor=self.args.pos_rfactor, pooling='max',
                            dropout_prob=self.args.dropout, pos_noise=self.args.pos_noise)
            initialize_weights(self.hanet4)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

    def forward(self, x, gts=None, aux_gts=None, pos=None, attention_map=False, attention_loss=False):

        x_size = x.size()  # 800

        x = self.layer0(x)  # 400
        x = self.layer1(x)  # 400
        low_level = x
        x = self.layer2(x)  # 100

        x = self.layer3(x)  # 100

        aux_out = x
        x = self.layer4(x)  # 100

        if self.num_attention_layer > 0:
            if attention_map:
                attention_maps = [torch.Tensor() for i in range(self.num_attention_layer)]
                pos_maps = [torch.Tensor() for i in range(self.num_attention_layer)]
                map_index = 0

        if self.args.hanet[0]==1:
            if attention_map:
                x, attention_maps[map_index], pos_maps[map_index] = self.hanet0(aux_out, x, pos, return_attention=True, return_posmap=True)
                map_index += 1
            else:
                x = self.hanet0(aux_out, x, pos)

        represent = x

        x = self.aspp(x)

        if self.args.hanet[1]==1:
            if attention_map:
                x, attention_maps[map_index], pos_maps[map_index] = self.hanet1(represent, x, pos, return_attention=True, return_posmap=True)
                map_index += 1
            else:
                x = self.hanet1(represent, x, pos)
            
        dec0_up = self.bot_aspp(x)

        if self.args.hanet[2]==1:
            if attention_map:
                dec0_up, attention_maps[map_index], pos_maps[map_index] = self.hanet2(x, dec0_up, pos, return_attention=True, return_posmap=True)
                map_index += 1
            else:
                dec0_up = self.hanet2(x, dec0_up, pos)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)

        if self.args.hanet[3]==1:
            if attention_map:
                dec1, attention_maps[map_index], pos_maps[map_index] = self.hanet3(dec0, dec1, pos, return_attention=True, return_posmap=True)
                map_index += 1
            else:
                dec1 = self.hanet3(dec0, dec1, pos)

        dec2 = self.final2(dec1)

        if self.args.hanet[4]==1:
            if attention_map:
                dec2, attention_maps[map_index], pos_maps[map_index] = self.hanet4(dec1, dec2, pos, return_attention=True, return_posmap=True)
                map_index += 1
            elif attention_loss:
                dec2, last_attention = self.hanet4(dec1, dec2, pos, return_attention=False, return_posmap=False, attention_loss=True)
            else:
                dec2 = self.hanet4(dec1, dec2, pos)

        main_out = Upsample(dec2, x_size[2:])

        if self.training:
            loss1 = self.criterion(main_out, gts)

            if self.args.aux_loss is True:
                aux_out = self.dsn(aux_out)
                if aux_gts.dim() == 1:
                    aux_gts = gts
                aux_gts = aux_gts.unsqueeze(1).float()
                aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
                aux_gts = aux_gts.squeeze(1).long()
                loss2 = self.criterion_aux(aux_out, aux_gts)
                if attention_loss:
                    return (loss1, loss2, last_attention)
                else:
                    return (loss1, loss2)
            else:
                if attention_loss:
                    return (loss1, last_attention)
                else:
                    return loss1
        else:
            if attention_map:
                return main_out, attention_maps, pos_maps
            else:
                return main_out


def get_final_layer(model):
    unfreeze_weights(model.final)
    return model.final


def DeepR50V3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3PlusHANet(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepR50V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3PlusHANet(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3PlusHANet(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3PlusHANet(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepR152V3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 152 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-152")
    return DeepV3PlusHANet(num_classes, trunk='resnet-152', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)



def DeepResNext50V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    Resnext 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-50 32x4d")
    return DeepV3PlusHANet(num_classes, trunk='resnext-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepResNext101V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    Resnext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-101 32x8d")
    return DeepV3PlusHANet(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3PlusHANet(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3PlusHANet(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepWideResNet101V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3PlusHANet(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet101V3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3PlusHANet(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepResNext101V3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3PlusHANet(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepResNext101V3PlusD_HANet_OS4(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3PlusHANet(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D4', skip='m1', args=args)

def DeepShuffleNetV3PlusD_HANet_OS32(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3PlusHANet(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepMNASNet05V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_0_5")
    return DeepV3PlusHANet(num_classes, trunk='mnasnet_05', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMNASNet10V3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_1_0")
    return DeepV3PlusHANet(num_classes, trunk='mnasnet_10', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)


def DeepShuffleNetV3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3PlusHANet(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD_HANet(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3PlusHANet(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3PlusHANet(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepShuffleNetV3PlusD_HANet_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3PlusHANet(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
