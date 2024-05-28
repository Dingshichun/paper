import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from CBAM import ChannelAttention,SpatialAttention
# _make_divisible函数将输入通道数更改到最接近的8的整数倍，返回值为新的通道数new_ch，方便设备进行运算。
# 比如输入通道数为55，那么更改为56,49则更改为48。
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

#
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 这个类实现的是MBConv结构中的卷积、批归一化和激活三个操作，集中到一个类中方便实现。
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int, # 输入特征矩阵的通道
                 out_planes: int, # 输出特征矩阵的通道
                 kernel_size: int = 3, #卷积核尺寸，默认为3，还可以是5
                 stride: int = 1, # 步距，默认是1，还可以是2。
                 groups: int = 1, # 控制使用普通的卷积还是深度可分离卷积depthwise convolution（DW），
                 norm_layer: Optional[Callable[..., nn.Module]] = None, # MBConv中的BN结构，即批归一化，默认为None
                 activation_layer: Optional[Callable[..., nn.Module]] = None): # BN结构之后的激活函数，默认为None
        padding = (kernel_size - 1) // 2 # 根据卷积核尺寸判断填充数据的大小。
        if norm_layer is None: # 如果BN为None，将其设置为2维批归一化BatchNorm2d
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:  # 如果激活函数为空，将其设置为Hardswish
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes), # 对输出特征进行批归一化
                                               activation_layer()) # 激活

# 压缩激发模块SE，在MBConv结构中有使用，视为一个完整结构进行调用即可
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x

# 设置每一个MBConv模块的参数，因为Efficientnet主要由不同的MBConv结构组成，
# 每个MBConv结构的参数不同。
class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 卷积核的尺寸为 3 or 5 两种
                 input_c: int,         # 输入通道数
                 out_c: int,           # 输出通道数
                 # expanded_ratios是MBConv中第一个1×1卷积层的通道扩充倍数，
                 # 即将输入特征矩阵的channel扩充为expanded_ratio倍，，为 1 or 6
                 expanded_ratio: int,
                 stride: int,          # 深度可分离卷积depthwise convolution的步距，为 1 or 2
                 use_se: bool,         # ，是否使用SE模块，默认为True
                 drop_rate: float,     # MBConv中Dropout的随机失活比率
                 index: str,           # 记录当前的MBConv结构的名称，类似于1a, 2a, 2b, ...
                 width_coefficient: float): # 网络宽度缩放的倍率因子，宽度指的就是卷积核的个数。b0到b7倍率不同。
        # adjust_channels是自定义的静态方法，自适应调整输入通道数，调整到最接近的8的整数倍。
        # 乘以width_coefficient后再调整到8的整数倍。
        # b0到b7模型输入和输出都不同，都是在b0的基础上乘以某倍率因子。
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index
    # adjust_channels是自定义的静态方法，即自适应调整输入通道数
    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)

# MBConv模块
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig, # 定义MBConv结构参数的类
                 norm_layer: Callable[..., nn.Module]): # 批归一化结构BN
        super(InvertedResidual, self).__init__()
        # 判断步距是否为1或2，抛出值错误
        # 如果MBConv结构中的步距参数stride不是1或2
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        # 判断MBConv结构是否使用shortcut捷径分支，
        # 结合MBConv的结构图进行理解，
        # 只有当步距为1，且输入特征矩阵的通道数等于输出特征矩阵的通道数时才使用捷径分支。
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        # 定义一个有序字典OrderedDict用于存放MBConv结构中的各个模块。
        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand，判断expanded_ratio是否为1，如果为1，则expanded_c和input_c相等，
        # 所以cnf.expanded_c != cnf.input_c表示expanded_ratio不为1。
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise卷积
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c, # 经过DW卷积时，通道数不改变，所以输入通道数和输出通道数都是expanded_c
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})
        # depthwise卷积之后是SE模块。
        if cnf.use_se: # 是否使用SE模块，默认使用SE模块。添加SE模块
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})
        # SE模块之后是1X1卷积，即Conv1X1。搭建SE模块之后的1X1卷积层，命名为project_conv
        # 调用ConvBNActivation类，这里的Conv1X1同样有卷积和BN，只是没有最后的激活，所以激活函数为Identity，表示不做处理。
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})
        # Identity表示不做任何处理。因为经过1X1卷积之后只有批归一化，没有激活函数，所以激活函数做Identity处理。

        # 将前面得到的layers，也就是MBConv的结构传入nn.Sequential()，就得到MBConv的主分支，
        # 将主分支传递给block。
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        # 判断stride是否大于1
        self.is_strided = cnf.stride > 1

        # 只有在使用捷径分支shortcut时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity() # 不做处理
    # 定义前向传播过程
    def forward(self, x: Tensor) -> Tensor:
        # block是MBConv的主分支，包括1X1卷积（包括BN和激活），depthwise卷积（包括BN和激活），SE模块，1X1卷积（包括BN）。
        result = self.block(x)
        # 然后是随机失活dropout
        result = self.dropout(result)
        # 如果使用了捷径分支，将输入x和最后经过dropout输出的result做相加处理
        if self.use_res_connect:
            result += x

        return result



class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float, # 宽度倍率因子
                 depth_coefficient: float, # 深度倍率因子，指的是每个stage中MBConv结构重复的次数。
                 num_classes: int = 1000, # 类别数，默认为1000，
                 dropout_rate: float = 0.2, # stage9中的全连接层FC之前的随机失活率，默认为0.2
                 drop_connect_rate: float = 0.2, # MBConv中dropout层的随机失活率，不是固定的0.2，而是从0慢慢到0.2
                 block: Optional[Callable[..., nn.Module]] = None, # 就是MBConv模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None # 普通的BN结构
                 ):
        super(EfficientNet, self).__init__()

        # 默认配置表，记录stage2到stage8的参数
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        # drop_connect_rate是从0逐渐变化到0.2，这里首先初始化为默认的0.2，后续再进行调整，
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]
        # 定义重复次数函数，仅针对stage2到stage8的MBConv结构的重复次数，即深度。
        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))
        #MBConv模块
        if block is None:
            block = InvertedResidual
        # 批归一化结构BN
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1) # 构建BN层需要设置的超参数

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting，
        # 为MBConv的配置文件InvertedResidualConfig传入参数width_coefficient
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0 # 统计搭建MBConv模块的次数
        # num_blocks是总的MBConv结构的重复次数。
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = [] # 存储MBConv模块的配置文件
        for stage, args in enumerate(default_cnf): # 遍历每个stage，enumerate方法得到数据的同时返回数据的索引。
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))): # 遍历每个stage中的MBConv模块
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv ，stage1，即第一个3X3卷积，命名为stem_conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=3, # 输入的是RGB，所以输入通道数为3
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        # 遍历配置文件列表
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top，stage9
        # 创建stage9
        # stage9中1×1卷积的输入通道数last_conv_input_c是stage8的输出通道数，即inverted_residual_setting[-1].out_c
        last_conv_input_c = inverted_residual_setting[-1].out_c
        # stage9中1×1卷积的输出通道数是1280，根据参数表可知。
        last_conv_output_c = adjust_channels(1280)
        # 构建1×1卷积层
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})
        # 将有序字典传递给nn.Sequential可创建self.features，features包含stage1到stage8以及stage9中的1×1卷积
        self.features = nn.Sequential(layers)
        # 然后是池化avgpool，池化之后是全连接FC
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 构建全连接层FC
        classifier = [] # 创建一个空的分类器
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        # 全连接层的输入节点个数是stage9中1×1卷积的输出通道数，输出节点个数是最终的类别数。
        # 使用append方法将全连接层添加到classifier分类器中。
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        # 使用nn.Sequential方法实例化分类器。
        self.classifier = nn.Sequential(*classifier)

        # initial weights 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)
