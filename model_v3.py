from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

# 实现将输入ch调整为离它最近的8的倍数
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

# Conv+batchNorm2d+激活函数
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 # 输入特征矩阵的channel
                 in_planes: int,
                 # 输出特征矩阵的channel，对应于卷积的卷积核个数
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 # groups--如果将groups设置成1，就是普通的卷积；如果将groups设置成输入特征层的in_channel，就是DW卷积
                 groups: int = 1,
                 # BN层
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 # 激活函数
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))

# 注意力机制（SE）模块，由两个全连接层组成
# 第一个全连接层的节点个数等于输入特征矩阵的channel/4
# 第二个全连接层的节点个数与输入特征矩阵的channel一致
# 第一个全连接层的激活函数为relu，第二个全连接层的激活函数为hard-sigmoid
class SqueezeExcitation(nn.Module):
    # squeeze_factor--默认设置为4，第一个全连接层的节点个数等于输入特征矩阵的channel/4
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        # 将输入特征矩阵每一个channel上的数据平均池化到一个1x1的大小
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # 将全连接层输出的数据与每一个x相乘，相当于是给它一个权重
        return scale * x

# InvertedResidualConfig对应的是mobilenet v3中的每一个bneck结构的参数配置
class InvertedResidualConfig:
    def __init__(self,
                 # 输入channel
                 input_c: int,
                 # Dwise（DW卷积）所对应的卷积层的卷积核大小
                 kernel: int,
                 # 对应于第一个1x1卷积层所使用的卷积核个数，起到一个升维的作用
                 expanded_c: int,
                 # 输出channel
                 out_c: int,
                 # 是否使用SE模块（注意力机制模块）
                 use_se: bool,
                 # 激活函数
                 activation: str,
                 # Dwise卷积层所对应的步距
                 stride: int,
                 # 调节每一个卷积层所使用channel的倍率因子
                 width_multi: float):
        # 先将输入的input_c*倍率因子width_multi，然后调用_make_divisible函数架将channel调整到离它最近的8的倍数
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        # 是否使用HS（hard-swish）激活函数，是为true，不是为false
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

# 每一层bneck的结构
class InvertedResidual(nn.Module):
    def __init__(self,
                 # 传入config文件，即刚刚定义的InvertedResidualConfig
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        # 判断stride是否为1或者2，如果都不是，那就是一个非法的数字
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 判断是否使用捷径分支（short-cut连接）
        # 使用的条件：stride == 1和input_c == output_c
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        # 判断使用哪个激活函数
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        # expanded_c和input_c相等的话，就没有进行一个升维的操作，此时就没有1x1卷积层
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        # DW卷积的输入channel和输出channel是相等的
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        # 判断是否使用SE模块
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project
        # 最后一个1x1的卷积层实现降维，采用的激活函数为线性激活
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        # 判断是否使用short-cut连接
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 # 一系列bneck结构参数的列表
                 inverted_residual_setting: List[InvertedResidualConfig],
                 # 倒数第二层卷积层的输出channel个数
                 last_channel: int,
                 # 需要分类的类别个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        # 默认为空，就是上面定的InvertedResidual模块
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # partial--为我们的BatchNorm2d传入了两个默认的参数（eps--防止分母为0）
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        # 获取第一个卷积层输出的channel，它就是你一个bneck结构的输入channel
        firstconv_output_c = inverted_residual_setting[0].input_c
        # 创建第一个卷积层，卷积核大小3x3，激活函数采用Hardswish
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        # 将第一个卷积层后的bneck结构依次添加到layers中
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        # 获取最后一个bneck结构的输出channel，它就是我们下一个卷积层的输入channel
        lastconv_input_c = inverted_residual_setting[-1].out_c
        # 卷积层的输出channel为输入channel的6倍
        lastconv_output_c = 6 * lastconv_input_c
        # 定义bneck结构后的卷积层，卷积核大小为1x1，使用的激活函数为Hardswish
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # features--主干特征提取网络，包括开始的3x3卷积层，一系列bneck结构，和最后的一个1x1卷积层
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:pytorch官方用mobilenet_v3_large在imagenet数据集上训练好的一个预训练权重
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    # 调节每一个卷积层所使用channel的倍率因子，默认为1.0
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    # 默认不使用，使用的话会对最后三个bneck结构的channel进行调整，会使网络大小变小
    reduce_divider = 2 if reduced_tail else 1

    # 每一个bneck参数的设置
    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
