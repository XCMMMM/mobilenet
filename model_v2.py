from torch import nn
import torch

# ch--卷积核个数，输出特征矩阵的channel
# divisor--基数，要将ch调整为它的整数倍
# min_ch--最小通道数，默认为none
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    # 实现将输入ch调整为离它最近的8的倍数
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# Conv+batchNorm2d+Relu6
class ConvBNReLU(nn.Sequential):
    # groups--如果将groups设置成1，就是普通的卷积；如果将groups设置成输入特征层的in_channel，就是DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # padding--填充参数根据kernel_size设定
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

# 定义倒残差结构
class InvertedResidual(nn.Module):
    # expand_ratio--扩展因子，之前表格Table 1中的t
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # 隐层的channel，其实就是表格中第一层1x1 conv2d的输出channel--（tk）
        hidden_channel = in_channel * expand_ratio
        # self.use_shortcut--是否使用捷径分支，记住使用捷径分支的条件，stride=1且输入特征矩阵的shape和输出特征矩阵的shape相同
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 如果扩展因子expand_ratio==1，就不用添加1x1的卷积层
        # 因为此时既没有改变输入特征层的高宽，也没有改变channel大小
        # 如果扩展因子expand_ratio!=1，就需要添加1x1的卷积层
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            # 输入特征层的channel和输出特征层的channel是相同的，都等于hidden_channel
            # 此时groups=hidden_channel，即等于输入特征层的channel，此时卷积为DW卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 1x1的卷积层，此时针对我们倒残差结构的最后一个卷积层，使用了线性的激活函数
            # 由于线性激活函数y=x，就直接不需要对我们的输入做变换，所以就不需要再额外添加激活函数了
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    # x--输入的特征矩阵
    def forward(self, x):
        # 如果使用捷径分支，就将捷径分支上的输出加上主分支上的输出
        if self.use_shortcut:
            return x + self.conv(x)
        # 如果不适用捷径分支，就直接返回主分支上的输出
        else:
            return self.conv(x)

# 定义MobileNetV2网络结构
class MobileNetV2(nn.Module):
    # alpha--控制卷积层所采用的卷积核个数的倍率，超参数
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        # 将我们刚定义的InvertedResidual类传给block
        block = InvertedResidual
        # input_channel--采用卷积的卷积核个数，即输出特征矩阵的channel，由于有alpha，所以需要乘以alpha
        # _make_divisible--将卷积核个数调整为round_nearest的整数倍
        input_channel = _make_divisible(32 * alpha, round_nearest)
        # 1280--对应于表格中倒数第三行的1x1的卷积层，它对应的卷积核个数为1280
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        # 对应于表格中每一行对应的参数，每一行依次为t,c,n,s
        # t--上图表格中通过第一个1x1的卷积层所采用的卷积核的扩展倍率。
        # c--输出特征矩阵的深度，即上图表格中的k‘。
        # n--bottleneck重复的次数，即倒残差结构。
        # s--步幅。注意：只代表每一个block所对应的第一层bottleneck的步幅，其他的都为1。（一个block由一系列bottleneck组成）。
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        # 输入特征矩阵的channel为3（RGB），输出特征矩阵的channel
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            # 循环搭建每个block中的倒残差结构，n为倒残差结构重复的次数
            for i in range(n):
                # 倒残差结构中，除了第一层，其它层的stride都为1
                stride = s if i == 0 else 1
                # 通过append添加倒残差结构，卷积层所采用的卷积核的扩展倍率expand_ratio为t
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # 添加1x1卷积层
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 初始化为均值为0，方差为0.01
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
