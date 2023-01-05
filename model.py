import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet18, ResNet18_Weights
import copy

from torch.autograd import Variable

from utils import mixup_data


def get_standard_model(dataset_name, device,pretrained=True):
    if dataset_name.endswith("mnist"):
        return MnistNet(device=device)
    else:
        return CifarResNet(device=device, n_out=(10 if dataset_name.endswith("10") else 100), pretrained=pretrained)
        # return Vgg()


def get_vae(dataset_name, h_dim1, h_dim2, z_dim):
    if dataset_name.endswith("mnist"):
        return MnistVAE(x_dim=28 * 28, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    else:
        return CifarVAE(image_size=32, channel_num=3, kernel_num=256, z_size=z_dim)


def get_gan(z_dim):
    return GAN(z_dim=z_dim)


def get_gan_initializer(z_dim):
    return GAN_initializer(z_dim)


class GAN_initializer(nn.Module):
    def __init__(self, z_dim):
        super(GAN_initializer, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, 3, 1)
        # self.conv2 = nn.Conv2d(16, 32, 3, 1)
        # self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216 // 2, 128)
        # self.fc2 = nn.Linear(128, 1024)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, z_dim),
        )
        self.z_dim = z_dim

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = x.view(-1, 1024, 1, 1)
        # return x
        return self.model(x).view(-1, self.z_dim, 1, 1)


def get_feature_extractor():
    return Feature_extractor()

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216 // 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def extract_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x

    def classify(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classify(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        device=None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.device = device

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=1.0, layer_mix=None):
        if mixup_hidden:
            if layer_mix is None:
                layer_mix = random.randint(0, 4)
            out = x
            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.layer1(out)
            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.layer2(out)
            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.layer3(out)
            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out, y_a, y_b, lam
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            return x


class CifarResNet(nn.Module):
    def __init__(self, device, n_out, pretrained):
        super(CifarResNet, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], device=device)
        if pretrained:
            print("Loading pretrained model")
            self.model.load_state_dict(torch.load("./models/cifar10/standard/resnet18.pt"))
        self.feature_extractor = copy.deepcopy(self.model)
        self.feature_extractor.fc = nn.Identity()
        self.classifier = copy.deepcopy(self.model.fc)
        del self.model

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=1.0, layer_mix=None):
        if mixup_hidden:
            out, y_a, y_b, lam = self.feature_extractor(x, target, mixup_hidden, mixup_alpha, layer_mix)
            out = self.classifier(out)
            return out, y_a, y_b, lam
        else:
            x = self.feature_extractor(x)
            x = self.classifier(x)
            return x


class MnistNet(nn.Module):
    def __init__(self, device="cuda"):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216 // 2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=1.0, layer_mix=None):
        if mixup_hidden:
            if layer_mix is None:
                layer_mix = random.randint(0, 3)
            out = x

            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv1(out)
            out = F.relu(out)

            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = torch.flatten(out, 1)

            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.dropout(out)

            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc2(out)
            out = self.softmax(out)
            return out, y_a, y_b, lam
        else:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            # x = self.softmax(x)
            return x


class MnistVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(MnistVAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


def vae_loss_function(recon_x, x, mu, log_var, x_dim):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


class CifarVAE(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, z_size):
        # configurations
        super().__init__()
        self.x_dim = 32*32*3
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        # encoder
        self.encoder_net = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=True)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=True)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=True)

        # decoder
        self.decoder_net = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def encoder(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
        encoded = self.encoder_net(x)
        return self.q(encoded)  # mu, log_var

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        x_reconstructed = self.decoder_net(z_projected)
        return x_reconstructed.view(-1, self.x_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    @staticmethod
    def _conv(channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    @staticmethod
    def _deconv(channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    @staticmethod
    def _linear(in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        ls = z_dim
        
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(ls, int(ls/2), kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(int(ls/2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/2), int(ls/4), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/4)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/4), int(ls/8), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/8)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/8), int(ls/16), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/16)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/16), 1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.Tanh()
            )

    def forward(self, x):
        x = self.seq(x)
        x = x[:, :, 0:28, 0:28]
        return x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        ls = z_dim
        
        self.seq = nn.Sequential(
            nn.Conv2d(1, int(ls/16), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/16)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/16), int(ls/8), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/8)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/8), int(ls/4), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/4)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/4), int(ls/2), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/2)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/2), 1,  kernel_size=3, stride=2, padding=1, bias=True),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.seq(x)
        return x


class GAN(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.generator:Generator = Generator(z_dim)
        self.discriminator:Discriminator = Discriminator(z_dim)

    def forward(self, x):
        return self.generator(x)

    



class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(), nn.Linear(512, 10))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=1.0, layer_mix=None):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """

        if mixup_hidden:
            if layer_mix is None:
                layer_mix = random.randint(0, 4)
            out = x

            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_block1(out)

            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_block2(out)

            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_block3(out)

            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_block4(out)

            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_block5(out)

            out = out.view(out.size(0), -1)
            score = self.classifier(out)

            return score, y_a, y_b, lam

        else:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.conv_block5(x)
            # flatten
            x = x.view(x.size(0), -1) 
            score = self.classifier(x) 

            return score

