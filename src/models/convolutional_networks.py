import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import torchvision
import math

# The whole document is a rewritten version of https://github.com/YuanGongND/psla#Use-PSLA-Training-Pipeline-For-New-Datasets-and-Tasks
class efficient_net(nn.Module):
    def __init__(self, label_dimension, pretrain):
        super(efficient_net, self).__init__()
        self.label_dimension = label_dimension
        self.feature_out_dim = 1408

        if pretrain:
            # print("Using EfficientNetb2 model, pretrained on Imagenet")
            self.network = EfficientNet.from_pretrained(
                "efficientnet-b2", in_channels=1
            )

        else:
            # print("Training EfficientNetb2 model from SCRATCH - buckle up, Imagenet won't save you now!")
            self.network = EfficientNet.from_name("efficientnet-b2", in_channels=1)

        self.mean_pooling = MeanPooling(self.feature_out_dim, self.label_dimension)

        self.avgpool = nn.AvgPool2d((4, 1))
        self.network._fc = nn.Identity()

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.network.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2, 3)
        # print("Shape of input to mean pooling layer: ", x.shape)
        out, norm_att = self.mean_pooling(x)
        return out


class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out):
        super(MeanPooling, self).__init__()

        self.cla = nn.Conv2d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        self.init_layer(self.cla)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        cla = self.cla(x)
        cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)
        cla = torch.mean(cla, dim=2)
        x = self.softmax(cla)

        return x, []

    def init_layer(self, layer):
        if layer.weight.ndimension() == 4:
            (n_out, n_in, height, width) = layer.weight.size()
            n = n_in * height * width
        elif layer.weight.ndimension() == 2:
            (n_out, n) = layer.weight.size()

        std = math.sqrt(2.0 / n)
        scale = std * math.sqrt(3.0)
        layer.weight.data.uniform_(-scale, scale)

        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

    def activate(self, x):
        return torch.sigmoid(x)


class mobile_net(nn.Module):
    def __init__(self, label_dimension, pretrain):
        super(mobile_net, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)
        self.softmax = nn.Softmax(dim=1)

        self.model.features[0][0] = torch.nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )  # Change input dimension to be 1 channel instead of 3 as is standard
        self.model.classifier = torch.nn.Linear(
            in_features=1280, out_features=label_dimension, bias=True
        )  # Init final layer for classification, transferlearning

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = self.softmax(self.model(x))
        return out


if __name__ == "__main__":
    input_tdim = 1103
    # ast_mdl = ResNetNewFullAttention(pretrain=False)
    psla_mdl = efficient_net(label_dimension=527, pretrain=True)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([64, input_tdim, 128])
    for i in range(5):
        test_output = psla_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    # print(test_output.shape)

