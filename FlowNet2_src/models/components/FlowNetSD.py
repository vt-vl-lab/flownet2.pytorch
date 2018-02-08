import torch
import torch.nn as nn
import torch.nn.init as nn_init

from .misc import conv, deconv, predict_flow
'Parameter count = 45,371,666'


class FlowNetSD(nn.Module):

    def __init__(self, with_bn=True):
        super(FlowNetSD, self).__init__()

        self.with_bn = with_bn
        self.conv0 = conv(6, 64, with_bn=with_bn)
        self.conv1 = conv(64, 64, stride=2, with_bn=with_bn)
        self.conv1_1 = conv(64, 128, with_bn=with_bn)
        self.conv2 = conv(128, 128, stride=2, with_bn=with_bn)
        self.conv2_1 = conv(128, 128, with_bn=with_bn)
        self.conv3 = conv(128, 256, stride=2, with_bn=with_bn)
        self.conv3_1 = conv(256, 256, with_bn=with_bn)
        self.conv4 = conv(256, 512, stride=2, with_bn=with_bn)
        self.conv4_1 = conv(512, 512, with_bn=with_bn)
        self.conv5 = conv(512, 512, stride=2, with_bn=with_bn)
        self.conv5_1 = conv(512, 512, with_bn=with_bn)
        self.conv6 = conv(512, 1024, stride=2, with_bn=with_bn)
        self.conv6_1 = conv(1024, 1024, with_bn=with_bn)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.inter_conv5 = conv(1026, 512, with_bn=with_bn, with_relu=False)
        self.inter_conv4 = conv(770, 256, with_bn=with_bn, with_relu=False)
        self.inter_conv3 = conv(386, 128, with_bn=with_bn, with_relu=False)
        self.inter_conv2 = conv(194, 64, with_bn=with_bn, with_relu=False)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,
