import torch.nn as nn
import torch
import torch.nn.functional as F

class vgg16(nn.Module):

    def __init__(self, model_dict=None):
        super(vgg16, self).__init__()

        # conv: in channels, out channels, kernel size, stride, padding
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # pool: kernel_size, stride, padding
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # upsampling direction
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv10_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv10_3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)


        if model_dict:
            self.load(model_dict)

    def load(self, model_dict):
        print("loading checkpoint...")
        # conv: in channels, out channels, kernel size, stride, padding
        self.conv1_1.weight.data = model_dict['conv1_1_weight']
        self.conv1_1.bias.data = model_dict['conv1_1_bias']
        self.conv1_2.weight.data = model_dict['conv1_2_weight']
        self.conv1_2.bias.data = model_dict['conv1_2_bias']

        self.conv2_1.weight.data = model_dict['conv2_1_weight']
        self.conv2_1.bias.data = model_dict['conv2_1_bias']
        self.conv2_2.weight.data = model_dict['conv2_2_weight']
        self.conv2_2.bias.data = model_dict['conv2_2_bias']

        self.conv3_1.weight.data = model_dict['conv3_1_weight']
        self.conv3_1.bias.data = model_dict['conv3_1_bias']
        self.conv3_2.weight.data = model_dict['conv3_2_weight']
        self.conv3_2.bias.data = model_dict['conv3_2_bias']
        self.conv3_3.weight.data = model_dict['conv3_3_weight']
        self.conv3_3.bias.data = model_dict['conv3_3_bias']

        self.conv4_1.weight.data = model_dict['conv4_1_weight']
        self.conv4_1.bias.data = model_dict['conv4_1_bias']
        self.conv4_2.weight.data = model_dict['conv4_2_weight']
        self.conv4_2.bias.data = model_dict['conv4_2_bias']
        self.conv4_3.weight.data = model_dict['conv4_3_weight']
        self.conv4_3.bias.data = model_dict['conv4_3_bias']

        self.conv5_1.weight.data = model_dict['conv5_1_weight']
        self.conv5_1.bias.data = model_dict['conv5_1_bias']
        self.conv5_2.weight.data = model_dict['conv5_2_weight']
        self.conv5_2.bias.data = model_dict['conv5_2_bias']
        self.conv5_3.weight.data = model_dict['conv5_3_weight']
        self.conv5_3.bias.data = model_dict['conv5_3_bias']


        # this is to check if the full network was pretrained or only the encoder part
        # if we find one, we assume all are present
        # for now untested
        if "conv6_1" in model_dict:
            print("found pretrained decoder")
            self.conv6_1.weight.data = model_dict['conv6_1_weight']
            self.conv6_1.bias.data = model_dict['conv6_1_bias']
            self.conv6_2.weight.data = model_dict['conv6_2_weight']
            self.conv6_2.bias.data = model_dict['conv6_2_bias']
            self.conv6_3.weight.data = model_dict['conv6_3_weight']
            self.conv6_3.bias.data = model_dict['conv6_3_bias']

            self.conv7_1.weight.data = model_dict['conv7_1_weight']
            self.conv7_1.bias.data = model_dict['conv7_1_bias']
            self.conv7_2.weight.data = model_dict['conv7_2_weight']
            self.conv7_2.bias.data = model_dict['conv7_2_bias']
            self.conv7_3.weight.data = model_dict['conv7_3_weight']
            self.conv7_3.bias.data = model_dict['conv7_3_bias']

            self.conv8_1.weight.data = model_dict['conv8_1_weight']
            self.conv8_1.bias.data = model_dict['conv8_1_bias']
            self.conv8_2.weight.data = model_dict['conv8_2_weight']
            self.conv8_2.bias.data = model_dict['conv8_2_bias']
            self.conv8_3.weight.data = model_dict['conv8_3_weight']
            self.conv8_3.bias.data = model_dict['conv8_3_bias']

            self.conv9_1.weight.data = model_dict['conv9_1_weight']
            self.conv9_1.bias.data = model_dict['conv9_1_bias']
            self.conv9_2.weight.data = model_dict['conv9_2_weight']
            self.conv9_2.bias.data = model_dict['conv9_2_bias']

            self.conv10_1.weight.data = model_dict['conv10_1_weight']
            self.conv10_1.bias.data = model_dict['conv10_1_bias']
            self.conv10_2.weight.data = model_dict['conv10_2_weight']
            self.conv10_2.bias.data = model_dict['conv10_2_bias']
            self.conv10_3.weight.data = model_dict['conv10_3_weight']
            self.conv10_3.bias.data = model_dict['conv10_3_bias']




    def forward(self, x):
        # downsample
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu((self.conv2_1(x)))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        # upsample
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = self.upsample1(x)
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = F.relu(self.conv7_3(x))
        x = self.upsample2(x)
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        x = F.relu(self.conv8_3(x))
        x = self.upsample3(x)
        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x))
        x = self.upsample4(x)
        x = F.relu(self.conv10_1(x))
        x = F.relu(self.conv10_2(x))
        x = torch.sigmoid(self.conv10_3(x))

        return x