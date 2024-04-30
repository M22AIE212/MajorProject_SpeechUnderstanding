import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8*227*227, 512),
            nn.ReLU(inplace=True),
        )

        self.fc_out = nn.Linear(512, 2)

    def forward_once(self, x):
        output = self.conv_layers(x)
        output = output.view(output.size()[0], -1)
        output = self.fc_layers(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return self.fc_out(torch.abs(output1 - output2))
