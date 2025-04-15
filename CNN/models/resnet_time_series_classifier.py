"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import torch.nn as nn

from .resnet import ResNet

from .custom_layers import Conv1dPadSame


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

    if isinstance(m, Conv1dPadSame):
        nn.init.xavier_uniform_(m.conv.weight)
        m.conv.bias.data.fill_(0.0)

    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class ResNetTimeSeriesClassifier(nn.Module):
    def __init__(self, classifier_params, encoder_params, dropout=0.):
        super(ResNetTimeSeriesClassifier, self).__init__()

        self.encoder = ResNet(**encoder_params)

        self.encoder.apply(init_weights)

        self.classifier = nn.Linear(self.encoder.encoding_size,
                                    classifier_params['out_channels'])
        # Xavier/He weights initialization
        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.encoder(x)
        x = self.classifier(x)

        return x
