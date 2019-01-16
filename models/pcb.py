import torch

from torch import nn
from torchvision.models import resnet50


class PCBModel(nn.Module):
    def __init__(self, num_class=None, num_parts=6, bottleneck_dims=256, pool_type="avg", share_embed=False):
        super(PCBModel, self).__init__()

        assert pool_type in ['max', 'avg']
        self.backbone = resnet50(pretrained=True)

        # remove the final downsample
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        self.backbone.layer4[0].conv2.stride = (1, 1)

        if pool_type == "max":
            self.part_pool = nn.AdaptiveAvgPool2d((num_parts, 1))
        else:
            self.part_pool = nn.AdaptiveMaxPool2d((num_parts, 1))

        # classifier
        if share_embed:
            embed = nn.Sequential(nn.Linear(2048, bottleneck_dims, bias=False),
                                  nn.BatchNorm1d(num_features=bottleneck_dims),
                                  nn.ReLU(inplace=True))
            self.embed = nn.ModuleList([embed for _ in range(num_parts)])

        else:
            self.embed = nn.ModuleList([nn.Sequential(nn.Linear(2048, bottleneck_dims, bias=False),
                                                      nn.BatchNorm1d(num_features=bottleneck_dims),
                                                      nn.ReLU(inplace=True)) for _ in range(num_parts)])

        self.classifier = None
        if num_class is not None:
            self.classifier = nn.ModuleList(
                [nn.Linear(bottleneck_dims, num_class, bias=False) for _ in range(num_parts)])

    def forward(self, x):
        x = self.backbone_forward(x)

        # part pooling
        x = self.part_pool(x)
        x = x.squeeze()

        if not self.training:
            return self.eval_forward(x)
        else:
            return self.train_forward(x)

    def backbone_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def train_forward(self, x):
        part_logits = []
        for i in range(x.size(2)):
            embed = self.embed[i](x[:, :, i])
            embed = self.classifier[i](embed)

            part_logits.append(embed)

        return torch.cat(part_logits, dim=0)

    def eval_forward(self, x):
        embeds = []
        for i in range(x.size(2)):
            embed = self.embed[i](x[:, :, i])
            embeds.append(embed)

            # embeds.append(x[:, :, i])

        return torch.cat(embeds, dim=1)
