import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self, use_perception_loss=True):
        super(GeneratorLoss, self).__init__()
        self.use_perception_loss = use_perception_loss
        if use_perception_loss:
            vgg = vgg16(pretrained=True)
            loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            self.loss_network = loss_network
        else:
            self.loss_network = None
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, fake_out, fake_img, target):
        # Image Loss
        mse_loss = self.mse_loss(fake_img, target)

        # Adversarial Loss
        bce_loss = self.bce_loss(fake_out, torch.ones_like(fake_out))

        # intermediate results
        total_loss = bce_loss + mse_loss

        # Perception Loss
        # only applied to Feng dataset
        if self.use_perception_loss:
            vgg_outimage = self.loss_network(fake_img)
            vgg_targetimage = self.loss_network(target)
            vgg_loss = self.mse_loss(vgg_outimage, vgg_targetimage.detach())
            total_loss += 0.006 * vgg_loss
        return total_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
