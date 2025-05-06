import torch
from monai.losses import DiceLoss
from torch import nn


class SmoothRegularizer3D(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel
        self.kernel = nn.Parameter(
            torch.tensor([[[[[-1 / 26, -1 / 26, -1 / 26],
                             [-1 / 26, -1 / 26, -1 / 26],
                             [-1 / 26, -1 / 26, -1 / 26]],
                            [[-1 / 26, -1 / 26, -1 / 26],
                             [-1 / 26, 1, -1 / 26],
                             [-1 / 26, -1 / 26, -1 / 26]],
                            [[-1 / 26, -1 / 26, -1 / 26],
                             [-1 / 26, -1 / 26, -1 / 26],
                             [-1 / 26, -1 / 26, -1 / 26]]]]]).repeat(channel, 1, 1, 1, 1))

        print(self.kernel.shape)

    def forward(self, x) -> torch.Tensor:
        return torch.mean(torch.abs(torch.conv3d(x, self.kernel, groups=self.channel)))


class SmoothRegularizer2D(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel
        self.kernel = nn.Parameter(
            torch.tensor([[[[-1 / 8, -1 / 8, -1 / 8],
                            [-1 / 8, 1, -1 / 8],
                            [-1 / 8, -1 / 8, -1 / 8]]]]).repeat(channel, 1, 1, 1))

    def forward(self, x) -> torch.Tensor:
        return torch.mean(torch.abs(torch.conv2d(x, self.kernel, groups=self.channel)))


class L1Regularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return torch.mean(torch.abs(x))


class L2Regularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return torch.mean(x ** 2)


class ImageRegularizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        return torch.sum(self.relu(x - 1)) + torch.sum(self.relu(-x))


class Loss(nn.Module):
    def __init__(self,
                 weight_image_regularizer: float = 1.0,
                 weight_l1: float = 200.0,
                 weight_smooth: float = 500.0,
                 channel: int = 1,
                 dims: int = 3):
        super().__init__()
        if dims not in [2, 3]:
            raise ValueError(f"{dims} dim input not supported!")

        self.weight_image = weight_image_regularizer
        self.weight_l1 = weight_l1
        self.weight_smooth = weight_smooth

        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        # self.image_reg = ImageRegularizer()
        self.value_reg = L1Regularizer()
        smooth_reg = SmoothRegularizer2D if dims == 2 else SmoothRegularizer3D
        self.smooth_reg = smooth_reg(channel)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                change: torch.Tensor,
                new_image: torch.Tensor):
        # normal loss
        prediction_loss = self.loss_fn(pred, target)

        # penalize values out of image range
        # image_reg = self.image_reg(new_image)

        # penalize high change values
        # value_reg = self.value_reg(change)

        # regularize smoothness
        # smooth_reg = self.smooth_reg(change)

        # loss = prediction_loss + value_reg * self.weight_l1 + smooth_reg * self.weight_smooth

        loss_dict = {
            "loss": prediction_loss, # loss.detach().item(),
            "prediction_loss": prediction_loss.detach().item(),
            # "image_reg": image_reg.detach().item(),
            "value_reg": 0.0, # value_reg.detach().item(),
            "smooth_reg": 0.0 # smooth_reg.detach().item(),
        }
        return prediction_loss, loss_dict


if __name__ == '__main__':
    loss_fn = Loss(channel=2)
    image = torch.randint(0, 255, (1, 256, 256, 256)) / 255
    pred = torch.randn(1, 2, 256, 256, 256)
    target = torch.randint(0, 1, (1, 256, 256, 256))
    change = torch.randn(1, 2, 256, 256, 256)
    new_image = image + change
    out = loss_fn(pred, target, change, new_image)
    print(out)
