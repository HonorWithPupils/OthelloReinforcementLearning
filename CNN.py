import torch.nn as nn

from tqdm.auto import tqdm

from torch import Tensor
from typing import Tuple, List, Union


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# CNN part


class Block(nn.Module):
    """convolution => [BN] => ReLU"""

    def __init__(
        self, dim_in: int, dim_out: int, kernel_size: Union[int, Tuple[int, int]] = 3
    ) -> None:
        super().__init__()

        kernel_size = pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[0] // 2

        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU()  # nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """x => Block * 2 => y => x + y"""

    def __init__(
        self, dim_in: int, dim_out: int, kernel_size: Union[int, Tuple[int, int]]
    ) -> None:
        super().__init__()

        self.block1 = Block(dim_in, dim_out, kernel_size)
        self.block2 = Block(dim_out, dim_out, kernel_size)
        self.res_conv = (
            nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:

        y = self.block1(x)
        y = self.block2(y)

        return y + self.res_conv(x)


class ResNet(nn.Module):
    def __init__(
        self, dims: List[int], kernel_size: Union[int, Tuple[int, int]]
    ) -> None:
        super().__init__()

        blocks = []

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            blocks.append(ResNetBlock(dim_in, dim_out, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:

        return self.blocks(x)
