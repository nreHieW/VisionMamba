import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from mamba_ssm import Mamba


class BasicBlock(nn.Module):
    def __init__(self, dim: int, ssm_drop: float = 0.0):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(ssm_drop)

    def forward(self, x, residual=None):
        residual = (x + residual) if residual is not None else x
        x = self.norm(x)
        hidden_states = self.mamba(x)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, residual


class PatchEmbedding(nn.Module):
    def __init__(self, height: int, width: int, patch_size: int, dim: int, in_channels):
        super().__init__()
        # takes in (b, c, h, w) -> (b, l, d)
        self.n_patches = (height / patch_size) * (width / patch_size)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        out = self.conv(x)  # (b, c, h, w) -> (b, dim, h', w') where h' * w' = L
        out = out.flatten(2)  # (b, dim, L)
        out = out.transpose(1, 2)  # (b, L, dim)
        return out


class BiDirectionalConcatBlock(nn.Module):
    def __init__(self, dim: int, ssm_drop: float = 0.0) -> None:
        super().__init__()
        self.out_dim = dim
        self.dim = dim // 2
        self.mamba1 = Mamba(
            d_model=self.dim,
        )
        self.mamba2 = Mamba(
            d_model=self.dim,
        )
        self.norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(ssm_drop)

    def forward(self, x, residual=None):
        residual = (x + residual) if residual is not None else x
        forward, backward = torch.split(x, self.dim, dim=-1)
        reverse_input = torch.flip(backward, dims=[1])
        hidden_states1 = self.mamba1(forward)
        hidden_states2 = self.mamba2(reverse_input)
        hidden_states = torch.cat((hidden_states1, hidden_states2), dim=-1)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, residual


class BiDirectionalAddBlock(nn.Module):
    def __init__(self, dim: int, ssm_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.mamba1 = Mamba(
            d_model=self.dim,
        )
        self.mamba2 = Mamba(
            d_model=self.dim,
        )
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(ssm_drop)

    def forward(self, x, residual=None):
        residual = (x + residual) if residual is not None else x
        x = self.norm(x)
        rev_input = torch.flip(x, dims=[1])
        hidden_states1 = self.mamba1(x)
        hidden_states2 = self.mamba2(rev_input)
        hidden_states = hidden_states1 + hidden_states2
        hidden_states = self.dropout(hidden_states)
        return hidden_states, residual


class MambaBackbone(nn.Module):
    def __init__(self, n_layers: int, dim: int, block_type: str, ssm_drop: float = 0.0):
        super().__init__()
        if block_type == "basic":
            block_fn = BasicBlock
        elif block_type == "bi_concat":
            block_fn = BiDirectionalConcatBlock
        elif block_type == "bi_add":
            block_fn = BiDirectionalAddBlock
        else:
            raise NotImplementedError(f"Block {block_type} not implemented")
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    ssm_drop=ssm_drop,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual)
        x = (x + residual) if residual is not None else x
        return self.norm(x)


class VisionMamba(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        dim: int,
        n_layers: int,
        n_classes: int,
        block_type: str,
        channels: int = 3,
        ssm_drop: float = 0.0,
    ):
        super().__init__()
        assert (
            height % patch_size == 0 and width % patch_size == 0
        ), "Invalid Patch Factor"

        self.patch_size = patch_size
        self.to_patch_embedding = PatchEmbedding(
            height, width, self.patch_size, dim, channels
        )

        self.backbone = MambaBackbone(n_layers, dim, block_type, ssm_drop)
        self.norm = nn.LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.head = nn.Linear(dim, n_classes, bias=False)

    def forward(self, x):
        b_size = x.shape[0]
        x = self.to_patch_embedding(x)
        cls_token = self.cls_token.expand(b_size, -1, -1)
        x = torch.cat((x, cls_token), dim=1)

        x = self.backbone(x)

        cls_token_end = x[:, -1]
        pred = self.head(cls_token_end)

        return pred
