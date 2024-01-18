from models.vit import ViT
from models.vision_mamba import VisionMamba
from models.resnet import ResNet

__all__ = ["ViT", "VisionMamba", "ResNet"]


def build_model(config):
    if config.name == "vit":
        model = ViT(
            height=config.height,
            width=config.width,
            patch_size=config.patch_size,
            dim=config.dim,
            n_layers=config.n_layers,
            n_classes=config.num_classes,
            n_heads=config.n_heads,
            mlp_factor=config.mlp_factor,
            channels=config.channels,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            mlp_drop=config.mlp_drop,
            bias=config.bias,
        )
    elif config.name == "mamba":
        model = VisionMamba(
            height=config.height,
            width=config.width,
            patch_size=config.patch_size,
            dim=config.dim,
            n_layers=config.n_layers,
            n_classes=config.num_classes,
            channels=config.channels,
            block_type=config.block_fn,
            ssm_drop=config.ssm_drop,
        )
    elif config.name == "resnet":
        model = ResNet(
            num_blocks=config.num_blocks,
            num_classes=config.num_classes,
            block_fn=config.block_fn,
            dimensions=config.dimensions,
            first_kernel_size=config.first_kernel_size,
            identity_method=config.identity_method,
        )
    else:
        raise NotImplementedError(f"Model {config.name} not implemented")
    return model
