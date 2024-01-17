from dataclasses import dataclass, field
from argparse import ArgumentParser
import torch


@dataclass
class ModelArgs:
    name: str
    num_classes: int = 10

    # Vit and Mamba
    patch_size: int = 4
    dim: int = 256
    n_layers: int = 6
    height: int = 32
    width: int = 32
    channels: int = 3
    bias: bool = True

    # ViT
    mlp_factor: int = 4
    n_heads: int = 8
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    mlp_drop: float = 0.0

    # Mamba
    ssm_drop: float = 0.0

    # ResNet
    num_blocks: list = field(default_factory=lambda: [1, 1, 1, 1])
    block_fn: str = "residual"
    dimensions: list = field(default_factory=lambda: [64, 128, 256, 512])
    first_kernel_size: int = 3
    identity_method: str = "B"

    def get_relevant_args(self) -> dict:
        base = {
            "name": self.name,
            "num_classes": self.num_classes,
        }

        if self.name == "vit" or self.name == "mamba":
            base["patch_size"] = self.patch_size
            base["dim"] = self.dim
            base["n_layers"] = self.n_layers
            base["height"] = self.height
            base["width"] = self.width
            base["channels"] = self.channels
            base["bias"] = self.bias

        if self.name == "mamba":
            base["ssm_drop"] = self.ssm_drop

        if self.name == "vit":
            base["mlp_factor"] = self.mlp_factor
            base["n_heads"] = self.n_heads
            base["attn_drop"] = self.attn_drop
            base["proj_drop"] = self.proj_drop
            base["mlp_drop"] = self.mlp_drop

        if self.name == "resnet":
            base["num_blocks"] = self.num_blocks
            base["block_fn"] = self.block_fn
            base["dimensions"] = self.dimensions
            base["first_kernel_size"] = self.first_kernel_size
            base["identity_method"] = self.identity_method

        return base

    def __str__(self):
        base_str = f"Model: {self.name}\n\tNum classes: {self.num_classes}\n\t"
        if self.name == "vit" or self.name == "mamba":
            base_str += f"Patch size: {self.patch_size}\n\t"
            base_str += f"Dim: {self.dim}\n\t"
            base_str += f"Num layers: {self.n_layers}\n\t"
            base_str += f"Height: {self.height}\n\t"
            base_str += f"Width: {self.width}\n\t"
            base_str += f"Channels: {self.channels}\n\t"
            base_str += f"Bias: {self.bias}\n\t"
        if self.name == "vit":
            base_str += f"MLP factor: {self.mlp_factor}\n\t"
            base_str += f"Num heads: {self.n_heads}\n\t"
            base_str += f"Attn drop: {self.attn_drop}\n\t"
            base_str += f"Proj drop: {self.proj_drop}\n\t"
            base_str += f"MLP drop: {self.mlp_drop}\n\t"
        if self.name == "resnet":
            base_str += f"Num blocks: {self.num_blocks}\n\t"
            base_str += f"Block fn: {self.block_fn}\n\t"
            base_str += f"Dimensions: {self.dimensions}\n\t"
            base_str += f"First kernel size: {self.first_kernel_size}\n\t"
            base_str += f"Identity method: {self.identity_method}\n\t"

        return base_str


@dataclass
class TrainingArgs:
    train_dtype: str  # bf16 / fp32 / fp16

    save_path: str

    wandb: bool = True
    wandb_project: str = ""
    run_name: str = ""
    lr: float = 3e-4
    num_epochs: int = 200
    weight_decay: float = 0.0001
    optimizer: str = "sgd"  # or adamw
    betas: tuple = (0.9, 0.999)  # AdamW
    pct_start: float = 0.01  # OneCycleLR
    final_lr_ratio: float = 0.07  # OneCycleLR
    anneal_strategy: str = "linear"  # or cosine
    batch_size: int = 1024
    eval_batch_size: int = 2500
    seed: int = 1337
    log_interval: int = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision: bool = False

    def __str__(self):
        base_str = f"Training args:\n\t"
        base_str += f"Train dtype: {self.train_dtype}\n\t"
        base_str += f"Device: {self.device}\n\t"
        base_str += f"Mixed precision: {self.mixed_precision}\n\t"
        base_str += f"Save path: {self.save_path}\n\t"

        if self.wandb:
            base_str += f"Wandb project: {self.wandb_project}\n\t"
            base_str += f"Run name: {self.run_name}\n\t"

        base_str += f"Learning rate: {self.lr}\n\t"
        base_str += f"Num epochs: {self.num_epochs}\n\t"
        base_str += f"Weight decay: {self.weight_decay}\n\t"
        base_str += f"Optimizer: {self.optimizer}\n\t"
        if self.optimizer == "adamw":
            base_str += f"Betas: {self.betas}\n\t"
        base_str += f"Pct start: {self.pct_start}\n\t"
        base_str += f"Final lr ratio: {self.final_lr_ratio}\n\t"
        base_str += f"Anneal strategy: {self.anneal_strategy}\n\t"
        base_str += f"Batch size: {self.batch_size}\n\t"
        base_str += f"Eval batch size: {self.eval_batch_size}\n\t"
        base_str += f"Seed: {self.seed}\n\t"
        base_str += f"Log interval: {self.log_interval}\n\t"

        return base_str


def parse_args() -> (ModelArgs, TrainingArgs):
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="mamba")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mlp-factor", type=int, default=4)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--ssm-drop", type=float, default=0.0)
    parser.add_argument("--attn-drop", type=float, default=0.0)
    parser.add_argument("--proj-drop", type=float, default=0.0)
    parser.add_argument("--mlp-drop", type=float, default=0.0)
    parser.add_argument("--num-blocks", nargs="+", type=int, default=[1, 1, 1, 1])
    parser.add_argument("--block-fn", type=str, default="residual")
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=[64, 128, 256, 512]
    )
    parser.add_argument("--first-kernel-size", type=int, default=3)
    parser.add_argument("--identity-method", type=str, default="B")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--pct-start", type=float, default=0.01)
    parser.add_argument("--final-lr-ratio", type=float, default=0.07)
    parser.add_argument("--anneal-strategy", type=str, default="linear")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--train-dtype", type=str, default="fp32")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--save-path", type=str, required=True)

    args = parser.parse_args()

    model_args = ModelArgs(
        name=args.model,
        height=args.height,
        width=args.width,
        patch_size=args.patch_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_factor=args.mlp_factor,
        bias=args.bias,
        channels=args.channels,
        attn_drop=args.attn_drop,
        proj_drop=args.proj_drop,
        mlp_drop=args.mlp_drop,
        num_blocks=args.num_blocks,
        block_fn=args.block_fn,
        dimensions=args.dimensions,
        first_kernel_size=args.first_kernel_size,
        identity_method=args.identity_method,
    )

    training_args = TrainingArgs(
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        lr=args.lr,
        optimizer=args.optimizer,
        betas=args.betas,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        pct_start=args.pct_start,
        final_lr_ratio=args.final_lr_ratio,
        anneal_strategy=args.anneal_strategy,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        log_interval=args.log_interval,
        train_dtype=args.train_dtype,
        mixed_precision=args.mixed_precision,
        save_path=args.save_path,
    )

    return model_args, training_args
