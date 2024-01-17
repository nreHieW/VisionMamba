import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import yaml

from torch.utils.data import DataLoader

from models import build_model
from args import parse_args, ModelArgs, TrainingArgs


def train(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    train_args: TrainingArgs,
):
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
    }[train_args.train_dtype]
    model = model.to(train_args.device, dtype=torch_dtype)
    model.train()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num params: {num_params:,}")

    num_batch = len(trainloader)
    print("Total batches:", num_batch)
    total_steps = num_batch * train_args.num_epochs

    ce_loss = nn.CrossEntropyLoss()

    # https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py#L484

    if train_args.optimizer == "adamw":  # note that adamw with fp16 is not fully stable
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.lr,
            betas=train_args.betas,
            weight_decay=train_args.weight_decay,
            eps=1e-6,
        )
    elif train_args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_args.lr,
            weight_decay=train_args.weight_decay,
            foreach=True,
            nesterov=True,
            momentum=0.85,
        )
    else:
        raise NotImplementedError(f"Optimizer {train_args.optimizer} not implemented")

    initial_div_factor = 1e16
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_args.lr,
        pct_start=train_args.pct_start,
        div_factor=initial_div_factor,
        final_div_factor=1.0 / (initial_div_factor * train_args.final_lr_ratio),
        total_steps=total_steps,
        anneal_strategy=train_args.anneal_strategy,
        cycle_momentum=False,
    )

    best_acc = 0.0
    for epoch in range(train_args.num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader, 1):
            inputs, targets = inputs.to(
                device=train_args.device, dtype=torch_dtype
            ), targets.to("cuda")
            optimizer.zero_grad()

            if train_args.mixed_precision:
                with torch.autocast(device_type=train_args.device, dtype=torch_dtype):
                    y = model(inputs)
                    loss = ce_loss(y, targets)
            else:
                y = model(inputs)
                loss = ce_loss(y.float(), targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % train_args.log_interval == 0:
                _, preds = y.max(1)
                acc = preds.eq(targets).sum().item() / targets.size(0)
                train_log(
                    batch_idx,
                    loss.item(),
                    acc,
                    scheduler.get_last_lr()[0],
                    epoch,
                    train_args.wandb,
                )

        with torch.no_grad():
            eval_acc = []
            eval_loss = []
            for eval_batch, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(
                    device=train_args.device, dtype=torch_dtype
                ), targets.to("cuda")
                y = model(inputs)
                loss = ce_loss(y.float(), targets)
                eval_loss.append(loss.item())

                _, preds = y.max(1)
                acc = preds.eq(targets).sum().item() / targets.size(0)
                eval_acc.append(acc)

        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_acc = sum(eval_acc) / len(eval_acc)

        eval_log(eval_loss, eval_acc, epoch, train_args.wandb)

        if eval_acc > best_acc:
            best_acc = eval_acc
            print(f"Saving model with accuracy {best_acc}")
            torch.save(model.state_dict(), f"{train_args.save_path}/model.pt")

    return model, best_acc


def train_log(batch_idx, loss, acc, lr, epoch, log_wandb: bool):
    print(f"Batch {batch_idx}: Loss: {loss} | Accuracy {acc}")
    if log_wandb:
        wandb.log(
            {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss,
                "accuracy": acc,
                "lr": lr,
            }
        )


def eval_log(eval_loss, eval_acc, epoch, log_wandb: bool):
    print(
        f"======= Epoch {epoch}: Eval Loss {eval_loss}, Eval Accuracy {eval_acc} ========"
    )
    if log_wandb:
        wandb.log({"epoch": epoch, "eval_loss": eval_loss, "eval_accuracy": eval_acc})


def main():
    model_args, training_args = parse_args()

    # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=training_args.batch_size, shuffle=False, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=training_args.eval_batch_size, shuffle=False, num_workers=2
    )

    model = build_model(model_args)

    if training_args.wandb:
        if training_args.run_name == "":
            raise ValueError("Run name must be specified")
        if training_args.wandb_project == "":
            raise ValueError("Wandb project must be specified")

        cfg = {}
        for k, v in model_args.get_relevant_args().items():
            cfg[f"model_{k}"] = v

        for k, v in vars(training_args).items():
            cfg[f"training_{k}"] = v

        run = wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name,
            config=cfg,
            notes=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    print(model_args)
    print(training_args)

    if not os.path.exists(training_args.save_path):
        os.makedirs(training_args.save_path)

    model, best_acc = train(model, trainloader, testloader, training_args)

    print(f"Best accuracy: {best_acc}")

    with open(f"{training_args.save_path}/model_args.yaml", "w") as f:
        yaml.dump(model_args.__dict__, f)

    with open(f"{training_args.save_path}/training_args.yaml", "w") as f:
        yaml.dump(training_args.__dict__, f)

    if training_args.wandb:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(f"{training_args.save_path}/model.pt")
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
