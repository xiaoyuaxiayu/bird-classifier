"""Train and predict the 70-class bird image classifier.

Examples:
    python bird_classifier.py train --student-id 09118000 --student-name 李飞飞
    python bird_classifier.py predict --checkpoint outputs\best.pt ^
        --student-id 09118000 --student-name 李飞飞

The default paths match the supplied experiment dataset.  The script uses the
bounding boxes in trainDataSet_mixed.txt when constructing training samples.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "实验一数据集" / "实验一数据集"
TRAIN_ROOT = DATA_ROOT / "bird"
TEST_ROOT = DATA_ROOT / "test_DataSet"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    box: tuple[int, int, int, int] | None = None


def read_classes(path: Path) -> dict[str, int]:
    classes: dict[str, int] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<id> <name>'")
            class_id, class_name = fields
            classes[class_name] = int(class_id) - 1
    if len(classes) != 70:
        raise ValueError(f"Expected 70 classes, found {len(classes)} in {path}")
    return classes


def read_training_samples(annotation_path: Path, image_root: Path,
                          class_to_index: dict[str, int]) -> list[Sample]:
    samples: list[Sample] = []
    with annotation_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.strip().split()
            if len(fields) != 6:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected 6 fields, found {len(fields)}"
                )
            _, relative_path, x1, y1, x2, y2 = fields
            class_name, image_name = relative_path.split("/", maxsplit=1)
            if class_name not in class_to_index:
                raise ValueError(f"Unknown class '{class_name}' at line {line_number}")
            image_path = image_root / class_name / image_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            samples.append(
                Sample(image_path, class_to_index[class_name],
                       (int(x1), int(y1), int(x2), int(y2)))
            )
    if not samples:
        raise ValueError(f"No samples found in {annotation_path}")
    return samples


def crop_box(image: Image.Image, box: tuple[int, int, int, int] | None) -> Image.Image:
    if box is None:
        return image
    width, height = image.size
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


class BirdDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = crop_box(source.convert("RGB"), sample.box)
        return self.transform(image), sample.label


class TestDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        with Image.open(self.image_paths[index]) as source:
            image = source.convert("RGB")
        return self.transform(image), index + 1


def make_model(
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    fine_tune_last_block: bool = False,
) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if fine_tune_last_block:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def transforms_for_training(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    weights_mean = (0.485, 0.456, 0.406)
    weights_std = (0.229, 0.224, 0.225)
    train = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(weights_mean, weights_std),
    ])
    evaluation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(weights_mean, weights_std),
    ])
    return train, evaluation


def split_samples(samples: list[Sample], validation_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(len(samples) * validation_ratio))
    validation_indices = set(indices[:validation_size])
    train = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    validation = [sample for index, sample in enumerate(samples) if index in validation_indices]
    return train, validation


def accuracy(model: nn.Module, loader: DataLoader[tuple[Tensor, int]], device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def checkpoint_data(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_accuracy: float,
    args: argparse.Namespace,
    class_to_index: dict[str, int],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "num_classes": 70,
        "image_size": args.image_size,
        "class_to_index": class_to_index,
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "fine_tune_last_block": args.fine_tune_last_block,
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    class_to_index = read_classes(args.classes)
    samples = read_training_samples(args.annotations, args.image_root, class_to_index)
    if args.train_limit:
        samples = samples[:args.train_limit]
    train_samples, validation_samples = split_samples(samples, args.validation_ratio, args.seed)
    train_transform, evaluation_transform = transforms_for_training(args.image_size)
    train_loader = DataLoader(
        BirdDataset(train_samples, train_transform), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=False,
    )
    validation_loader = DataLoader(
        BirdDataset(validation_samples, evaluation_transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = make_model(
        70, args.pretrained, args.freeze_backbone, args.fine_tune_last_block
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.last_output.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_accuracy = -1.0
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model"])
        if "optimizer" in resume_checkpoint:
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer"])
            except ValueError:
                print(
                    "optimizer state is incompatible with the current trainable "
                    "layers; starting a fresh optimizer",
                    flush=True,
                )
        start_epoch = int(resume_checkpoint.get("epoch", args.resume_epoch)) + 1
        if "best_accuracy" in resume_checkpoint:
            best_accuracy = float(resume_checkpoint["best_accuracy"])
        else:
            best_accuracy = accuracy(model, validation_loader, device)
            print(
                f"legacy checkpoint validation accuracy={best_accuracy:.4f}; "
                "preserving it as the initial best",
                flush=True,
            )
        print(f"resuming from {args.resume}: starting epoch {start_epoch}/{args.epochs}")
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}; target is only {args.epochs}"
        )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            labels = labels.to(device)
            logits = model(images.to(device))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.numel()
        val_accuracy = accuracy(model, validation_loader, device)
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total if train_total else 0.0
        print(
            f"epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} val_accuracy={val_accuracy:.4f}",
            flush=True,
        )
        torch.save(
            checkpoint_data(model, optimizer, epoch, best_accuracy, args, class_to_index),
            args.last_output,
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                checkpoint_data(model, optimizer, epoch, best_accuracy, args, class_to_index),
                args.output,
            )
    print(f"saved checkpoint: {args.output} (best_val_accuracy={best_accuracy:.4f})")
    print(f"saved resumable checkpoint: {args.last_output}")


def predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(
        checkpoint["num_classes"], bool(checkpoint["pretrained"]), freeze_backbone=False
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    _, evaluation_transform = transforms_for_training(checkpoint["image_size"])
    image_paths = sorted(
        (path for path in args.test_root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: int(path.stem),
    )
    if len(image_paths) != 1666:
        raise ValueError(f"Expected 1666 test images, found {len(image_paths)}")
    loader = DataLoader(
        TestDataset(image_paths, evaluation_transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    output_path = args.output / f"{args.student_id}{args.student_name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=" ", lineterminator="\n")
        writer.writerow([f"{args.student_id}{args.student_name}"])
        with torch.inference_mode():
            for images, image_ids in loader:
                predicted = model(images.to(device)).argmax(1).cpu() + 1
                for image_id, class_id in zip(image_ids.tolist(), predicted.tolist()):
                    writer.writerow([image_id, class_id])
    print(f"saved predictions: {output_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="70-class bird classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--classes", type=Path, default=TRAIN_ROOT / "classes.txt")
    train_parser.add_argument("--annotations", type=Path, default=TRAIN_ROOT / "trainDataSet_mixed.txt")
    train_parser.add_argument("--image-root", type=Path, default=TRAIN_ROOT / "images")
    train_parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "best.pt")
    train_parser.add_argument("--last-output", type=Path, default=ROOT / "outputs" / "last.pt")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--resume-epoch", type=int, default=1,
        help="epoch represented by a legacy checkpoint without epoch metadata",
    )
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--validation-ratio", type=float, default=0.15)
    train_parser.add_argument("--train-limit", type=int, default=0)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--fine-tune-last-block", action="store_true")
    train_parser.set_defaults(function=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "best.pt")
    predict_parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    predict_parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    predict_parser.add_argument("--student-id", required=True)
    predict_parser.add_argument("--student-name", required=True)
    predict_parser.add_argument("--batch-size", type=int, default=32)
    predict_parser.add_argument("--workers", type=int, default=0)
    predict_parser.add_argument("--cpu", action="store_true")
    predict_parser.set_defaults(function=predict)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
