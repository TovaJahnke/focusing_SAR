"""
Training a YOLO ship detector with scene-aware k-fold cross-validation.

The default split strategy groups chips by source scene prefix (for example
"P0001" in "P0001_1200_2000_3600_4400.jpg") so related chips do not leak into
both train and held-out folds.

Examples:
  python train_yolo_kfold.py --k 5 --epochs 100 --batch 8 --device cuda
  python train_yolo_kfold.py --k 5 --dry-run
  python train_yolo_kfold.py --k 5 --fold 0 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scene-aware k-fold cross-validation training for YOLO"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per fold")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Training device (cuda/cpu)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt", help="Model or weights path"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience in epochs",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to HRSID_JPG (defaults to yolo/HRSID_JPG next to this script)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Where Ultralytics run folders and summaries are written",
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=None,
        help="Where generated fold datasets are written",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="yolov8s_hrsid_kfold",
        help="Prefix for Ultralytics run names",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train only one fold index instead of all folds",
    )
    parser.add_argument(
        "--split-mode",
        choices=("scene", "image"),
        default="scene",
        help="Keep source scenes together or split individual chips randomly",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy images/labels instead of creating symlinks",
    )
    parser.add_argument(
        "--keep-fold-dirs",
        action="store_true",
        help="Keep existing fold directories instead of recreating them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create manifests and fold datasets without starting training",
    )
    return parser.parse_args()


def get_image_files(jpeg_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in jpeg_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_scene_key(image_path: Path) -> str:
    stem = image_path.stem
    return stem.split("_", 1)[0]


def build_folds(
    image_files: list[Path], k: int, seed: int, split_mode: str
) -> list[dict]:
    if k < 2:
        raise ValueError("--k must be at least 2")
    if not image_files:
        raise ValueError("No images found to split")

    rng = np.random.default_rng(seed)

    if split_mode == "image":
        shuffled = list(image_files)
        rng.shuffle(shuffled)
        fold_parts = [list(part) for part in np.array_split(shuffled, k)]
        folds = []
        for fold_id, val_images in enumerate(fold_parts):
            train_images = [
                image for other_id, part in enumerate(fold_parts) if other_id != fold_id
                for image in part
            ]
            folds.append(
                {
                    "fold": fold_id,
                    "train_images": sorted(path.name for path in train_images),
                    "val_images": sorted(path.name for path in val_images),
                    "train_groups": [],
                    "val_groups": [],
                }
            )
        return folds

    grouped_images: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_files:
        grouped_images[get_scene_key(image_path)].append(image_path)

    group_items = list(grouped_images.items())
    if k > len(group_items):
        raise ValueError(
            f"--k={k} is larger than the number of source scenes ({len(group_items)})"
        )

    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_groups: list[list[str]] = [[] for _ in range(k)]
    fold_images: list[list[Path]] = [[] for _ in range(k)]
    fold_sizes = [0] * k

    for group_key, group_paths in group_items:
        target_fold = min(range(k), key=lambda idx: (fold_sizes[idx], idx))
        fold_groups[target_fold].append(group_key)
        fold_images[target_fold].extend(sorted(group_paths))
        fold_sizes[target_fold] += len(group_paths)

    folds = []
    for fold_id, val_images in enumerate(fold_images):
        train_images = [
            image for other_id, part in enumerate(fold_images) if other_id != fold_id
            for image in part
        ]
        train_groups = [
            group for other_id, groups in enumerate(fold_groups) if other_id != fold_id
            for group in groups
        ]
        folds.append(
            {
                "fold": fold_id,
                "train_images": sorted(path.name for path in train_images),
                "val_images": sorted(path.name for path in val_images),
                "train_groups": sorted(train_groups),
                "val_groups": sorted(fold_groups[fold_id]),
            }
        )

    return folds


def safe_remove_dir(path: Path, fold_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = fold_root.resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to delete directory outside fold root: {path}")
    shutil.rmtree(path)


def materialize_sample(
    image_name: str,
    source_dir: Path,
    image_dest_dir: Path,
    label_dest_dir: Path,
    copy_files: bool,
) -> None:
    src_image = source_dir / image_name
    dst_image = image_dest_dir / image_name
    label_name = src_image.with_suffix(".txt").name
    src_label = source_dir / label_name
    dst_label = label_dest_dir / label_name

    if copy_files:
        shutil.copy2(src_image, dst_image)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")
        return

    dst_image.symlink_to(src_image.resolve())
    if src_label.exists():
        dst_label.symlink_to(src_label.resolve())
    else:
        dst_label.write_text("", encoding="utf-8")


def prepare_fold_dataset(
    fold_data: dict,
    jpeg_dir: Path,
    fold_root: Path,
    copy_files: bool,
    keep_fold_dirs: bool,
) -> Path:
    fold_dir = fold_root / f"fold_{fold_data['fold']}"
    if fold_dir.exists() and not keep_fold_dirs:
        safe_remove_dir(fold_dir, fold_root)

    for relative_dir in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (fold_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    if keep_fold_dirs and any((fold_dir / "images" / "train").iterdir()):
        return write_data_yaml(fold_dir)

    for split_name, split_dir in (("train_images", "train"), ("val_images", "val")):
        for image_name in fold_data[split_name]:
            materialize_sample(
                image_name=image_name,
                source_dir=jpeg_dir,
                image_dest_dir=fold_dir / "images" / split_dir,
                label_dest_dir=fold_dir / "labels" / split_dir,
                copy_files=copy_files,
            )

    return write_data_yaml(fold_dir)


def write_data_yaml(fold_dir: Path) -> Path:
    data_yaml = fold_dir / "data.yaml"
    yaml_content = "\n".join(
        [
            f"path: {fold_dir.resolve()}",
            "",
            "train: images/train",
            "val: images/val",
            "test: images/val",
            "",
            "names:",
            "  0: ship",
            "",
        ]
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")
    return data_yaml


def collect_numeric_metrics(raw_metrics: dict) -> dict[str, float]:
    numeric_metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, (int, float, np.floating)):
            numeric_metrics[key] = float(value)
    return numeric_metrics


def train_fold(
    fold_data: dict,
    data_yaml: Path,
    args: argparse.Namespace,
) -> dict:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics training dependencies are missing. Install with:\n"
            "  pip install ultralytics torch"
        ) from exc

    device = args.device
    batch = args.batch
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA is not available; falling back to CPU.")
        device = "cpu"
        if batch > 4:
            batch = 4
            print(f"[warn] Reduced batch size to {batch} for CPU training.")

    fold_id = fold_data["fold"]
    print(f"\n{'=' * 70}")
    print(
        f"Training fold {fold_id}: "
        f"{len(fold_data['train_images'])} train / {len(fold_data['val_images'])} held-out"
    )
    print(f"{'=' * 70}\n")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        workers=args.workers,
        device=device,
        project=str(args.project),
        name=f"{args.name_prefix}_fold_{fold_id}",
        patience=args.patience,
        fraction=1.0,
        save=True,
        exist_ok=True,
        verbose=True,
    )

    metrics = {
        "fold": fold_id,
        "train_images": len(fold_data["train_images"]),
        "val_images": len(fold_data["val_images"]),
        "train_groups": len(fold_data["train_groups"]),
        "val_groups": len(fold_data["val_groups"]),
        "data_yaml": str(data_yaml.resolve()),
    }

    raw_metrics = getattr(results, "results_dict", {}) or {}
    metrics.update(collect_numeric_metrics(raw_metrics))

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        metrics["save_dir"] = str(Path(save_dir).resolve())

    return metrics


def summarize_results(fold_results: list[dict]) -> dict[str, dict[str, float]]:
    ignored_keys = {
        "fold",
        "train_images",
        "val_images",
        "train_groups",
        "val_groups",
        "data_yaml",
        "save_dir",
    }
    metric_names = sorted(
        {
            key
            for result in fold_results
            for key, value in result.items()
            if key not in ignored_keys and isinstance(value, (int, float))
        }
    )

    aggregates: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(result[metric_name]) for result in fold_results if metric_name in result]
        if not values:
            continue
        aggregates[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return aggregates


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
    dataset_root = args.dataset_root or (here / "HRSID_JPG")
    project_dir = args.project or (here / "runs")
    fold_root = args.fold_root or (dataset_root / "kfold_data")
    jpeg_dir = dataset_root / "JPEGImages"

    if not jpeg_dir.exists():
        raise FileNotFoundError(f"JPEGImages directory not found: {jpeg_dir}")

    image_files = get_image_files(jpeg_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {jpeg_dir}")

    scene_count = len({get_scene_key(path) for path in image_files})
    print(f"[info] Dataset root: {dataset_root}")
    print(f"[info] Found {len(image_files)} images across {scene_count} source scenes")
    print(f"[info] Split mode: {args.split_mode}")

    folds = build_folds(
        image_files=image_files,
        k=args.k,
        seed=args.seed,
        split_mode=args.split_mode,
    )

    split_manifest = {
        "k": args.k,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "dataset_root": str(dataset_root.resolve()),
        "total_images": len(image_files),
        "total_scenes": scene_count,
        "folds": [
            {
                "fold": fold["fold"],
                "train_images": len(fold["train_images"]),
                "val_images": len(fold["val_images"]),
                "train_groups": len(fold["train_groups"]),
                "val_groups": len(fold["val_groups"]),
                "val_group_names": fold["val_groups"],
            }
            for fold in folds
        ],
    }
    save_json(project_dir / f"{args.name_prefix}_splits.json", split_manifest)

    selected_fold_ids = range(args.k) if args.fold is None else [args.fold]
    if any(fold_id < 0 or fold_id >= args.k for fold_id in selected_fold_ids):
        raise ValueError(f"--fold must be in the range [0, {args.k - 1}]")

    fold_results = []
    for fold_id in selected_fold_ids:
        fold_data = folds[fold_id]
        data_yaml = prepare_fold_dataset(
            fold_data=fold_data,
            jpeg_dir=jpeg_dir,
            fold_root=fold_root,
            copy_files=args.copy_files,
            keep_fold_dirs=args.keep_fold_dirs,
        )

        print(
            f"[info] Fold {fold_id} prepared at {data_yaml.parent} "
            f"({len(fold_data['train_images'])} train / {len(fold_data['val_images'])} held-out)"
        )

        if args.dry_run:
            fold_results.append(
                {
                    "fold": fold_id,
                    "train_images": len(fold_data["train_images"]),
                    "val_images": len(fold_data["val_images"]),
                    "train_groups": len(fold_data["train_groups"]),
                    "val_groups": len(fold_data["val_groups"]),
                    "data_yaml": str(data_yaml.resolve()),
                }
            )
            continue

        metrics = train_fold(fold_data=fold_data, data_yaml=data_yaml, args=args)
        fold_results.append(metrics)
        save_json(project_dir / f"{args.name_prefix}_fold_{fold_id}_results.json", metrics)

    summary = {
        "k": args.k,
        "epochs_per_fold": args.epochs,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "dataset_root": str(dataset_root.resolve()),
        "project_dir": str(project_dir.resolve()),
        "fold_root": str(fold_root.resolve()),
        "total_images": len(image_files),
        "total_scenes": scene_count,
        "dry_run": args.dry_run,
        "results": fold_results,
    }
    if not args.dry_run:
        summary["aggregates"] = summarize_results(fold_results)

    summary_path = project_dir / f"{args.name_prefix}_summary.json"
    save_json(summary_path, summary)
    print(f"\n[info] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
