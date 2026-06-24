"""
YOLOv8s Training Script for PCB Defect Detection
==================================================
PURPOSE:
    Fine-tunes YOLOv8s (pretrained on COCO) on the PCB defect dataset.
    Uses transfer learning - the pretrained backbone already understands
    visual features, so we only need to adapt it to PCB defects.

INPUT:
    data/splits/dataset.yaml  (created by prepare_dataset.py)

OUTPUT:
    runs/detect/pcb_defect_detector/
    ├── weights/best.pt         (best checkpoint)
    ├── weights/last.pt         (final checkpoint)
    ├── results.csv             (training metrics per epoch)
    ├── confusion_matrix.png    (class confusion analysis)
    ├── PR_curve.png            (Precision-Recall curve)
    └── val_batch*.jpg          (visualization of val predictions)

TRANSFER LEARNING EXPLAINED:
    YOLOv8s was pretrained on COCO (80 classes, 120k images).
    The backbone (feature extractor) learned general visual patterns.
    We "transfer" these features to PCB defects by:
    1. Starting with COCO weights (not random initialization)
    2. Training for fewer epochs (100 vs 300+ from scratch)
    3. Using a lower learning rate to preserve useful features

METRICS TO WATCH:
    - mAP50: Mean Average Precision at IoU=0.5 (main metric)
    - mAP50-95: mAP averaged across IoU thresholds 0.5 to 0.95
    - Precision: Of all predictions, what fraction is correct?
    - Recall: Of all true defects, what fraction was found?
    - Box Loss: How accurate are the bounding box coordinates?
    - Cls Loss: How well does it classify defect types?

USAGE:
    python detector/train.py
    python detector/train.py --config configs/training.yaml --resume
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional
import yaml
import torch
from loguru import logger
from ultralytics import YOLO
import mlflow

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Training metrics won't be tracked.")


def _refresh_dataset_yaml_path(dataset_yaml: Path) -> None:
    """
    Keep Ultralytics dataset.yaml usable after the project folder is renamed.

    prepare_dataset.py writes an absolute `path`. That is valid when generated,
    but it becomes stale if the project directory is moved or renamed.
    """
    with open(dataset_yaml, encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f) or {}

    current_root = dataset_yaml.parent.resolve()
    configured_root = Path(dataset_config.get("path", current_root))

    split_dirs = [
        current_root / dataset_config.get("train", "train/images"),
        current_root / dataset_config.get("val", "val/images"),
        current_root / dataset_config.get("test", "test/images"),
    ]

    if configured_root.resolve() == current_root:
        return

    if not all(path.exists() for path in split_dirs):
        return

    logger.warning(
        f"Dataset YAML path points to {configured_root}, "
        f"updating it to current split folder: {current_root}"
    )
    dataset_config["path"] = str(current_root)
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_config, f, default_flow_style=False, sort_keys=False)


def _disable_ultralytics_mlflow_callback() -> None:
    """Disable Ultralytics' built-in MLflow callback to avoid duplicate runs."""
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS["mlflow"] = False
        import ultralytics.utils.callbacks.mlflow as ultralytics_mlflow

        ultralytics_mlflow.mlflow = None
        ultralytics_mlflow.callbacks = {}
        logger.info("Ultralytics MLflow callback disabled; using project MLflow tracking")
    except Exception as exc:
        logger.warning(f"Could not disable Ultralytics MLflow callback: {exc}")


def _sanitize_mlflow_metric_name(name: str) -> str:
    """Convert Ultralytics metric names to MLflow-safe keys."""
    return name.replace("(", "_").replace(")", "")


def _get_best_weights_path(model: YOLO, config: dict) -> Path:
    """Return best.pt from the actual Ultralytics run directory."""
    trainer = getattr(model, "trainer", None)
    trainer_best = getattr(trainer, "best", None)
    if trainer_best:
        return Path(trainer_best)

    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir:
        return Path(trainer_save_dir) / "weights" / "best.pt"

    return Path(config["output"]["project"]) / config["output"]["name"] / "weights" / "best.pt"


def train_detector(
    config_path: str = "configs/training.yaml",
    resume: bool = False,
    resume_path: Optional[str] = None,
) -> dict:
    """
    Train YOLOv8s on PCB defect dataset.

    Args:
        config_path: Path to training configuration YAML.
        resume: Whether to resume from a previous checkpoint.
        resume_path: Path to checkpoint to resume from.

    Returns:
        Dictionary of training results/metrics.
    """
    # --- Load configuration ---
    logger.info(f"Loading training config from: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Validate prerequisites ---
    dataset_yaml = Path(config["dataset"]["path"])
    if not dataset_yaml.exists():
        logger.error(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Please run: python scripts/prepare_dataset.py"
        )
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    _refresh_dataset_yaml_path(dataset_yaml)

    # --- GPU Memory Check ---
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gpu_mem:.1f}GB")
        
        # Warn if batch size may be too large for VRAM
        if gpu_mem < 6 and config["training"]["batch_size"] > 8:
            logger.warning(
                f"Low VRAM ({gpu_mem:.1f}GB) with batch_size={config['training']['batch_size']}. "
                "Consider reducing batch_size or enabling gradient checkpointing."
            )
    else:
        logger.warning("No GPU detected. Training on CPU will be very slow!")

    _disable_ultralytics_mlflow_callback()

    # --- Initialize MLflow tracking ---
    mlflow_enabled = config.get("mlflow", {}).get("enabled", False) and MLFLOW_AVAILABLE
    if mlflow_enabled:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=config["output"]["name"])
        
        # Log hyperparameters
        mlflow.log_params({
            "model": config["model"]["architecture"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "image_size": config["training"]["image_size"],
            "lr0": config["optimizer"]["lr0"],
            "pretrained": config["model"]["pretrained"],
        })
        logger.info("MLflow tracking started")

    # --- Load YOLOv8s Model ---
    if resume and resume_path:
        # Resume from checkpoint
        logger.info(f"Resuming training from: {resume_path}")
        model = YOLO(resume_path)
    elif resume:
        # Auto-detect last checkpoint
        last_pt = Path(config["output"]["project"]) / config["output"]["name"] / "weights/last.pt"
        if last_pt.exists():
            logger.info(f"Resuming from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            logger.warning("No checkpoint found to resume. Starting fresh.")
            model = YOLO(config["model"]["pretrained_weights"])
    else:
        # Start from COCO pretrained weights
        # YOLOv8s.pt is automatically downloaded by ultralytics if not present
        logger.info(f"Loading pretrained: {config['model']['pretrained_weights']}")
        model = YOLO(config["model"]["pretrained_weights"])

    logger.info("Starting YOLOv8s training...")
    logger.info(
        f"Epochs: {config['training']['epochs']} | "
        f"Batch: {config['training']['batch_size']} | "
        f"ImgSize: {config['training']['image_size']}"
    )

    # --- Training Arguments ---
    # All supported by Ultralytics YOLO.train()
    train_args = {
        # Dataset
        "data": str(dataset_yaml),
        
        # Training scale
        "epochs": config["training"]["epochs"],
        "batch": config["training"]["batch_size"],
        "imgsz": config["training"]["image_size"],
        "workers": config["training"]["workers"],
        "device": config["training"]["device"],
        "seed": config["training"]["seed"],
        
        # Optimizer
        "optimizer": config["optimizer"]["name"],
        "lr0": config["optimizer"]["lr0"],
        "lrf": config["optimizer"]["lrf"],
        "momentum": config["optimizer"]["momentum"],
        "weight_decay": config["optimizer"]["weight_decay"],
        "warmup_epochs": config["optimizer"]["warmup_epochs"],
        "warmup_momentum": config["optimizer"]["warmup_momentum"],
        "warmup_bias_lr": config["optimizer"]["warmup_bias_lr"],
        
        # Augmentation
        "hsv_h": config["augmentation"]["hsv_h"],
        "hsv_s": config["augmentation"]["hsv_s"],
        "hsv_v": config["augmentation"]["hsv_v"],
        "degrees": config["augmentation"]["degrees"],
        "translate": config["augmentation"]["translate"],
        "scale": config["augmentation"]["scale"],
        "shear": config["augmentation"]["shear"],
        "perspective": config["augmentation"]["perspective"],
        "flipud": config["augmentation"]["flipud"],
        "fliplr": config["augmentation"]["fliplr"],
        "mosaic": config["augmentation"]["mosaic"],
        "mixup": config["augmentation"]["mixup"],
        
        # Loss weights
        "box": config["loss"]["box"],
        "cls": config["loss"]["cls"],
        "dfl": config["loss"]["dfl"],
        
        # Output
        "project": config["output"]["project"],
        "name": config["output"]["name"],
        "save": config["output"]["save"],
        "save_period": config["output"]["save_period"],
        "plots": config["output"]["plots"],
        "verbose": config["output"]["verbose"],
        
        # Resume
        "resume": resume,
    }

    # --- Run Training ---
    results = model.train(**train_args)

    # --- Post-training: Copy best weights to models/ ---
    best_weights_src = _get_best_weights_path(model, config)
    best_weights_dst = Path("models/detector/best.pt")
    best_weights_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, best_weights_dst)
        logger.info(f"Best weights copied from {best_weights_src} to: {best_weights_dst}")
    else:
        logger.warning(f"Best weights not found at: {best_weights_src}")

    # --- Extract and log metrics ---
    metrics_dict = {}
    if hasattr(results, "results_dict"):
        metrics_dict = results.results_dict
        logger.info("Training Metrics:")
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    if mlflow_enabled:
        safe_metrics = {
            _sanitize_mlflow_metric_name(k): v
            for k, v in metrics_dict.items()
            if isinstance(v, (int, float))
        }
        try:
            if safe_metrics:
                mlflow.log_metrics(safe_metrics)
            if best_weights_src.exists():
                mlflow.log_artifact(str(best_weights_src))
            logger.info("MLflow run completed")
        except Exception as exc:
            logger.warning(f"MLflow logging failed after training completed: {exc}")
        finally:
            mlflow.end_run()

    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Best weights: {best_weights_dst}")
    logger.info("=" * 50)

    return metrics_dict


def validate_model(
    weights_path: str = "models/detector/best.pt",
    config_path: str = "configs/training.yaml",
) -> dict:
    """
    Run validation on the test set to get final metrics.

    Args:
        weights_path: Path to model weights.
        config_path: Path to config (for dataset path).

    Returns:
        Validation metrics dict.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = YOLO(weights_path)
    
    # Validate on test set
    results = model.val(
        data=config["dataset"]["path"],
        split="test",
        conf=config["evaluation"]["conf_threshold"],
        iou=config["evaluation"]["iou_threshold"],
        max_det=config["evaluation"]["max_det"],
        plots=True,
        verbose=True,
    )

    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        
    logger.info("Validation Results on Test Set:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    from typing import Optional
    
    parser = argparse.ArgumentParser(description="Train YOLOv8s PCB Defect Detector")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--resume-path", default=None, help="Specific checkpoint to resume")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--weights", default="models/detector/best.pt")
    args = parser.parse_args()

    if args.validate_only:
        validate_model(args.weights, args.config)
    else:
        train_detector(args.config, args.resume, args.resume_path)
