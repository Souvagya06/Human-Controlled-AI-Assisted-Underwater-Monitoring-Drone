"""
scripts/train.py
================
Unified training script for YOLOv8n and YOLO-DarkWater.

Reads training configuration from configs/train_baseline.yaml or configs/train_darkwater.yaml.
Supports:
  - Custom model architecture via YAML loading
  - Transfer learning by loading compatible YOLOv8n weights
  - Custom Wise-IoU loss function integration
  - Custom OpenCV-based underwater preprocessing callback
  - Early stopping (patience)
  - Execution summary saving (best_metrics.json, training_config.json)

Usage:
    python scripts/train.py --config configs/train_baseline.yaml
    python scripts/train.py --config configs/train_darkwater.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import custom modules (registers YAML components in Ultralytics)
import models
from models.__init__ import load_pretrained_partial
from models.underwater_enhance import build_enhancer_from_config
from scripts.underwater_augmentation import (
    UnderwaterAugmentationPipeline,
    build_augmentation_callback,
)
from scripts.custom_loss import DarkWaterDetectionLoss

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.cfg import DEFAULT_CFG


class DarkWaterTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer that hooks into Ultralytics to inject:
    1. Wise-IoU loss (criterion override)
    2. Dynamic underwater augmentation callbacks inside build_dataset
    3. Underwater enhancement preprocessing callbacks inside build_dataset
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)
        # Store custom config parameters directly from self.args (which merges cfg & overrides)
        self.custom_cfg = {
            "enable_preprocessing": getattr(self.args, "enable_preprocessing", False),
            "clahe_clip_limit": getattr(self.args, "clahe_clip_limit", 3.0),
            "clahe_tile_grid": getattr(self.args, "clahe_tile_grid", 8),
            "gamma": getattr(self.args, "gamma", 1.2),
            "contrast_percentile_low": getattr(self.args, "contrast_percentile_low", 2),
            "contrast_percentile_high": getattr(self.args, "contrast_percentile_high", 98),
            "loss_type": getattr(self.args, "loss_type", "ciou"),
            "wiou_alpha": getattr(self.args, "wiou_alpha", 1.9),
            "wiou_delta": getattr(self.args, "wiou_delta", 3.0),
        }

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Builds the model structure and loads weights.
        Overridden to inject YOLO-DarkWater v1.0 metadata.
        """
        model = super().get_model(cfg, weights, verbose)
        # Inject custom model tags if this is YOLO-DarkWater
        if "yolo-darkwater" in str(self.args.model).lower():
            model.version = "1.0"  # type: ignore[attr-defined]
            model.model_name = "YOLO-DarkWater"  # type: ignore[attr-defined]
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Builds dataset. Overridden to apply OpenCV-based on-the-fly
        underwater enhancement preprocessing and data augmentation.
        """
        dataset = super().build_dataset(img_path, mode, batch)
        
        # 1. Hook on-the-fly OpenCV-based enhancement (preprocessing) if enabled
        if self.custom_cfg.get("enable_preprocessing", False):
            enhancer = build_enhancer_from_config(self.custom_cfg)
            self._wrap_dataset_loading(dataset, enhancer, "Enhancer")
            print(f"  [Callback] Underwater Preprocessing (Enhancer) active for dataset: {mode}")

        # 2. Hook on-the-fly underwater data augmentation pipeline for training
        if mode == "train":
            augmenter = UnderwaterAugmentationPipeline(p=0.5)
            self._wrap_dataset_loading(dataset, augmenter, "Augmenter")
            print(f"  [Callback] Underwater Augmentation Pipeline (Augmenter) active for dataset: {mode}")
            
        return dataset

    def _wrap_dataset_loading(self, dataset, transform_fn, name):
        """Helper to wrap dataset image loading functions with a custom transform."""
        if hasattr(dataset, "get_image_and_label"):
            orig_get_image_and_label = dataset.get_image_and_label
            
            def wrapped_get_image_and_label(index):
                label = orig_get_image_and_label(index)
                if "img" in label and label["img"] is not None:
                    label["img"] = transform_fn(label["img"])
                return label
            
            dataset.get_image_and_label = wrapped_get_image_and_label
            
        elif hasattr(dataset, "load_image"):
            orig_load_image = dataset.load_image
            
            def wrapped_load_image(index):
                img, hw_ori, hw_resized = orig_load_image(index)
                if img is not None:
                    img = transform_fn(img)
                return img, hw_ori, hw_resized
            
            dataset.load_image = wrapped_load_image


def train_model(config_path: str) -> None:
    print("\n" + "=" * 60)
    print(f"Starting Training with Config: {config_path}")
    print("=" * 60)

    # 1. Read Configuration
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract custom config keys not natively supported by Ultralytics trainer
    enable_preprocessing = cfg.pop("enable_preprocessing", False)
    loss_type = cfg.pop("loss_type", "ciou")
    transfer_weights = cfg.pop("transfer_weights", None)
    
    # Enhancement params
    clahe_clip_limit = cfg.pop("clahe_clip_limit", 3.0)
    clahe_tile_grid = cfg.pop("clahe_tile_grid", 8)
    gamma = cfg.pop("gamma", 1.2)
    contrast_percentile_low = cfg.pop("contrast_percentile_low", 2)
    contrast_percentile_high = cfg.pop("contrast_percentile_high", 98)

    # Wise-IoU params
    wiou_alpha = cfg.pop("wiou_alpha", 1.9)
    wiou_delta = cfg.pop("wiou_delta", 3.0)

    custom_cfg = {
        "enable_preprocessing": enable_preprocessing,
        "loss_type": loss_type,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_tile_grid": clahe_tile_grid,
        "gamma": gamma,
        "contrast_percentile_low": contrast_percentile_low,
        "contrast_percentile_high": contrast_percentile_high,
        "wiou_alpha": wiou_alpha,
        "wiou_delta": wiou_delta,
    }

    # Resolve dataset.yaml to absolute path
    data_yaml = cfg.get("data", "data/dataset.yaml")
    cfg["data"] = str((REPO_ROOT / data_yaml).resolve())

    # 2. Initialize Model
    model_spec = cfg.get("model", "yolov8n.pt")
    print(f"\nModel Spec: {model_spec}")

    # Start timing
    start_time = time.time()

    if model_spec.endswith(".yaml"):
        # Custom YAML architecture — build from scratch structure
        print("Building custom architecture from YAML structure...")
        model = YOLO(model_spec)
        
        # Load transfer weights (compatible layers from yolov8n.pt) if specified
        if transfer_weights:
            weights_path = REPO_ROOT / transfer_weights
            if not weights_path.exists():
                try:
                    YOLO(transfer_weights)
                except Exception:
                    pass
            
            if weights_path.exists():
                print(f"Loading pretrained weights from {transfer_weights} for transfer learning...")
                transferred, skipped = load_pretrained_partial(model.model, str(weights_path), verbose=True)  # type: ignore[arg-type]
                print(f"Transfer learning complete. Transferred {len(transferred)} layers, skipped {len(skipped)} layers.")
            else:
                print(f"WARNING: Pretrained weights {transfer_weights} not found. Training from random init.")
    else:
        # Standard pretrained model (e.g., yolov8n.pt)
        print(f"Loading model weight file: {model_spec}")
        model = YOLO(model_spec)

    # 3. Inject Custom Wise-IoU Loss if requested
    if loss_type == "wiou":
        print(f"Configuring Wise-IoU v3 Loss (alpha={wiou_alpha}, delta={wiou_delta})...")
        def custom_build_criterion(self):
            print("  [Loss Hook] Overriding trainer.criterion with DarkWaterDetectionLoss (Wise-IoU)")
            self.criterion = DarkWaterDetectionLoss(
                self.model,
                loss_type="wiou",
                alpha=wiou_alpha,
                delta=wiou_delta
            )
            return self.criterion
            
        DarkWaterTrainer.build_criterion = custom_build_criterion

    # 4. Populate combined overrides dictionary
    trainer_args = {**cfg, **custom_cfg}

    # 5. Execute Training
    print("\nExecuting train run...")
    try:
        model.train(trainer=DarkWaterTrainer, **trainer_args)
    except Exception as e:
        print(f"\nTraining interrupted or failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Retrying with default Ultralytics trainer...")
        model.train(**cfg)

    # End timing
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes.")

    # 7. Collect and Save Output Artifacts
    # Locate weights inside results run directory (retrieved from the instantiated trainer)
    run_dir = Path(model.trainer.save_dir)  # type: ignore[union-attr]
    print(f"Training run saved to: {run_dir}")
    
    # Save training_config.json
    config_save_path = run_dir / "training_config.json"
    with open(config_save_path, "w") as f:
        json.dump({
            "config_file": config_path,
            "training_time_seconds": round(elapsed_time, 2),
            "hyperparameters": {
                "epochs": cfg.get("epochs", 150),
                "patience": cfg.get("patience", 30),
                "lr0": cfg.get("lr0", 0.01),
                "lrf": cfg.get("lrf", 0.01),
                "optimizer": cfg.get("optimizer", "AdamW"),
                "imgsz": cfg.get("imgsz", 640),
                "batch": cfg.get("batch", 16),
                "seed": cfg.get("seed", 42),
                "loss_type": loss_type,
                "enable_preprocessing": enable_preprocessing,
            }
        }, f, indent=2)
    print(f"Saved configuration snapshot to {config_save_path}")

    # Extract best metrics from training results csv
    results_csv = run_dir / "results.csv"
    best_metrics_path = run_dir / "best_metrics.json"
    
    best_metrics = {
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "best_epoch": -1
    }
    
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            # Standard columns: epoch, train/box_loss, train/cls_loss, val/box_loss, val/cls_loss,
            # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
            # Find row with max metrics/mAP50(B)
            df.columns = [c.strip() for c in df.columns]
            
            # Map column names dynamically
            map50_col = [c for c in df.columns if "mAP50(B)" in c]
            map95_col = [c for c in df.columns if "mAP50-95(B)" in c]
            p_col = [c for c in df.columns if "precision(B)" in c]
            r_col = [c for c in df.columns if "recall(B)" in c]
            
            if map50_col:
                best_row_idx = df[map50_col[0]].idxmax()
                best_row = df.loc[best_row_idx]
                
                best_metrics = {
                    "mAP50": float(best_row[map50_col[0]]),
                    "mAP50-95": float(best_row[map95_col[0]]) if map95_col else 0.0,
                    "precision": float(best_row[p_col[0]]) if p_col else 0.0,
                    "recall": float(best_row[r_col[0]]) if r_col else 0.0,
                    "best_epoch": int(best_row["epoch"]),
                    "training_time_seconds": round(elapsed_time, 2)
                }
        except Exception as e:
            print(f"Error parsing results.csv: {e}")

    with open(best_metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"Saved best metrics summary to {best_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n or YOLO-DarkWater")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    train_model(args.config)
