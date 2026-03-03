import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score

# Disable W&B logging
os.environ["WANDB_MODE"] = "offline"


# =========================
# CONFIGURATION
# =========================
DATASET_PATH = "dataset"          # Path to dataset folder
RUNS_DIR = "runs"                 # Training output directory
MODEL_WEIGHTS = "yolo11s.pt"      # Pretrained weights
EPOCHS = 80
IMG_SIZE = 640
BATCH_SIZE = 16


def train_model():
    os.makedirs(RUNS_DIR, exist_ok=True)

    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=os.path.join(DATASET_PATH, "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=25,
        project=RUNS_DIR,
        name="YOLOv11_Optimized",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        amp=True,
        device=0 if torch.cuda.is_available() else "cpu",

        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        close_mosaic=15
    )

    return model


def evaluate_model(model):
    metrics = model.val(data=os.path.join(DATASET_PATH, "data.yaml"))

    print("\n================ VALIDATION RESULTS ================")
    print(f"mAP@50       : {metrics.box.map50:.4f}")
    print(f"mAP@50-95    : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("====================================================\n")

    return metrics


if __name__ == "__main__":
    import torch

    trained_model = train_model()
    evaluate_model(trained_model)
