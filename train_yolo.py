
import os
import shutil
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — edit as needed
# ─────────────────────────────────────────────────────────────
DATASET_YAML  = "dataset.yaml"          # created by prepare_dataset.py
MODEL_SIZE    = "yolov8m.pt"            # nano=yolov8n, small=yolov8s, medium=yolov8m
                                        # medium used here — 10 classes need more capacity
                                        # downgrade to yolov8s.pt if GPU runs out of memory
EPOCHS        = 30                     # increase to 200 for better results if time allows
IMGSZ         = 320                     # input resolution (must be multiple of 32)
BATCH_SIZE    = 16                      # reduce to 8 if GPU runs out of memory
WORKERS       = 4                       # dataloader workers
PATIENCE      = 20                      # early-stopping patience (epochs without improvement)
PROJECT_DIR   = "models"               # parent folder for all training runs
RUN_NAME      = "wood_defect_yolo"     # subfolder name — best.pt ends up here
CONF_THRESH   = 0.25                    # confidence threshold for validation
IOU_THRESH    = 0.45                    # NMS IoU threshold
# ─────────────────────────────────────────────────────────────


def check_environment():
    print("\n[ENV] Checking environment...")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  ✓  ultralytics {ultralytics.__version__}")
    except ImportError:
        raise ImportError(
            "ultralytics not installed.\n"
            "Run:  pip install ultralytics"
        )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓  CUDA available  —  {gpu_name}  ({vram:.1f} GB VRAM)")
        device = 0
    else:
        print("  ⚠  No GPU detected — training on CPU (will be slow)")
        print("     Consider reducing EPOCHS to 30 and IMAGE SIZE to 320 for CPU runs")
        device = "cpu"

    if not os.path.isfile(DATASET_YAML):
        raise FileNotFoundError(
            f"dataset.yaml not found at '{DATASET_YAML}'.\n"
            f"Run prepare_dataset.py first."
        )

    print(f"  ✓  dataset.yaml found: {Path(DATASET_YAML).resolve()}")
    return device


def train(device):
    from ultralytics import YOLO
    import yaml

    # Load class count from yaml for reporting
    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)
    nc = cfg.get("nc", "?")
    names = cfg.get("names", [])
    print(f"\n[DATASET] Classes ({nc}): {names}")

    print(f"\n[TRAIN] Starting YOLOv8 training...")
    print(f"  Model      : {MODEL_SIZE}")
    print(f"  Epochs     : {EPOCHS}  (early stop after {PATIENCE} no-improvement epochs)")
    print(f"  Image size : {IMGSZ}px")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Device     : {'GPU' if device == 0 else 'CPU'}")
    print(f"  Output     : {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print()

    model = YOLO(MODEL_SIZE)  # downloads pretrained weights on first run

    results = model.train(
        data        = DATASET_YAML,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH_SIZE,
        workers     = WORKERS,
        patience    = PATIENCE,
        project     = PROJECT_DIR,
        name        = RUN_NAME,
        device      = device,
        exist_ok    = True,          # overwrite previous run if name matches
        save        = True,
        save_period = 10,            # checkpoint every 10 epochs
        val         = True,
        plots       = True,          # save training curves to project dir
        conf        = CONF_THRESH,
        iou         = IOU_THRESH,
        # Augmentation (helps with small/limited datasets)
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.1,
        fliplr      = 0.5,
        mosaic      = 1.0,
        mixup       = 0.1,
    )
    return results


def copy_best_weights():
    """Copy best.pt to models/ root so vision_pipeline.py finds it immediately."""
    best_src = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    best_dst = Path(PROJECT_DIR) / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n  ✓  best.pt copied → {best_dst.resolve()}")
    else:
        print(f"\n  ⚠  best.pt not found at {best_src} — check training output")


def evaluate(device):
    """Run validation on test split and print metrics."""
    from ultralytics import YOLO

    best_pt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not best_pt.exists():
        print("  Skipping evaluation — best.pt not found")
        return

    print("\n[EVAL] Running evaluation on TEST split...")
    model = YOLO(str(best_pt))
    metrics = model.val(
        data    = DATASET_YAML,
        split   = "test",
        device  = device,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        verbose = True,
    )

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS (Test Split)")
    print("=" * 55)
    try:
        print(f"  mAP@50        : {metrics.box.map50:.4f}")
        print(f"  mAP@50-95     : {metrics.box.map:.4f}")
        print(f"  Precision     : {metrics.box.mp:.4f}")
        print(f"  Recall        : {metrics.box.mr:.4f}")
    except Exception:
        print("  (Could not parse detailed metrics — check ultralytics output above)")
    print("=" * 55)


def main():
    print("=" * 55)
    print("  YOLOV8 TRAINING — WOOD DEFECT DETECTION")
    print("=" * 55)

    device = check_environment()
    results = train(device)
    copy_best_weights()
    evaluate(device)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print(f"  Weights : models/{RUN_NAME}/weights/best.pt")
    print(f"  Plots   : models/{RUN_NAME}/")
    print("\n  Next step:  python main.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
