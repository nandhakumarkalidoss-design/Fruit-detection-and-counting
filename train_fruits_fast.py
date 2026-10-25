
import yaml
from pathlib import Path
from ultralytics import YOLO

# ==================== SETTINGS ====================
DATASET_DIR = Path(r"C:\Users\Lenovo\fruit\Fruits by YOLO.v1i.yolov8")
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_NAME = "yolov8n.pt"
EPOCHS = 30
IMAGE_SIZE = 416
BATCH_SIZE = 8
WORKERS = 2
PROJECT_NAME = str(DATASET_DIR / "runs" / "fruit_training")
RUN_NAME = "fruit_fast_30"
# ==================================================

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    # Configure dataset paths.
    with open(DATA_YAML, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["path"] = str(DATASET_DIR)
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    # Verify class configuration.
    names = config.get("names")
    if not names or len(names) != 9:
        raise ValueError(
            f"Expected 9 fruit classes in data.yaml, found: {names}"
        )

    with open(DATA_YAML, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # Verify all dataset image folders.
    for folder in ["train/images", "valid/images", "test/images"]:
        image_folder = DATASET_DIR / folder
        if not image_folder.is_dir():
            raise FileNotFoundError(f"Missing image folder: {image_folder}")

    print("=" * 55)
    print("YOLOv8 FRUIT DETECTION TRAINING")
    print("=" * 55)
    print("Classes:", names)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Device: CPU")
    print("Training may take some time. Please wait.")

    # Load pretrained lightweight YOLOv8 Nano model.
    model = YOLO(MODEL_NAME)

    # Train for up to 30 epochs.
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=WORKERS,
        cache=False,
        amp=False,
        pretrained=True,
        patience=30,
        optimizer="SGD",
        cos_lr=False,
        close_mosaic=5,
        plots=True,
        save=True,
        val=True,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        seed=42
    )

    # Locate saved weights.
    weights_dir = Path(PROJECT_NAME) / RUN_NAME / "weights"
    best_model = weights_dir / "best.pt"
    last_model = weights_dir / "last.pt"

    print("\n" + "=" * 55)
    print("TRAINING FINISHED")
    print("=" * 55)
    print("Best model:", best_model)
    print("Last model:", last_model)

    if not best_model.exists():
        raise FileNotFoundError(
            "best.pt was not created. Check the training output for errors."
        )

    # Evaluate the best model on the validation dataset.
    print("\nEvaluating the best model...")
    trained_model = YOLO(str(best_model))
    metrics = trained_model.val(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=WORKERS,
        plots=True
    )

    print("\nValidation results:")
    print("Precision:", round(float(metrics.box.mp), 4))
    print("Recall:", round(float(metrics.box.mr), 4))
    print("mAP50:", round(float(metrics.box.map50), 4))
    print("mAP50-95:", round(float(metrics.box.map), 4))
    print("\nUse this model for video testing:")
    print(best_model.resolve())

if __name__ == "__main__":
    main()
