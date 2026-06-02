from ultralytics import YOLO
from pathlib import Path
import tifffile
import numpy as np
import cv2

WEIGHTS_PATH = Path("-")
IMAGE_PATH = Path("-")  # TIFF to run

CONF_THRESHOLD = 0.15


def load_tiff_as_rgb(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))  # shape could be (H,W), (H,W,C), or (N,H,W,...) etc.
    # If multi-page, take the first page
    if arr.ndim == 3 and arr.shape[0] > 3:  # e.g. (N,H,W)
        arr = arr[0]
    # Ensure (H,W) or (H,W,C)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]  # take first 3 bands

    arr = arr.astype(np.float32)
    # Simple scaling to 0–255 uint8
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    arr_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr_u8  # (H,W,3) uint8


def main():
    print(f"[info] Using weights: {WEIGHTS_PATH.resolve()}")
    print(f"[info] Using image:   {IMAGE_PATH.resolve()}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    model = YOLO(str(WEIGHTS_PATH))

    if IMAGE_PATH.suffix.lower() in {".tif", ".tiff"}:
        img = load_tiff_as_rgb(IMAGE_PATH)
        results = model.predict(
            source=img,      # pass ndarray directly
            imgsz=640,
            conf=CONF_THRESHOLD,
            save=True,
        )
    else:
        results = model.predict(
            source=str(IMAGE_PATH),
            imgsz=640,
            conf=CONF_THRESHOLD,
            save=False,
        )

    r = results[0]



    # Get visualization (BGR image)
    annotated = r.plot()

    # Create output path: same name but .png
    out_path = IMAGE_PATH.with_suffix(".jpg")
    cv2.imwrite(str(out_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  
    print(f"[info] Saved result to: {out_path}")
    print("[info] Inference done. Boxes:")
    print(r.boxes)
    print("[info] Visualized result saved (check latest runs/detect/predict*/).")


if __name__ == "__main__":
    main()
