from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ====================================
# CONFIG
# ====================================
INPUT_DIR = Path("predictions_3d")   # folder with .npy files
OUTPUT_DIR = Path("visualizations")

OUTPUT_DIR.mkdir(exist_ok=True)


# ====================================
# NORMALIZE IMAGE (important for PNG)
# ====================================
def normalize(img):
    img = img.astype(np.float32)

    if img.max() == img.min():
        return np.zeros_like(img)

    img = (img - img.min()) / (img.max() - img.min())
    return img


# ====================================
# SAVE SINGLE SLICE
# ====================================
def save_png(img, save_path):
    img = normalize(img)

    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(img, cmap="gray")

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()


# ====================================
# CONVERT NPY FILE
# ====================================
def convert_npy(npy_path):

    # skip probability files (optional)
    if "probs" in npy_path.name:
        return

    print(f"Processing: {npy_path.name}")

    arr = np.load(npy_path)

    case_name = npy_path.stem
    case_out = OUTPUT_DIR / case_name
    case_out.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 2D IMAGE
    # -------------------------
    if arr.ndim == 2:
        save_png(arr, case_out / "slice_0.png")

    # -------------------------
    # 3D VOLUME
    # -------------------------
    elif arr.ndim == 3:
        for i in range(arr.shape[2]):
            save_png(arr[:, :, i], case_out / f"slice_{i}.png")

    else:
        print(f"Skipping unsupported shape: {arr.shape}")


# ====================================
# MAIN
# ====================================
def main():

    npy_files = list(INPUT_DIR.rglob("*.npy"))

    if not npy_files:
        print("No .npy files found")
        return

    print(f"Found {len(npy_files)} files")

    for file in npy_files:
        convert_npy(file)

    print("✅ Conversion complete")


if __name__ == "__main__":
    main()