import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from matplotlib.colors import ListedColormap
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ======== CONFIGURATION ========
MRI_MODALITY = "flair"  # Can also try "t1ce", "t1", "t2"

# High resolution settings
DPI = 300
FIGSIZE = (10, 10)

# ======== IMPROVED COLOR MAP with Green, Yellow, and Blue ========
BRATS_CMAP = ListedColormap([
    (0, 0, 0, 0),          # 0 = background (transparent)
    (0, 0.4, 1, 0.8),      # 1 = Blue - NET/NCR 
    (0, 1, 0, 0.7),        # 2 = Green - Edema
    (1, 1, 0, 0.85),       # 3 = Yellow - Enhancing Tumor
])

# ======== ALIGNMENT FUNCTIONS ========

def align_prediction_to_mri(pred_vol, mri_vol):
    """Align prediction volume to MRI volume by resizing"""
    if pred_vol.shape == mri_vol.shape:
        print(f"  ✓ Volumes already aligned: {pred_vol.shape}")
        return pred_vol
    
    print(f"  → Aligning prediction {pred_vol.shape} to MRI {mri_vol.shape}")
    
    # Calculate zoom factors
    zoom_factors = [mri_vol.shape[i] / pred_vol.shape[i] for i in range(3)]
    print(f"  → Zoom factors: [{zoom_factors[0]:.2f}, {zoom_factors[1]:.2f}, {zoom_factors[2]:.2f}]")
    
    # Resize using nearest neighbor to preserve class labels
    aligned_pred = ndimage.zoom(pred_vol, zoom_factors, order=0)
    
    # Ensure exact dimensions match
    if aligned_pred.shape != mri_vol.shape:
        print(f"  → Fine-tuning from {aligned_pred.shape} to {mri_vol.shape}")
        aligned_pred = match_dimensions(aligned_pred, mri_vol.shape)
    
    print(f"  ✓ Aligned to: {aligned_pred.shape}")
    return aligned_pred

def match_dimensions(array, target_shape):
    """Match array dimensions to target shape by cropping or padding"""
    result = array.copy()
    
    for axis in range(3):
        current_size = result.shape[axis]
        target_size = target_shape[axis]
        
        if current_size > target_size:
            # Crop
            diff = current_size - target_size
            start = diff // 2
            end = start + target_size
            if axis == 0:
                result = result[start:end, :, :]
            elif axis == 1:
                result = result[:, start:end, :]
            else:
                result = result[:, :, start:end]
        elif current_size < target_size:
            # Pad
            diff = target_size - current_size
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width = [(0, 0), (0, 0), (0, 0)]
            pad_width[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode='constant', constant_values=0)
    
    return result

# ======== LOADERS ========

def load_prediction(pred_file):
    """Load prediction file and convert to class labels"""
    try:
        pred = np.load(pred_file)
        if pred.ndim == 4:
            pred = np.argmax(pred, axis=-1)
        print(f"  Prediction: {pred.shape}, classes={np.unique(pred)}, slices={pred.shape[2]}")
        return pred
    except Exception as e:
        print(f"  [ERROR] Failed to load prediction: {e}")
        return None

def load_all_mri_modalities(raw_data_dir):
    """
    Backend version:
    MRI files are directly inside:
    backend/storage/uploads/<case_id>/
    """

    patient_dir = Path(raw_data_dir)

    if not patient_dir.exists():
        print(f"  [ERROR] MRI folder not found: {patient_dir}")
        return None

    print(f"  Scanning folder: {patient_dir}")

    nii_files = {}

    for f in patient_dir.iterdir():
        if f.suffix in [".nii", ".gz"] or f.name.endswith(".nii.gz"):
            fname_lower = f.name.lower()

            if 'flair' in fname_lower:
                nii_files['flair'] = f
            elif 't1ce' in fname_lower:
                nii_files['t1ce'] = f
            elif 't2' in fname_lower:
                nii_files['t2'] = f
            elif 't1' in fname_lower:
                nii_files['t1'] = f
            elif 'seg' not in fname_lower:  # Skip segmentation files
                nii_files['other'] = f
            print(f"    Found: {f.name}")

    if not nii_files:
        print(f"  [ERROR] No NIfTI files found in {patient_dir}")
        return None

    modality_priority = ['flair', 't1ce', 't2', 't1','other']

    for modality in modality_priority:
        if modality in nii_files:
            load_file = nii_files[modality]
            print(f"  → Loading MRI ({modality}): {load_file.name}")

            try:
                vol = nib.load(str(load_file)).get_fdata()

                # simple normalization (safer for backend)
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

                print(f"  ✓ Loaded MRI: {vol.shape}")
                return vol

            except Exception as e:
                print(f"  [ERROR] Failed to load {modality}: {e}")
                continue

    print("  [ERROR] Could not load any MRI modality")
    return None

# ======== SAVE HELPERS ========

def save_overlay_slice(mri_slice, pred_slice, patient_id, slice_idx, output_dir, total_slices):
    """Save MRI with prediction overlay in high resolution"""
    out_file = output_dir / f"slice_{slice_idx:03d}_overlay.png"
    
    # Check if MRI slice has data
    mri_has_data = np.max(mri_slice) > 0.01
    
    if not mri_has_data:
        # If no MRI data, skip this slice or create a warning
        return None
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Display MRI in grayscale
    ax.imshow(mri_slice, cmap="gray", interpolation='bilinear', 
              origin='lower', vmin=0, vmax=1)
    
    # Overlay prediction with transparency (only where tumor exists)
    if np.max(pred_slice) > 0:
        ax.imshow(pred_slice, cmap=BRATS_CMAP, alpha=0.65, vmin=0, vmax=3, 
                  interpolation='nearest', origin='lower')
    
    # Add title
    ax.set_title(f"{patient_id} - Slice {slice_idx+1}/{total_slices}", 
                fontsize=14, fontweight="bold", pad=10)
    ax.axis('off')
    
    # Save
    plt.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return out_file

def save_comparison_view(mri_slice, pred_slice, patient_id, slice_idx, output_dir, total_slices):
    """Save side-by-side comparison"""
    out_file = output_dir / f"slice_{slice_idx:03d}_comparison.png"
    
    # Check if MRI has data
    if np.max(mri_slice) <= 0.01:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MRI only
    axes[0].imshow(mri_slice, cmap="gray", interpolation='bilinear', origin='lower', vmin=0, vmax=1)
    axes[0].set_title("MRI", fontsize=12, fontweight="bold")
    axes[0].axis('off')
    
    # MRI + Overlay
    axes[1].imshow(mri_slice, cmap="gray", interpolation='bilinear', origin='lower', vmin=0, vmax=1)
    if np.max(pred_slice) > 0:
        axes[1].imshow(pred_slice, cmap=BRATS_CMAP, alpha=0.65, vmin=0, vmax=3, 
                       interpolation='nearest', origin='lower')
    axes[1].set_title("MRI + Prediction Overlay", fontsize=12, fontweight="bold")
    axes[1].axis('off')
    
    # Prediction only
    axes[2].imshow(pred_slice, cmap=BRATS_CMAP, vmin=0, vmax=3, 
                   interpolation='nearest', origin='lower')
    axes[2].set_title("Segmentation (Blue=NET/NCR, Green=Edema, Yellow=ET)", 
                     fontsize=10, fontweight="bold")
    axes[2].axis('off')
    
    fig.suptitle(f"{patient_id} - Slice {slice_idx+1}/{total_slices}", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return out_file

def create_legend_image(output_dir):
    """Create color legend"""
    legend_file = output_dir / "color_legend.png"
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    colors = [(0, 0.4, 1), (0, 1, 0), (1, 1, 0)]
    labels = ['Blue: NET/NCR (Non-enhancing tumor)', 
              'Green: Edema (Peritumoral edema)', 
              'Yellow: ET (Enhancing tumor)']
    
    y_positions = [0.7, 0.45, 0.2]
    
    for color, label, y_pos in zip(colors, labels, y_positions):
        rect = plt.Rectangle((0.1, y_pos), 0.12, 0.12, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.27, y_pos + 0.06, label, 
               fontsize=13, fontweight='bold', va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("BraTS Tumor Segmentation - Color Legend", 
                fontsize=15, fontweight='bold', pad=15)
    
    plt.savefig(legend_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Legend saved: {legend_file}")
    return legend_file

# ======== MAIN VISUALIZATION ========

def visualize_patient(patient_id,pred_file,output_dir,raw_data_dir,save_comparison=True,save_all_slices=False):
    """Generate overlay visualizations"""
    print("\n" + "="*60)
    print(f"VISUALIZING: {patient_id}")
    print("="*60)

    # Create output directory
    out_dir = Path(output_dir) 
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create legend
    create_legend_image(out_dir)

    # Load prediction
    pred_vol = load_prediction(pred_file)
    if pred_vol is None:
        return

    # Load MRI (try all modalities)
    mri_vol = load_all_mri_modalities(raw_data_dir)
    if mri_vol is None:
        print("  [ERROR] Cannot create overlays without valid MRI data")
        return

    # Align prediction to MRI
    pred_vol_aligned = align_prediction_to_mri(pred_vol, mri_vol)

    # Process slices
    depth = pred_vol_aligned.shape[2]
    print(f"\n  📊 Processing {depth} slices...")

    saved_count = 0
    slices_with_tumor = 0
    slices_with_mri = 0
    
    for slice_idx in range(depth):
        mri_slice = mri_vol[:, :, slice_idx]
        pred_slice = pred_vol_aligned[:, :, slice_idx]
        
        has_mri = np.max(mri_slice) > 0.01
        has_tumor = np.max(pred_slice) > 0
        
        if has_mri:
            slices_with_mri += 1
        if has_tumor:
            slices_with_tumor += 1
        
        # Only save slices with MRI data
        if has_mri and (save_all_slices or has_tumor):
            # Save overlay
            result = save_overlay_slice(mri_slice, pred_slice, patient_id, slice_idx, out_dir, depth)
            if result:
                saved_count += 1
            
            # Save comparison for key slices
            if save_comparison and has_tumor and (slice_idx % 10 == 0 or np.sum(pred_slice > 0) > 200):
                save_comparison_view(mri_slice, pred_slice, patient_id, slice_idx, out_dir, depth)
        
        # Progress
        if (slice_idx + 1) % 20 == 0 or (slice_idx + 1) == depth:
            print(f"  → Progress: {slice_idx + 1}/{depth} slices processed...")

    print(f"\n  ✅ Summary:")
    print(f"     - Total slices: {depth}")
    print(f"     - Slices with MRI data: {slices_with_mri}")
    print(f"     - Slices with tumor: {slices_with_tumor}")
    print(f"     - Images saved: {saved_count}")
    print(f"     - Output: {out_dir}")
    print(f"  🎨 Colors: Blue (NET/NCR), Green (Edema), Yellow (ET)")

# ======== BATCH RUN ========

def visualize_all(save_comparison=True, save_all_slices=False):
    """Process all patients"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    pred_files = sorted([f for f in PRED_DIR.glob("*_3d.npy") if "_probs_" not in f.name])

    print("="*60)
    print("  3D BRAIN TUMOR VISUALIZATION")
    print("  Green-Yellow-Blue Color Scheme with MRI Background")
    print("="*60)
    print(f"[CONFIG] DPI: {DPI}")
    print(f"[CONFIG] Comparisons: {'Yes' if save_comparison else 'No'}")
    print(f"[CONFIG] Save all slices: {'Yes' if save_all_slices else 'Only tumor slices'}")
    print(f"[INFO] Found {len(pred_files)} prediction files")
    
    if not RAW_DATA_DIR.exists():
        print(f"[ERROR] Raw data directory not found: {RAW_DATA_DIR}")
        return
    
    success_count = 0
    for idx, pf in enumerate(pred_files, 1):
        pid = pf.stem.replace("_3d", "")
        print(f"\n[{idx}/{len(pred_files)}] Processing: {pid}")
        
        try:
            visualize_patient(pid, pf, save_comparison=save_comparison, save_all_slices=save_all_slices)
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"✅ Complete! Processed {success_count}/{len(pred_files)} patients")
    print("="*60)

if __name__ == "__main__":
    # save_all_slices=True will save all slices (even without tumor)
    # save_all_slices=False will only save slices with tumor (recommended)
    visualize_all(save_comparison=True, save_all_slices=False)