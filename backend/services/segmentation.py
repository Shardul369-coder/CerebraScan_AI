from src.predict_and_stack import run_inference

MODEL_PATH = "checkpoints/seg_best.h5"

def run_segmentation(img_dir, case_id):

    output_dir = f"backend/storage/masks/{case_id}"

    return run_inference(
        model_path=MODEL_PATH,
        test_img_dir=img_dir,
        output_dir=output_dir
    )