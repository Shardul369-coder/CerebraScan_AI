import os

UPLOAD_DIR = "backend/storage/uploads"

def save_upload(file, case_id):
    path = f"{UPLOAD_DIR}/{case_id}"
    os.makedirs(path, exist_ok=True)
    return path