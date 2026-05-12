import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================
# SETTINGS
# ==============================

INFO_PATH = "dev_info.npy"
NEW_SIZE = (256, 256)
NUM_WORKERS = min(32, cpu_count())  # Don't use 208 cores! 32 is enough.

# ==============================

def resize_and_replace(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"Skipped: {img_path}"

        resized = cv2.resize(img, NEW_SIZE, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(img_path, resized)
        return None
    except Exception as e:
        return f"Error: {img_path} -> {str(e)}"


def process_video(video_idx):
    info = info_dict[video_idx]

    # If frame_paths exist
    if "frame_paths" in info:
        img_list = info["frame_paths"]
    else:
        return [f"No frame_paths key in video {video_idx}"]

    errors = []
    for img_path in img_list:
        err = resize_and_replace(img_path)
        if err:
            errors.append(err)

    return errors


if __name__ == "__main__":

    print("Loading info dictionary...")
    info_dict = np.load(INFO_PATH, allow_pickle=True).item()

    video_indices = [k for k in info_dict.keys() if isinstance(k, int)]

    print(f"Total videos: {len(video_indices)}")
    print(f"Resize to: {NEW_SIZE}")
    print(f"Using {NUM_WORKERS} CPU cores\n")

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_video, video_indices),
                            total=len(video_indices)))

    all_errors = [err for sublist in results for err in sublist if sublist]

    if len(all_errors) > 0:
        print("\nSome errors occurred:")
        for e in all_errors[:10]:
            print(e)
    else:
        print("\nAll images resized successfully ✅")

