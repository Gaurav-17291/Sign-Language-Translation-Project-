import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import multiprocessing as mproc
from functools import partial

def get_padded_bbox(landmarks, img_w, img_h, padding_ratio=0.25):
    ##Calculates a bounding box with padding, kept within image boundaries.
    x_coords = [landmark.x * img_w for landmark in landmarks.landmark]
    y_coords = [landmark.y * img_h for landmark in landmarks.landmark]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_pad = box_w * padding_ratio
    y_pad = box_h * padding_ratio

    x_min_padded = max(0, int(x_min - x_pad))
    y_min_padded = max(0, int(y_min - y_pad))
    x_max_padded = min(img_w, int(x_max + x_pad))
    y_max_padded = min(img_h, int(y_max + y_pad))

    if x_min_padded >= x_max_padded or y_min_padded >= y_max_padded:
        return 0, 0, 10, 10

    return x_min_padded, y_min_padded, x_max_padded, y_max_padded

def process_single_video(video_folder, output_root, target_size)
    import sys, os
    if os.getcwd() in sys.path: 
        sys.path.remove(os.getcwd())
    if '' in sys.path: 
        sys.path.remove('')

    try:
        import mediapipe as mp
        import mediapipe.python.solutions 
        mp_holistic = mp.solutions.holistic
    except Exception as e:
        print(f"\n[WORKER CRASH] Real Error: {e}")
        raise e

    video_name = os.path.basename(video_folder)
    split_name = os.path.basename(os.path.dirname(video_folder))

    vid_face_dir = os.path.join(output_root, "face", split_name, video_name)
    vid_lh_dir = os.path.join(output_root, "left_hand", split_name, video_name)
    vid_rh_dir = os.path.join(output_root, "right_hand", split_name, video_name)

    os.makedirs(vid_face_dir, exist_ok=True)
    os.makedirs(vid_lh_dir, exist_ok=True)
    os.makedirs(vid_rh_dir, exist_ok=True)

    frames = sorted(glob(os.path.join(video_folder, "*.png")))
    if not frames:
        return

    blank_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        for frame_path in frames:
            frame_name = os.path.basename(frame_path)
            image = cv2.imread(frame_path)

            if image is None:
                continue

            img_h, img_w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # FACE CROP 
            if results.face_landmarks:
                x1, y1, x2, y2 = get_padded_bbox(results.face_landmarks, img_w, img_h, padding_ratio=0.20)
                face_crop = image[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, target_size)
            else:
                face_crop = blank_image

            # LEFT HAND CROP
            if results.left_hand_landmarks:
                x1, y1, x2, y2 = get_padded_bbox(results.left_hand_landmarks, img_w, img_h, padding_ratio=0.30)
                lh_crop = image[y1:y2, x1:x2]
                lh_crop = cv2.resize(lh_crop, target_size)
            else:
                lh_crop = blank_image

            # RIGHT HAND CROP 
            if results.right_hand_landmarks:
                x1, y1, x2, y2 = get_padded_bbox(results.right_hand_landmarks, img_w, img_h, padding_ratio=0.30)
                rh_crop = image[y1:y2, x1:x2]
                rh_crop = cv2.resize(rh_crop, target_size)
            else:
                rh_crop = blank_image

            jpg_name = frame_name.replace('.png', '.jpg')
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

            cv2.imwrite(os.path.join(vid_face_dir, jpg_name), face_crop, encode_param)
            cv2.imwrite(os.path.join(vid_lh_dir, jpg_name), lh_crop, encode_param)
            cv2.imwrite(os.path.join(vid_rh_dir, jpg_name), rh_crop, encode_param)


def extract_crops_fast(dataset_root, output_root, target_size=(224, 224)):
    print(f"Starting MediaPipe Extraction")

    splits = ["train", "dev", "test"]
    all_video_folders = []

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        folders = glob(os.path.join(split_dir, "*"))
        all_video_folders.extend(folders)

    if not all_video_folders:
        print("No videos found")
        return

    total_cores = mproc.cpu_count()
    usable_cores = 32

    worker = partial(process_single_video, output_root=output_root, target_size=target_size)

    with mproc.Pool(processes=usable_cores) as pool:
        list(tqdm(pool.imap_unordered(worker, all_video_folders), total=len(all_video_folders), desc="Processing Videos Fast"))

    print("Multi-Cue Extraction Complete")

if __name__ == "__main__":
    mproc.set_start_method('spawn', force=True)

    DATASET_ROOT = "/user1/student/mst/mst2024/md2409/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px"
    OUTPUT_ROOT = "/user1/student/mst/mst2024/md2409/multicue_dataset"

    extract_crops_fast(DATASET_ROOT, OUTPUT_ROOT, target_size=(224, 224))
