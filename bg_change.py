import cv2
import numpy as np
from tqdm import tqdm

input_video_path = 'results/output.avi'
output_video_path = 'results/new_video.avi'
background_path = 'bg_img/bg.png'

def blend_frame_with_background(frame, background, blur_size=5):
    blur_size = blur_size | 1 # In case user give an even number
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask_inv_blurred = cv2.bitwise_not(mask_blurred)

    bg_blurred = cv2.bitwise_and(background, background, mask=mask_inv_blurred)
    fg_blurred = cv2.bitwise_and(frame, frame, mask=mask_blurred)
    final_frame = cv2.add(bg_blurred, fg_blurred)
    return final_frame

def change_background(input_video_path, output_video_path, background_path, blur_size=15):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        bg_img = cv2.imread(background_path)
        if bg_img is None:
            raise IOError(f"Cannot load background image {background_path}")
        bg_img = cv2.resize(bg_img, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                final_frame = blend_frame_with_background(frame, bg_img, blur_size=blur_size)

                output_video.write(final_frame)
                pbar.update(1)
    finally:
        cap.release()
        output_video.release()


if __name__ == '__main__':
    change_background(input_video_path, output_video_path, background_path, blur_size=15)