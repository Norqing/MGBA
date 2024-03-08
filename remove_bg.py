import cv2
from rembg import remove
from tqdm import tqdm

input_video_path = 'videos/LQT_cut.mp4'
output_video_path = 'videos/output_video_2.avi'

def remove_background(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                alpha = remove(frame)
                bgr = alpha[:, :, :3]
                output_video.write(bgr)

                pbar.update(1)
    finally:
        cap.release()
        output_video.release()

if __name__ == '__main__':
    remove_background(input_video_path, output_video_path)