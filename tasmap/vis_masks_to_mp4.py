import cv2
import os
from natsort import natsorted
from PIL import Image
import imageio

# Set the directory containing PNGs and the output filenames
image_folder = '/workspace/MaskClustering/data/tasmap/processed/scene0000_00/output/vis_mask'
output_video = 'output_video_small.mp4'
output_gif = 'output_video_small.gif'

# Get list of PNG files and sort them naturally
images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images = natsorted(images)

# Read the first image to get the size
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, _ = frame.shape

# Resize factor
resize_factor = 0.5
new_width = int(width * resize_factor)
new_height = int(height * resize_factor)

# Create video writer
fps = 2  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

# Prepare frames for GIF
gif_frames = []

# Write each image to video and collect for GIF
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    video.write(resized_frame)

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    gif_frames.append(pil_image)

video.release()
print(f"Video saved as {output_video}")

# Save as GIF
gif_frames[0].save(
    output_gif,
    save_all=True,
    append_images=gif_frames[1:],
    duration=int(1000 / fps),  # duration per frame in ms
    loop=0
)
print(f"GIF saved as {output_gif}")
