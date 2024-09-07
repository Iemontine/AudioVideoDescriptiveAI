import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import torch
from panns_inference import SoundEventDetection, labels
import torchaudio
import matplotlib.pyplot as plt
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

# Convert video to images and extract timestamps
def video_to_images(video_path, output_dir, framewise_output, fps):
    clip = VideoFileClip(video_path)
    duration = clip.duration

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = []
    font_size = int(clip.size[1] * 0.05)  # 5% the height of the video
    font = ImageFont.truetype("arial.ttf", font_size)
    position = (10, 10)

    timestamps = []

    classwise_output = np.max(framewise_output, axis=0)
    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]
    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    for idx in idxes:
        line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
        lines.append(line)

    for frame_number in range(int(duration * fps)):
        timestamp = frame_number / fps
        timestamps.append(timestamp)
        img = clip.get_frame(timestamp)
        img_pil = Image.fromarray(img)

        # Add this calculation before the timestamp_text
        frame_index = int(frame_number * len(framewise_output) / (duration * fps))
        if frame_index < len(framewise_output):
            frame_events = framewise_output[frame_index]
            top_label_idx = np.argmax(frame_events)
            top_label = labels[top_label_idx]
        else:
            top_label = "No Classification"

        # Use top_label in the timestamp_text
        timestamp_text = f"Time: {timestamp:.2f}s \nSound Effect: {top_label}"

        # Masking technique to draw inverted text on the image
        text_mask = Image.new("L", img_pil.size, 0)
        draw_mask = ImageDraw.Draw(text_mask)
        draw_mask.text(position, timestamp_text, fill=255, font=font)
        inverted_image = ImageOps.invert(img_pil)                           # Invert the colors in the region of the text
        final_image = Image.composite(inverted_image, img_pil, text_mask)   # Combine the original image with the inverted region using the text mask

        # Save the resultant image to path
        image_path = os.path.join(output_dir, f"f{frame_number:04d}.jpg")
        final_image.save(image_path, "JPEG")
        images.append(image_path)

    return images, clip.fps, fps, timestamps

# Main execution
input_video_path = "./videos/input.mp4"
output_dir = "output"
fps = 24

audio_path = "extracted_audio.wav"
video_clip = VideoFileClip(input_video_path)
audio = video_clip.audio
audio.write_audiofile(audio_path)
(audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
audio = audio[None, :]  # (batch_size, segment_samples)
sed = SoundEventDetection(checkpoint_path='.\models\Cnn14_DecisionLevelMax.pth', device=device, interpolate_mode='nearest')
framewise_output = sed.inference(audio)

print("Converting video to image array...")
images, original_fps, reduced_fps, timestamps = video_to_images(input_video_path, output_dir, framewise_output[0], fps=fps)
print("Image array successfully created.")

clip = ImageSequenceClip(images, fps=reduced_fps)
original_audio = VideoFileClip(input_video_path).audio
output_video_path = "./videos/output.mp4"
clip = clip.set_audio(original_audio)  # Set the original audio
clip.write_videofile(output_video_path, codec='libx264')
print(f"Output video saved as {output_video_path}")