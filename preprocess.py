import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import AudioDataset, AudioClassifier, one_hot_encoder
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn as nn
import torchaudio
import json
import librosa


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_to_name = {}

def video_to_images(video_path, output_dir, fps):
    global id_to_name
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame_rate = clip.fps
    # Compute the number of frames per segment (10 seconds segment)
    frames_per_segment = int(frame_rate * 10)
    
    # Load the audio classifier model once

    with open('ontology.json') as f: ontology = json.load(f)
    id_to_name = {item['id']: item['name'] for item in ontology}
    model = AudioClassifier(num_classes=len(id_to_name)).to(device)
    model.load_state_dict(torch.load('./models/model_checkpoint_e6.pth'))
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = []
    font_size = int(clip.size[1] * 0.05)    # 5% the height of the video
    font = ImageFont.truetype("arial.ttf", font_size)
    position = (10, 10)
    
    # Keeps track of the current audio prediction
    current_sound_effect = None
    audio_segment_start = 0
    
    print(int(duration * fps))
    for frame_number in range(int(duration * fps)):
        timestamp = frame_number / fps
        img = clip.get_frame(timestamp)
        img_pil = Image.fromarray(img)
        
        # Update the audio segment and predict the sound effect
        if frame_number % frames_per_segment == 0:
            print(f"examining segment: {audio_segment_start} - {audio_segment_start + 10}")
            sound_effect = predict_sound_effect(audio_segment_start, audio_segment_start + 10, video_path, model, device)
            current_sound_effect = sound_effect
            audio_segment_start += 10

        timestamp_text = f"Time: {timestamp:.2f}s\nSound Effect: {current_sound_effect}"

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

    return images, frame_rate, fps

def predict_sound_effect(start_sec, end_sec, video_path, model, device):
    # Load the audio from the video segment
    clip = VideoFileClip(video_path)
    audio = clip.audio.subclip(start_sec, end_sec)
    audio.write_audiofile("temp_audio.wav", fps=48000)

    # Load the waveform and convert it to a mel spectrogram
    waveform, sample_rate = torchaudio.load("temp_audio.wav", normalize=True)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)  # Convert to mono if stereo
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Ensure it has a batch dimension
    
    # Define transform (same as in train.py)
    transform = torch.nn.Sequential(
        MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512),
        AmplitudeToDB()
    )
    
    mel_spectrogram = transform(waveform)

    # Ensure the mel spectrogram has the correct target length
    target_length = 600  # Use the same target length as in train.py
    if mel_spectrogram.shape[2] < target_length:
        padding = target_length - mel_spectrogram.shape[2]
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
    elif mel_spectrogram.shape[2] > target_length:
        mel_spectrogram = mel_spectrogram[:, :, :target_length]

    # print(f"Shape of mel_spectrogram: {mel_spectrogram.shape}")
    # mel_spectrogram_np = mel_spectrogram[0].detach().numpy()
    # plt.figure(figsize=(10, 5))
    # plt.imshow(mel_spectrogram_np, aspect='auto', origin='lower',
    #         extent=[0, mel_spectrogram_np.shape[1], 0, mel_spectrogram_np.shape[0]])
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(f'Mel Spectrogram of File: of length {end_sec - start_sec} seconds')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Mel Frequency (bins)')
    # plt.show()

    # Add batch and channel dimensions if needed
    mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(mel_spectrogram)  # Forward pass through the model

        print(outputs)
        print()
        print(torch.sigmoid(outputs))
        print()
        print(torch.sigmoid(outputs).round())

        predicted_labels = torch.sigmoid(outputs).round()  # Sigmoid for multi-label classification

        # Convert prediction to a human-readable label
        preds = predicted_labels.cpu().numpy().astype(int).tolist() 
        decoded_preds = one_hot_encoder.inverse_transform(preds)  # Decode one-hot to label index

        # Map predicted indices to sound labels
        predicted_sounds = [id_to_name[pred[0]] for pred in decoded_preds if pred[0] in id_to_name]

    return ', '.join(predicted_sounds) if predicted_sounds else f"Unknown --> {start_sec} - {end_sec}" 

input_video_path = "input.mp4"

dataset = 'eval_segments'
test_dataset = AudioDataset(f'./data/{dataset}.csv', f'./sounds_{dataset}', target_length=600)
_, id_to_name = test_dataset.load_data()

VideoFileClip("30_second_animation_assignment.mp4").write_videofile(input_video_path, codec='libx264', logger=None)
output_dir = "output"
print("Converting video to image array...")
images, original_fps, reduced_fps = video_to_images(input_video_path, output_dir, fps=24)
clip = ImageSequenceClip(images, fps=24)
print("Image array successfully created.")

# from IPython.display import display, Video

# sanity check: ensure the image array has been edited correctly
output_video_path = "output.mp4"
clip.write_videofile(output_video_path, codec='libx264', logger=None)
# display(Video(output_video_path, width=500))