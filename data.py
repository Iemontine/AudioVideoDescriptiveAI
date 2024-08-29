import csv
from yt_dlp import YoutubeDL
from yt_dlp import DownloadError
import subprocess
import os
import time

# TODO: once done, loop through everything and check all urls not downloaded

def trim_video(input_file, start_time, end_time, output_file):
    command = [
        'ffmpeg',
        '-n',
        '-loglevel', 'error',
        '-hide_banner',
        '-i', input_file,
        '-ss', start_time,
        '-to', end_time,
        '-c', 'copy',
        output_file
    ]
    subprocess.run(command, check=True)

def convert_to_wav(input_file, output_file, sample_rate=16000, channels=1):
    command = [
        'ffmpeg',
        '-n',
        '-loglevel', 'error',
        '-hide_banner',
        '-i', input_file,
        '-ar', str(sample_rate),
        '-ac', str(channels),
        output_file
    ]
    subprocess.run(command, check=True)

def download_from_youtube():
    dataset = 'balanced_train_segments copy'
    with open('.\data\\' + dataset + '.csv', 'r') as file:
        reader = csv.reader(file)
        for i in range(3): next(reader)
        for entry in reader:
            url = entry[0]
            video_url = f'https://www.youtube.com/watch?v={url}'
            start_time = entry[1]
            end_time = entry[2]

            in_file = f'in_{dataset}.mp4'
            mid_file = f'mid_{dataset}.mp4'
            out_file = f'./sounds_{dataset}/{url}.wav'

            try:
                print(f'Downloading {url}')
                with YoutubeDL({'format': 'best', 'outtmpl': in_file, 'quiet': True}) as ydl:
                    ydl.download([video_url])
            except DownloadError as e:
                if 'not a bot' in str(e):
                    time.sleep(15)
                continue

            try:
                trim_video(in_file, start_time, end_time, mid_file)
                os.remove(in_file)
            except Exception as e:
                os.remove(in_file)
                continue

            try:
                convert_to_wav(mid_file, out_file)
                os.remove(mid_file)
            except Exception as e:
                os.remove(mid_file)
                continue

if __name__ == '__main__':
    download_from_youtube()