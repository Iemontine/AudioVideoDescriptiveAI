import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import OneHotEncoder
from torchvision.models import resnet18, ResNet18_Weights

# load ontology
with open('ontology.json') as f: ontology = json.load(f)

# extract all unique IDs
ids = []
for item in ontology:
    ids.append(item['id'])

# save encoder, used to decode encoded labels later
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# dicts used to convert between ytid <-> id <-> encoded label
id_to_one_hot = {}
id_to_name = {}

# ====================== Dataset Loader ======================
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, quiet=True, display=False, reduced=False, target_length=600):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.quiet = quiet
        self.display = display
        self.data = []
        self.id_to_name = None
        self.parse_dataset(csv_file)
        self.load_data(reduced)

    # loads the dataset
    def load_data(self, reduced=False):
        # Remove samples without an existing audio file
        print(f"Loaded {len(self.data)} samples from dataset")
        self.data = [(ytid, start_sec, end_sec, ids) for ytid, start_sec, end_sec, ids in self.data if os.path.exists(os.path.join(self.audio_dir, f"{ytid}.flac"))]
        print(f"Filtered to {len(self.data)} samples with existing audio files")

        id_count = self.count_ids(quiet=self.quiet)
        
        global id_to_one_hot, id_to_name

        filtered_ids = list(set(id for sample in self.data for id in sample[3]))
        self.data = [(ytid, start_sec, end_sec, ids) for ytid, start_sec, end_sec, ids in self.data if any(id in filtered_ids for id in ids)]

        filtered_ids = list(set(id for sample in self.data for id in sample[3]))
        one_hot_encoder.fit([[id] for id in filtered_ids])
        id_to_one_hot = {id: one_hot_encoder.transform([[id]]).flatten() for id in filtered_ids}
        id_to_name = {item['id']: item['name'] for item in ontology if item['id'] in filtered_ids}
        self.id_to_name = id_to_name

        return id_to_one_hot, id_to_name
        
    # preprocesses and returns a datapoint: mel spectrogram and its corresponding true label
    def __getitem__(self, idx):
        ytid, start_sec, end_sec, ids = self.data[idx]              # extracts one sample from the loaded dataset
        audio_path = os.path.join(self.audio_dir, f"{ytid}.flac")   # path to the audio file

        # obtain waveform which measures amplitude over time
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)

        # we want as many things to be similar between inputs as possible
        # so, duplicate stereo to both L/R ears if it has multiple channels
        if waveform.shape[0] > 1:    
            waveform = torch.mean(waveform, dim=0)
        # convert mono audio clips to stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # display the waveform
        if self.display:
            self.display_waveform(waveform, ytid, start_sec, end_sec)

        # define transform used to convert waveform to mel spectrogram with decibel scaling
        transform = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512),
            AmplitudeToDB(), # mel spectrogram measures frequency amplitude over time, more informative for audio data
            TimeMasking(time_mask_param=30),         # Time masking for augmentation
            FrequencyMasking(freq_mask_param=10)     # Frequency masking for augmentation
        )
        # convert waveform to mel spectrogram
        mel_spectrogram = transform(waveform)

        # add the channel dimension if not present (needed for CNN)
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        if mel_spectrogram.shape[2] < self.target_length:   # loop content if too short  
            num_repeats = int(np.ceil(self.target_length / mel_spectrogram.shape[2]))
            mel_spectrogram = mel_spectrogram.repeat(1, 1, num_repeats)
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]
        elif mel_spectrogram.shape[2] > self.target_length: # truncate if too long
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]

        # display the mel spectrogram
        if self.display:
            self.display_spectrogram(mel_spectrogram, ytid, start_sec, end_sec, ids)

        # convert labels to float tensor
        one_hot_array = np.array([id_to_one_hot[id_] for id_ in ids if id_ in id_to_one_hot])
        one_hot_labels = torch.sum(torch.FloatTensor(one_hot_array), dim=0)

        return mel_spectrogram, one_hot_labels
    
    def display_waveform(self, waveform, ytid, start_sec, end_sec):
        print(f"Shape of waveform: {waveform.shape}")
        plt.figure(figsize=(10, 5))
        plt.plot(waveform[0].numpy())
        plt.title(f'Waveform of File: {ytid} of length {end_sec - start_sec} seconds')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.show()
    def display_spectrogram(self, mel_spectrogram, ytid, start_sec, end_sec, labels):
        print(f"Shape of mel_spectrogram: {mel_spectrogram.shape}")
        mel_spectrogram_np = mel_spectrogram[0].detach().numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(mel_spectrogram_np, aspect='auto', origin='lower',
                extent=[0, mel_spectrogram_np.shape[1], 0, mel_spectrogram_np.shape[0]])
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram of File: {ytid} ({id_to_name[labels[0]]}) of length {end_sec - start_sec} seconds')
        plt.xlabel('Time (frames)')
        plt.ylabel('Mel Frequency (bins)')
        plt.show()
    def count_ids(self, quiet=False):
        # Counts the occurrences of each label type
        label_count = {}
        for _, _, _, labels in self.data:
            for label in labels:
                if label not in label_count:    # register the label if it's not in the dictionary
                    label_count[label] = 1
                else:                           # increment the count if it's already in the dictionary
                    label_count[label] += 1

        # Output each label's true unencoded name and its count, sorted ascending by count
        if not quiet:
            sorted_label_count = sorted(label_count.items(), key=lambda item: item[1])
            for label, count in sorted_label_count:
                label_name = next(item["name"] for item in ontology if item["id"] == label)
                print(f"Label: {label_name}, Count: {count}")
        return label_count
    def parse_dataset(self, csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()[3:]
            for line in lines:
                parts = line.strip().split(', ')
                ytid = parts[0]
                start_sec = float(parts[1])
                end_sec = float(parts[2])

                labels = [label.replace("\"", "") for label in parts[3].split(',')]

                self.data.append((ytid, start_sec, end_sec, labels))
    def data_labels(self):
        return [labels for _, _, _, labels in self.data]
    def __len__(self):
        return len(self.data)
    def get_label_count(self):
        return len(id_to_name)

# ====================== Model Architecture ======================
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False)
        self.resnet.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x

# ====================== Training ======================
losses = []
def init_plot():
    plt.ion()
    plt.figure()
    plt.show()
def update_plot(loss, track='epoch'):
    losses.append(loss)
    plt.clf()
    plt.ylim(0, max(losses) * 1.1)
    plt.grid()
    if track == 'epoch':
        plt.title("Epoch Loss Over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Epoch Loss")
        plt.plot(losses, label='Epoch Loss', color='blue')
    elif track == 'batch':
        plt.title("Batch Loss Over Time")
        plt.xlabel("Iterations")
        plt.ylabel("Batch Loss")
        plt.plot(losses, label='Batch Loss', color='blue')
    plt.legend()
    plt.pause(0.2)
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, track='batch'):
    if track == 'batch' or track == 'epoch': init_plot()

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()       # set model to training mode
        running_loss = 0.0  # used to track loss of network over time
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

        for spectrogram, labels in progress_bar:    # calls __getitem__ of AudioDataset
            # move tensors to GPU
            spectrogram, labels = spectrogram.to(device), labels.to(device)

            optimizer.zero_grad()               # zero the parameter gradients

            # Mixed precision training
            with autocast():
                outputs = model(spectrogram)        # forward pass
                labels = labels.squeeze(1)          # remove extra dimension
                loss = criterion(outputs, labels)   # compute the loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model parameters
            scaler.step(optimizer)
            scaler.update()

            # continue to track statistics
            batch_loss = loss.item()
            running_loss += loss.item() * spectrogram.size(0)
            progress_bar.set_postfix(loss=batch_loss)
            if track == 'batch': update_plot(batch_loss, track=track)
        scheduler.step()
        if track == 'epoch': update_plot(batch_loss, track=track)
        torch.save(model.state_dict(), f'./models/model_checkpoint_e{epoch}.pth')

    plt.savefig('batch_loss_plot.png')

# ====================== Main ======================
if __name__ == "__main__":
    source = 'eval_segments'
    dataset = AudioDataset(f'./data/{source}.csv', f'./sounds_{source}', quiet=True, display=False, reduced=False, target_length=600)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = AudioClassifier(num_classes=len(id_to_name))
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=5, track='batch')