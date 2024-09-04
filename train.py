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
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from sklearn.preprocessing import OneHotEncoder

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
    def __init__(self, csv_file, audio_dir, quiet=True, reduced=False, target_length=600):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.quiet = quiet
        self.data = []
        self.parse_dataset(csv_file)
        self.load_data(reduced)

    # loads the dataset
    def load_data(self, reduced=False):
        # Remove samples without an existing audio file
        print(f"Loaded {len(self.data)} samples from dataset")
        self.data = [(ytid, start_sec, end_sec, ids) for ytid, start_sec, end_sec, ids in self.data if os.path.exists(os.path.join(self.audio_dir, f"{ytid}.flac"))]
        print(f"Filtered to {len(self.data)} samples with audio files")

        id_count = self.count_ids(quiet=self.quiet)
        
        global id_to_one_hot, id_to_name
        if reduced:
            # Remove labels with less than 100 examples or more than 150 examples
            filtered_ids = [id for id, count in id_count.items() if 100 <= count <= 150]
            self.data = [(ytid, start_sec, end_sec, ids) for ytid, start_sec, end_sec, ids in self.data if any(id in filtered_ids for id in ids)]
            print(f"Filtered to {len(self.data)} samples with more than 100 and less than 150 examples per label")
            self.count_ids()
            one_hot_encoder.fit([[id] for id in filtered_ids])
            id_to_one_hot = {id: one_hot_encoder.transform([[id]]).flatten() for id in filtered_ids}
            id_to_name = {item['id']: item['name'] for item in ontology if item['id'] in filtered_ids}
            return id_to_one_hot, id_to_name
        else:
            one_hot_encoder.fit([[id] for id in ids])
            id_to_one_hot = {id: one_hot_encoder.transform([[id]]).flatten() for id in ids}
            id_to_name = {item['id']: item['name'] for item in ontology}
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

        # define transform used to convert waveform to mel spectrogram with decibel scaling
        transform = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512),
            AmplitudeToDB() # mel spectrogram measures frequency amplitude over time, more informative for audio data
        )
        # convert waveform to mel spectrogram
        mel_spectrogram = transform(waveform)

        # display the waveform
        if not self.quiet:
            self.display_waveform(waveform, ytid, start_sec, end_sec)

        # add the channel dimension if not present (needed for CNN)
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        # resize the mel spectrogram to the target length
        if mel_spectrogram.shape[2] < self.target_length:   # pad if too short  
            padding = self.target_length - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        elif mel_spectrogram.shape[2] > self.target_length: # truncate if too long
            # TODO: loop audio file instead of truncating
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]

        # display the mel spectrogram
        if not self.quiet:
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
            lines = f.readlines()[3:]  # skip header lines
            for line in lines:
                parts = line.strip().split(', ')
                ytid = parts[0]
                start_sec = float(parts[1])
                end_sec = float(parts[2])

                # an entry can have multiple labels, separated by commas
                # currently only considers the first label with [0], TODO: investigate removal of [0]
                labels = [parts[3].split(',')[0].replace("\"", "")]

                self.data.append((ytid, start_sec, end_sec, labels))
    def data_labels(self):
        return [labels for _, _, _, labels in self.data]
    def __len__(self):
        return len(self.data)
    def get_label_count(self):
        return len(id_to_name)

# ====================== Model Architecture ======================
class AudioClassifier(nn.Module):   # CNN model, inheriting from pytorch neural network base class
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # todo: modify architecture? fine-tune?
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 16 * 75, 512)
        self.fc2 = nn.Linear(512, num_classes)      # output layer, note the output is num_classes logits, 
                                                    # where each index corresponding to a class

    def forward(self, x):
        # generate feature map
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        
        # flatten, then apply linear classifier
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # returns logits

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
    for epoch in range(num_epochs):
        model.train()       # set model to training mode
        running_loss = 0.0  # used to track loss of network over time
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for spectrogram, labels in progress_bar:    # calls __getitem__ of AudioDataset
            # move tensors to GPU
            spectrogram, labels = spectrogram.to(device), labels.to(device)

            optimizer.zero_grad()               # zero the parameter gradients
            outputs = model(spectrogram)        # forward pass
            
            labels = labels.squeeze(1)          # remove extra dimension

            loss = criterion(outputs, labels)   # compute the loss
            loss.backward()                     # backward pass
            optimizer.step()                    # optimize the network's weights

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
    source = 'balanced_train_segments'
    dataset = AudioDataset(f'./data/{source}.csv', f'./sounds_{source}', quiet=True, reduced=False, target_length=600)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AudioClassifier(num_classes=len(id_to_name))
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)        # stochastic gradient descent results in worse convergence
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=20, track='batch')