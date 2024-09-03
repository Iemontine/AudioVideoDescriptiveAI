import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# load ontology
with open('ontology.json') as f: ontology = json.load(f)

# extract all unique IDs
ids = []
for item in ontology:
    ids.append(item['id'])

# save encoder, used to decode labels later
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(ids)

# 9/2/2024: reworked mapping encoded label to the ontology ID, meaning ytid<->label<->encoded_label (probably) works
id_to_encoded_label = {}
id_to_name = {}
for item in ontology:
    id_to_encoded_label[item['id']] = label_encoder.transform([item['id']])[0]
    id_to_name[item['id']] = item['name']

# ====================== Dataset Loader ======================
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, target_length):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.data = []

        self.parse_dataset(csv_file)

        # Remove samples without an existing audio file
        print(f"Loaded {len(self.data)} samples from dataset")
        self.data = [(ytid, start_sec, end_sec, labels) for ytid, start_sec, end_sec, labels in self.data if os.path.exists(os.path.join(self.audio_dir, f"{ytid}.flac"))]
        print(f"Filtered to {len(self.data)} samples with audio files")

        label_count = self.count_labels(silent=True)
        
        # Remove labels with less than 100 examples or more than 150 examples
        filtered_labels = [label for label, count in label_count.items() if 100 <= count <= 150]
        self.data = [(ytid, start_sec, end_sec, labels) for ytid, start_sec, end_sec, labels in self.data if any(label in filtered_labels for label in labels)]
        print(f"Filtered to {len(self.data)} samples with more than 100 and less than 150 examples per label")

        self.count_labels()

        # One-hot-encoded labels
        self.mlb = MultiLabelBinarizer(classes=list(id_to_encoded_label.keys()))
        self.mlb.fit(self.data_labels())

    def count_labels(self, silent=False):
        # Counts the occurrences of each label type
        label_count = {}
        for _, _, _, labels in self.data:
            for label in labels:
                if label not in label_count:    # register the label if it's not in the dictionary
                    label_count[label] = 1
                else:                           # increment the count if it's already in the dictionary
                    label_count[label] += 1

        # Output each label's true unencoded name and its count, sorted ascending by count
        if not silent:
            sorted_label_count = sorted(label_count.items(), key=lambda item: item[1])
            for label, count in sorted_label_count:
                label_id = label_encoder.inverse_transform([id_to_encoded_label[label]])[0]
                label_name = next(item["name"] for item in ontology if item["id"] == label_id)
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

    # preprocesses the audio data, returns mel spectrogram and one-hot encoded labels
    def __getitem__(self, idx):
        ytid, start_sec, end_sec, labels = self.data[idx]
        audio_path = os.path.join(self.audio_dir, f"{ytid}.flac")
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        if waveform.shape[0] > 1:    # Convert waveform to mono if it has multiple channels
            waveform = torch.mean(waveform, dim=0)
        transform = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512),
            AmplitudeToDB()
        )

        # TODO: investigate feature masking to improve quality of data? augment data?

        # discovered several waveforms with only one channel, this fixes that
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Plot the waveform
        print(f"Shape of waveform: {waveform.shape}")
        plt.figure(figsize=(10, 5))
        plt.plot(waveform[0].numpy())
        plt.title(f'Waveform of File: {ytid} of length {end_sec - start_sec} seconds')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.show()

        # convert waveform to mel spectrogram
        mel_spectrogram = transform(waveform)

        # add the channel dimension if not present (needed for CNN)
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

        # resize the mel spectrogram to the target length
        if mel_spectrogram.shape[2] < self.target_length:   # pad if too short
            padding = self.target_length - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        elif mel_spectrogram.shape[2] > self.target_length: # truncate if too long
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]

        # Graph the mel spectrogram
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

        # convert labels to float tensor
        labels = self.mlb.transform([labels])[0]  # Get the binary labels
        labels = torch.FloatTensor(labels)  # Convert to float tensor

        return mel_spectrogram, labels

# ====================== Model Architecture ======================
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 16 * 75, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # generate feature map
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        
        # TODO: check feature map or output
        
        # flatten, then apply linear classifier, TODO: investigate separating this?
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
    plt.ylim(0, 1)
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
    if track == 'batch' or track == 'epoch':
        init_plot()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for inputs, labels in progress_bar:
            # move tensors to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                                   # zero the parameter gradients
            outputs = model(inputs)                                 # forward pass
            loss = criterion(outputs, labels)                       # compute the loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # optimize the network's weights

            # track statistics (TODO: not even sure this is working correctly, loss converges to 0 but model can't predict anything)
            batch_loss = loss.item()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=batch_loss)
            if track == 'batch':
                update_plot(batch_loss, track=track)
        scheduler.step()
        if track == 'epoch':
            update_plot(batch_loss, track=track)
        torch.save(model.state_dict(), f'./models/model_checkpoint_e{epoch}.pth')
    plt.savefig('batch_loss_plot.png')

# ====================== Main ======================
if __name__ == "__main__":
    source = 'balanced_train_segments'
    dataset = AudioDataset(f'./data/{source}.csv', f'./sounds_{source}', target_length=600)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AudioClassifier(num_classes=len(id_to_encoded_label))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10, track='none')