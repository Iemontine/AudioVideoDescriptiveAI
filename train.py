import json
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import torch.optim as optim
import os
import joblib
import matplotlib.pyplot as plt

# load ontology
with open('ontology.json') as f:
    ontology_data = json.load(f)

# extract all unique IDs
ids = []
for item in ontology_data:
    ids.append(item['id'])
    ids.extend(item['child_ids'])

# remove duplicates
ids = list(set(ids))

# save encoder, used to decode labels later
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(ids)
joblib.dump(label_encoder, "label_encoder.pkl")

# TODO: ensure one-hot encoded labels, can't tell if i did this correctly or not
label_map = dict(zip(ids, encoded_labels))

class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, label_map, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.load_csv(csv_file)     # initializes the dataset

    def __len__(self):
        return len(self.data)

    # load and preprocess the audio file
    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = os.path.join(self.audio_dir, item['ytid'] + '.flac')
        labels = torch.tensor(item['labels'], dtype=torch.long)

        # load the audio file
        y, sr = librosa.load(file_path, sr=16000, duration=10.0)
        start_sample = int(item['start_sec'] * sr)
        end_sample = int(item['end_sec'] * sr)
        y = y[start_sample:end_sample]

        # convert to log mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # convert to tensor, adding channel dimension
        log_mel_spec = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0) 

        # pad or truncate to a fixed length (for consistency in the input to nn)
        fixed_length = 224
        if log_mel_spec.shape[-1] > fixed_length:
            log_mel_spec = log_mel_spec[:, :, :fixed_length]    # truncate if too long
        else:
            pad_width = fixed_length - log_mel_spec.shape[-1]
            log_mel_spec = F.pad(log_mel_spec, (0, pad_width))  # pad if too short

        # create a one-hot encoded label vector (supposedly)
        one_hot_labels = torch.zeros(len(label_map), dtype=torch.float32)
        one_hot_labels[labels] = 1.0

        return log_mel_spec, one_hot_labels
    
    # intializes the dataset by loading the labels and audio files from the csv
    def load_csv(self, csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()[3:]  # skip header lines
            self.data = []
            for line in lines:
                # parse data
                parts = line.strip().split(', ')
                ytid = parts[0]
                start_sec = float(parts[1])
                end_sec = float(parts[2])
                pos_labels = [i.replace("\"", "") for i in parts[3].split(',')]

                # filter out labels not in the ontology
                labels = [label_map[label] for label in pos_labels if label in label_map]
                
                if labels:
                    file_path = os.path.join(self.audio_dir, ytid + '.flac')
                    # append to data if the file exists
                    if os.path.exists(file_path):
                        self.data.append({
                            'ytid': ytid,
                            'start_sec': start_sec,
                            'end_sec': end_sec,
                            'labels': labels
                        })
        print(f"Loaded {len(self.data)} samples from dataset")

class ClassifierNetwork(nn.Module):
    # define the network architecture
    def __init__(self, num_classes):
        super(ClassifierNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # forward pass through layers
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # initialize the dataset and dataloader
    csv_file = './data/balanced_train_segments.csv'
    audio_dir = './sounds_balanced_train_segments/'
    train_dataset = AudioDataset(csv_file=csv_file, audio_dir=audio_dir, label_map=label_map)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # initialize the model, loss function, and optimizer
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ClassifierNetwork(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # list to store loss values for each epoch
    loss_values = []
    # training loop
    num_epochs = 12
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)   # move tensors to GPU 
            optimizer.zero_grad()                                   # zero the parameter gradients
            outputs = model(inputs)                                 # forward pass
            loss = criterion(outputs, labels)                       # compute the loss
            loss.backward()                                         # backward pass
            optimizer.step()                                        # optimize the network's weights
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_values.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        scheduler.step()
        
    torch.save(model.state_dict(), "./models/model.pth")
    
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    # save plot to ./plots
    plt.savefig('./plots/training_loss.png')

if __name__ == "__main__":
    main()