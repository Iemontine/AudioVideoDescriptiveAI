from train import AudioClassifier
from train import label_map
import torch
import librosa
import torch.nn.functional as F
import numpy as np
import json
import joblib

def predict_audio(model, audio_path, fixed_length=224):
    # load the audio file
    y, sr = librosa.load(audio_path, sr=16000, duration=10.0)
    
    # convert to log mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # convert to tensor, adding channel dimension
    log_mel_spec = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # pad or truncate to a fixed length (for consistency in the input to nn)
    fixed_length = 224
    if log_mel_spec.shape[-1] > fixed_length:
        log_mel_spec = log_mel_spec[:, :, :fixed_length]    # truncate if too long
    else:
        pad_width = fixed_length - log_mel_spec.shape[-1]
        log_mel_spec = F.pad(log_mel_spec, (0, pad_width))  # pad if too short
    
    # make prediction using the model
    model.eval()
    with torch.no_grad():
        output = model(log_mel_spec)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# load ontology
with open('ontology.json') as f:
    ontology_data = json.load(f)

model = AudioClassifier(num_classes=len(label_map))
model.load_state_dict(torch.load("./models/model.pth"))
audio_path = "./sound/EkmKuR0RkZM.wav"  # TODO: currently loading something from the trainnig set, change this
predicted_label = predict_audio(model, audio_path)

# load the label encoder to find the original label
label_encoder = joblib.load("./models/label_encoder.pkl")
original_label = label_encoder.inverse_transform([predicted_label])[0]
id_to_name = {item['id']: item['name'] for item in ontology_data}
predicted_name = id_to_name.get(original_label, "Unknown Label")
print(f"Predicted label: {original_label}")
print(f"Predicted class: {predicted_name}")