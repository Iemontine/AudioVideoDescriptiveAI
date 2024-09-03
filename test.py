import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from train import AudioDataset, AudioClassifier, label_encoder, id_to_encoded_label, id_to_name

def evaluate_model(model, dataloader, device):
    model.eval()  # evaluation mode
    all_preds = []
    all_labels = []
    pred_integers = []
    label_integers = []

    with torch.no_grad():  # disable gradient computation (?)
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            avg_preds = torch.mean(outputs, dim=1)

            # sigmoid to get probabilities and then threshold to get binary predictions
            preds = torch.sigmoid(outputs) >= 0.01
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            # convert binary predictions and labels to their integer representations
            for pred, label in zip(preds, labels):
                pred_integer = np.argmax(pred)
                label_integer = np.argmax(label)
                pred_integers.append(pred_integer)
                label_integers.append(label_integer)
                # uncomment this to view what the model predicted and the actual label
                # THIS IS DEFINITELY WRONG BECAUSE IT IS PREDICTING LABELS NOT EVEN IN THE REDUCED DATASET 
                # print(id_to_name[label_encoder.inverse_transform([pred_integer])[0]], id_to_name[label_encoder.inverse_transform([label_integer])[0]])

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    overall_accuracy = accuracy_score(label_integers, pred_integers)
    overall_f1 = f1_score(label_integers, pred_integers, average='weighted')
    
    return overall_accuracy, overall_f1, pred_integers, label_integers

if __name__ == "__main__":
    batch_size = 32
    target_length = 600
    model_path = './models/model_checkpoint_e20.pth'

    dataset = 'balanced_train_segments'
    test_dataset = AudioDataset(f'./data/{dataset}.csv', f'./sounds_{dataset}', target_length=target_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AudioClassifier(num_classes=len(id_to_encoded_label))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))

    accuracy, f1, pred_integers, label_integers = evaluate_model(model, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")