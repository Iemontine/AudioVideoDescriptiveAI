import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from train import AudioDataset, AudioClassifier, one_hot_encoder

id_to_name = {}

import torch

def evaluate_model(model, dataloader, device, quiet=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # TODO: do this in such a way that supports multi-label classification
            preds = torch.sigmoid(outputs).round()
            preds = preds.cpu().numpy().astype(int).tolist()    # is there a way to do this without the cpu?
            preds = one_hot_encoder.inverse_transform(preds)

            trues = labels.cpu().numpy().astype(int).tolist()
            trues = one_hot_encoder.inverse_transform(trues)

            if not preds or not preds[0] or not trues or not trues[0]:
                continue

            if not quiet:
                print(f'{id_to_name[preds[0][0]]} --> {id_to_name[trues[0][0]]} --> {preds[0][0] == trues[0][0]}')
            all_preds.extend(preds)
            all_labels.extend(trues)

    # Manually calculate accuracy
    correct = sum([all_preds[i][0] == all_labels[i][0] for i in range(len(all_preds))])
    total = len(all_preds)
    overall_accuracy = correct / total if total > 0 else 0

    # Manually calculate F1 score
    tp, fp, fn = 0, 0, 0
    for i in range(len(all_preds)):
        if all_preds[i][0] == all_labels[i][0]:
            tp += 1
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # TODO: f1 is being calculated as equal to accuracy every time, investigate this

    return overall_accuracy, overall_f1


if __name__ == "__main__":
    multi = True
    quiet = True
    target_length = 600
    dataset = 'eval_segments'
    test_dataset = AudioDataset(f'./data/{dataset}.csv', f'./sounds_{dataset}', target_length=target_length)
    _, id_to_name = test_dataset.load_data()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = AudioClassifier(num_classes=test_dataset.get_label_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not multi:
        model = model.to(device)

        model_path = './models/model_checkpoint_e9.pth'
        model.load_state_dict(torch.load(model_path))

        accuracy, f1 = evaluate_model(model, test_dataloader, device, quiet=quiet)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    else:
        accuracy_scores = []
        f1_scores = []
        epochs = range(0, 20)

        for i in epochs:
            print(f"=== Model {i} ===")

            model = model.to(device)
            model_path = f'./models/model_checkpoint_e{i}.pth'
            model.load_state_dict(torch.load(model_path))

            accuracy, f1 = evaluate_model(model, test_dataloader, device, quiet=quiet)
            accuracy_scores.append(accuracy)
            f1_scores.append(f1)

            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1 Score: {f1:.4f}")

        # Plotting accuracy and f1 scores
        plt.plot(epochs, accuracy_scores, label='Accuracy')
        plt.plot(epochs, f1_scores, label='F1 Score')
        plt.grid()
        plt.ylim(0, 1.1*max(max(accuracy_scores), max(f1_scores)))
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Accuracy and F1 Score per Epoch')
        plt.legend()
        plt.savefig('./plots/accuracy_f1.png')
        plt.show()
