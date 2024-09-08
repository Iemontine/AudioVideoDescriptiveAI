import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from train import AudioDataset, AudioClassifier, one_hot_encoder
from sklearn.metrics import accuracy_score, f1_score

id_to_name = {}

def are_labels_related(label_a, label_b):
    # load ontology
    with open('./audio_classification/data/ontology.json') as f: ontology = json.load(f)

    child_ids = {}
    for item in ontology:
        child_ids[item['id']] = item['child_ids']
        
    # base case
    if label_a == label_b:
        return True
    
    def is_descendant(parent, target):
        if target in child_ids.get(parent, []):
            return True
        for child in child_ids.get(parent, []):
            if is_descendant(child, target):
                return True
        return False
    
    # recursive call
    return is_descendant(label_a, label_b)

def evaluate_model(model, epoch, dataloader, device, quiet=False):
    model.eval()
    tp = 0
    fp = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        for idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            sigmoid_outputs = torch.sigmoid(outputs)
            preds = (sigmoid_outputs > 0.5).int()
            if preds.sum().item() == 0:
                preds[0][sigmoid_outputs.argmax()] = 1

            preds = preds.cpu().numpy()
            trues = labels.cpu().numpy()

            pred_ids = []
            true_ids = []

            for pred, true in zip(preds, trues):
                # Find indices of active labels
                pred_active_indices = np.where(pred == 1)[0]
                true_active_indices = np.where(true == 1)[0]

                # Translate indices to their corresponding label IDs
                pred_labels = one_hot_encoder.inverse_transform(np.eye(len(pred))[pred_active_indices])
                true_labels = one_hot_encoder.inverse_transform(np.eye(len(true))[true_active_indices])

                # Flatten lists of IDs
                pred_ids.append(pred_labels.flatten())
                true_ids.append(true_labels.flatten())

                # print(pred_ids, true_ids)

                for pred_id in pred_ids[0]:
                    for true_id in true_ids[0]:
                        if are_labels_related(pred_id, true_id):
                            tp += 1
                            break
                        else:
                            fp += 1
                    progress_bar.set_postfix(accuracy=(tp / (tp + fp)) * 100)
            # pred_names = [[id_to_name[id] for id in pred] for pred in pred_ids]
            # true_names = [[id_to_name[id] for id in true] for true in true_ids]

    return (tp / (tp + fp)) * 100


if __name__ == "__main__":
    multi = True
    quiet = True   
    target_length = 600
    dataset = 'eval_segments'
    test_dataset = AudioDataset(f'./audio_classification/data/{dataset}.csv', f'./audio_classification/sounds_{dataset}', target_length=target_length)
    id_to_name = test_dataset.id_to_name
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = AudioClassifier(num_classes=test_dataset.get_label_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not multi:
        epoch = 5
        model = model.to(device)
        model_path = f'./audio_classification/models/model_checkpoint_e{epoch - 1}.pth'
        model.load_state_dict(torch.load(model_path))

        accuracy = evaluate_model(model, epoch, test_dataloader, device, quiet=quiet)
        print(f"Test Accuracy: {accuracy:.4f}")
        # print(f"Test F1 Score: {f1:.4f}")
    else:
        accuracy_scores = []
        f1_scores = []
        epochs = range(0, 10)

        for epoch in epochs:
            model = model.to(device)
            model_path = f'./audio_classification/models/model_checkpoint_e{epoch}.pth'
            model.load_state_dict(torch.load(model_path))

            accuracy = evaluate_model(model, epoch, test_dataloader, device, quiet=quiet)
            accuracy_scores.append(accuracy)

            print(f"Test Accuracy: {accuracy:.4f}")

        # Plotting accuracy and f1 scores
        plt.plot(epochs, accuracy_scores, label='Accuracy')
        plt.grid()
        plt.ylim(0, 1.1*max(accuracy_scores))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy per Epoch')
        plt.legend()
        plt.savefig('./audio_classification/plots/accuracy_f1.png')
        plt.show()