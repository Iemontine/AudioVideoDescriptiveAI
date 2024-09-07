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

# load ontology
with open('ontology.json') as f: ontology = json.load(f)

child_ids = {}
for item in ontology:
    child_ids[item['id']] = item['child_ids']

def evaluate_model(model, epoch, dataloader, device, quiet=False):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        for idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            sigmoid_outputs = torch.sigmoid(outputs)
            preds = (sigmoid_outputs > 0.1).int()
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

            # matched = False
            # for pred_id in pred_ids[0]:
            #     # Check if pred_id is in the true_ids or in any of its child_ids
            #     for true_id in true_ids[0]:
            #         if pred_id == true_id or pred_id in child_ids.get(true_id, []):
            #             tp += 1
            #             # print(tp/(tp+fp))
            #             # print(True)
            #             # print(True, pred_ids[0], true_ids[0], tp)
            #             matched = True  # Set the flag since we've found a match
            #             break
            #         print(pred_id, "-->", true_ids[0], child_ids.get(true_id, []))
            #     if matched:
            #         break  # Stop further comparisons once we have a match

            # if not matched:
            #     # print(False, pred_ids[0], true_ids[0], fp)
            #     fp += 1
            #     print(False)
            #     # print(tp/(tp+fp))


            pred_names = [[id_to_name[id] for id in pred] for pred in pred_ids]
            true_names = [[id_to_name[id] for id in true] for true in true_ids]

            if not quiet:
                print(f"Predicted: {pred_names}")
                print(f"True: {true_names}\n")

    # return overall_accuracy, overall_f1


if __name__ == "__main__":
    multi = False
    quiet = True   
    target_length = 600
    dataset = 'balanced_train_segments'
    test_dataset = AudioDataset(f'./data/{dataset}.csv', f'./sounds_{dataset}', target_length=target_length)
    id_to_name = test_dataset.id_to_name
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = AudioClassifier(num_classes=test_dataset.get_label_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not multi:
        epoch = 5
        model = model.to(device)
        model_path = f'./models/model_checkpoint_e{epoch - 1}.pth'
        model.load_state_dict(torch.load(model_path))

        accuracy, f1 = evaluate_model(model, epoch, test_dataloader, device, quiet=quiet)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    else:
        accuracy_scores = []
        f1_scores = []
        epochs = range(0, 4)

        for epoch in epochs:
            model = model.to(device)
            model_path = f'./models/model_checkpoint_e{epoch}.pth'
            model.load_state_dict(torch.load(model_path))

            accuracy, f1 = evaluate_model(model, epoch, test_dataloader, device, quiet=quiet)
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