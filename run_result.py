
import os
import random
import time
import torch
import numpy as np
import scipy.io as sio
import argparse
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from src.kan import KAN


def load_brca_data(direction, dataset_name, ratio):
    # Load data from .mat file
    target_path = f"{direction}/{dataset_name}.mat"
    data = sio.loadmat(target_path)

    # Extract features and labels
    features = data['X']
    num_views = len(features[0])
    views_data = [np.asarray(features[0][i]).astype(np.float32) for i in range(num_views)]
    labels = data['Y'].flatten()
    labels = label_from_zero(labels)
    shuffle_index = data['idx']
    num_classes = len(set(labels))

    # Split data into train, validation, and test sets
    shuffle_index = shuffle_index.astype(np.int32).reshape(-1)
    train_size, val_size = int(len(views_data[0]) * ratio), int(len(views_data[0]) * (ratio + 0.1))

    # Create empty lists to store train, validation, and test sets for each view
    train_data = []
    val_data = []
    test_data = []

    for view in views_data:
        train_data.append(torch.FloatTensor(view[shuffle_index[0:train_size]]))
        val_data.append(torch.FloatTensor(view[shuffle_index[train_size:val_size]]))
        test_data.append(torch.FloatTensor(view[shuffle_index[val_size:]]))

    labels_train = torch.LongTensor(labels[shuffle_index[0:train_size]].astype(np.int64))
    labels_val = torch.LongTensor(labels[shuffle_index[train_size:val_size]].astype(np.int64))
    labels_test = torch.LongTensor(labels[shuffle_index[val_size:]].astype(np.int64))

    return train_data, val_data, test_data, labels_train, labels_val, labels_test, num_classes


def label_from_zero(labels):
    min_num = min(set(labels))
    return labels - min_num


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_classification_results(labels_true, labels_pred, labels_prob):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    MACRO_P = metrics.precision_score(labels_true, labels_pred, average='macro')
    MACRO_R = metrics.recall_score(labels_true, labels_pred, average='macro')
    MACRO_F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    MICRO_F1 = metrics.f1_score(labels_true, labels_pred, average='micro')
    if labels_prob.shape[1] == 2:
        AUC = roc_auc_score(labels_true, labels_prob[:, 1])
    else:
        AUC = roc_auc_score(labels_true, labels_prob, multi_class='ovr', average='macro')
    return ACC, MACRO_P, MACRO_R, MACRO_F1, MICRO_F1, AUC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1009, help='Number of seed.')
    parser.add_argument('--n_repeated', type=int, default=10, help='Number of repeated experiments')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='2', help='The number of cuda device.')
    parser.add_argument('--direction', type=str, default='/home/hjg/set/dataset/', help='direction of datasets')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=[
        "LIHC-3MMIS"], help='List of datasets used for training/testing')
    parser.add_argument('--ratio', type=float, default=0.8, help='percentage training samples.')
    parser.add_argument('--batch_size', type=int, default=64, help='The number of batch size.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='The hidden dimension of KAN')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=180, help='The number of training epochs')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device


    for dataset_name in args.dataset_names:
        results_filename = f"{dataset_name}_results.txt"
        results_file = open(results_filename, "a+", encoding="utf-8")
        for ratio in np.arange(0.8, 0.9, 0.1):
            args.ratio = ratio
            print(f"ratio: {ratio}", file=results_file)
            all_ACC = []
            all_MaP = []
            all_MaR = []
            all_MaF = []
            all_MICRO_F1 = []
            all_AUC = []
            all_Time = []

            for _ in range(args.n_repeated):
                train_data, val_data, test_data, labels_train, labels_val, labels_test, num_classes = \
                    load_brca_data(direction=args.direction, dataset_name=dataset_name, ratio=args.ratio)

                # Calculate input dimension
                input_dim = sum(
                    [view.shape[1] for view in train_data])  # Sum of all views' feature dimensions

                # Compute the weights for each class
                class_sample_count = np.array(
                    [len(np.where(labels_train == t)[0]) for t in np.unique(labels_train)])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in labels_train])
                samples_weight = torch.from_numpy(samples_weight)
                sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                len(samples_weight),
                                                replacement=True)

                # Create a TensorDataset that supports multiple views
                trainDataset = TensorDataset(*train_data, labels_train)
                trainLoader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0,
                                         sampler=sampler)

                valDataset = TensorDataset(*val_data, labels_val)
                valLoader = DataLoader(dataset=valDataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=0)
                testDataset = TensorDataset(*test_data, labels_test)
                testLoader = DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=0)

                # Define model
                model = KAN([input_dim, args.hidden_dim, num_classes])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
                criterion = nn.CrossEntropyLoss()
                train_start_time = time.time()
                # Training phase
                model.train()

                for epoch in range(args.num_epochs):
                    with tqdm(trainLoader) as pbar:
                        for batch_data in pbar:
                            # Unpack the batch data
                            views = batch_data[:-1]  # All views data
                            labels = batch_data[-1]  # Labels

                            # Move data to device
                            views = [view.to(device) for view in views]
                            labels = labels.to(device)

                            optimizer.zero_grad()
                            input_data = torch.cat(views, dim=1)  # Concatenate all views' inputs
                            output = model(input_data)  # Pass concatenated input to the model
                            loss = criterion(output, labels)
                            loss.backward()
                            optimizer.step()

                            batch_accuracy = accuracy(output, labels)
                            pbar.set_postfix(loss=loss.item(), accuracy=batch_accuracy.item(),
                                             lr=optimizer.param_groups[0]['lr'])

                    scheduler.step()
                train_time = time.time() - train_start_time
                train_time_per_epoch = train_time / args.num_epochs

                # Validation phase
                model.eval()
                test_loss = 0
                test_accuracy = 0
                all_labels = []
                all_preds = []
                all_probs = []
                with torch.no_grad():
                    for batch_data in testLoader:
                        # Unpack the batch data
                        views = batch_data[:-1]  # All views data
                        labels_test = batch_data[-1]  # Labels

                        # Move data to device
                        views = [view.to(device) for view in views]
                        labels_test = labels_test.to(device)

                        # Concatenate all views' inputs
                        input_data = torch.cat(views, dim=1)

                        output = model(input_data)  # Pass concatenated input to the model


                        test_loss += criterion(output, labels_test).item()
                        test_accuracy += (output.argmax(dim=1) == labels_test).float().mean().item()
                        all_labels.extend(labels_test.cpu().numpy())
                        all_preds.extend(output.argmax(dim=1).cpu().numpy())
                        all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())


                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)
                all_probs = np.array(all_probs)
                ACC, MACRO_P, MACRO_R, MACRO_F1, MICRO_F1, AUC = get_classification_results(all_labels,
                                                                                            all_preds,
                                                                                            all_probs)

                all_ACC.append(ACC)
                all_MaP.append(MACRO_P)
                all_MaR.append(MACRO_R)
                all_MaF.append(MACRO_F1)
                all_MICRO_F1.append(MICRO_F1)
                all_AUC.append(AUC)
                all_Time.append(train_time_per_epoch)

            # Write results to file
            results_file.write(f"ACC MaP MaR MaF Time \n")
            results_file.write(f"{np.mean(all_ACC) * 100:.2f}+{np.std(all_ACC) * 100:.2f}\n")
            results_file.write(f"{np.mean(all_MaP) * 100:.2f}+{np.std(all_MaP) * 100:.2f}\n")
            results_file.write(f"{np.mean(all_MaR) * 100:.2f}+{np.std(all_MaR) * 100:.2f}\n")
            results_file.write(f"{np.mean(all_MaF) * 100:.2f}+{np.std(all_MaF) * 100:.2f}\n")
            results_file.write(f"{np.mean(all_Time):.2f}+{np.std(all_Time):.2f}\n")
            results_file.write("\n")

        results_file.close()




