from typing import List

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class Classifier(nn.Module):
    def __init__(self, dim: int = 2, num_classes: int = 4):
        super().__init__()
        self.layer = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.layer(x)


def evaluate_accuracy(
    model,
    words: List[str],
    hiddens: torch.Tensor,
    values: bool = False,
    n_splits: int = 5,
):
    res_per_layer = {}
    torch.manual_seed(42)
    w2idx = {w: i for i, w in enumerate(words)}

    all_hiddens = []
    all_labels = []

    idx = int(not values)
    for layer_idx in range(model.model.config.num_hidden_layers):
        layer_hiddens = []
        layer_labels = []
        for i, word in enumerate(words):
            layer_hiddens.append(hiddens[i][layer_idx + idx])
            layer_labels += [
                torch.tensor([w2idx[word]] * len(hiddens[i][layer_idx + idx]))
            ]
        layer_hiddens = torch.cat(layer_hiddens, dim=0)
        layer_labels = torch.cat(layer_labels, dim=0)
        all_hiddens.append(layer_hiddens)
        all_labels.append(layer_labels)

    for layer_idx in tqdm(range(model.model.config.num_hidden_layers)):
        cur_res = {
            "train_losses": [],
            "train_accs": [],
            "test_losses": [],
            "test_accs": [],
        }
        X = all_hiddens[layer_idx].numpy()
        y = all_labels[layer_idx].numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            trainset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
            testset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )

            trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
            testloader = DataLoader(testset, batch_size=64, shuffle=False)

            clf = Classifier(dim=trainset[0][0].shape[-1], num_classes=len(words))
            ce = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(clf.parameters(), lr=1e-2)
            num_epochs = 1

            train_losses, train_accs = [], []
            test_losses, test_accs = [], []

            for epoch in range(num_epochs):
                clf.train()
                for inputs, targets in trainloader:
                    outs = clf(inputs)
                    loss = ce(outs, targets)
                    train_accs.append(
                        (outs.argmax(-1) == targets).float().mean().item()
                    )
                    train_losses.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                clf.eval()
                with torch.no_grad():
                    for inputs, targets in testloader:
                        outs = clf(inputs)
                        loss = ce(outs, targets)
                        test_accs.append(
                            (outs.argmax(-1) == targets).float().mean().item()
                        )
                        test_losses.append(loss.item())

            cur_res["train_losses"].append(np.mean(train_losses))
            cur_res["train_accs"].append(np.mean(train_accs))
            cur_res["test_losses"].append(np.mean(test_losses))
            cur_res["test_accs"].append(np.mean(test_accs))

        res_per_layer[layer_idx] = {
            "train_loss": np.mean(cur_res["train_losses"]),
            "train_loss_std": np.std(cur_res["train_losses"]),
            "train_acc": np.mean(cur_res["train_accs"]),
            "train_acc_std": np.std(cur_res["train_accs"]),
            "test_loss": np.mean(cur_res["test_losses"]),
            "test_loss_std": np.std(cur_res["test_losses"]),
            "test_acc": np.mean(cur_res["test_accs"]),
            "test_acc_std": np.std(cur_res["test_accs"]),
        }

    return res_per_layer
