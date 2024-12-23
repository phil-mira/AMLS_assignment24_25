import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def model_evaluation(labels, predictions, model_name, dataset_type):
    """Evaluate the model using the sklearn classification report and produce a confusion matrix of the results for visualisation."""

    report = classification_report(labels, predictions)
    with open(f'Reports/Classification_report_for_{model_name}_on_{dataset_type}_data.txt', 'w') as f:
        f.write(f'Classification report for {model_name} on {
                dataset_type} data:\n{report}')

    fig, ax = plt.subplots(figsize=(10, 10))
    TITLE_FONT_SIZE = {"size": "20"}
    LABEL_FONT_SIZE = {"size": "40"}
    LABEL_SIZE = 20

    conf_matrix = np.array(confusion_matrix(predictions, labels))

    conf_matrix_norm = np.array(confusion_matrix(
        predictions, labels, normalize='true'))

    group_counts = ['Count: {0:0.0f}'.format(
        value) for value in conf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value)
                         for value in conf_matrix_norm.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]

    num_classes = 8
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix_norm, annot=labels, annot_kws={
                'size': 10}, fmt='', cmap='Blues', vmax=1.0, vmin=0.0)

    # Titles, axis labels, etc.
    title = f"Confusion Matrix for {model_name}\n({dataset_type} data)"
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(title, fontdict=TITLE_FONT_SIZE)
    ax.set_xlabel("Actual", fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel("Predicted", fontdict=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    plt.savefig(f"Figures/Confusion Matrix for {
                model_name} on {dataset_type} data.png")
    plt.close(fig)

    return report


def display_incorrect_images(data, labels, predictions, model_name, data_type):
    """Display 2 images from each class the results that were correctly predicted
       and display 2 images from each class that were incorrectly predicted."""

    fig, ax = plt.subplots(8, 4)
    fig.suptitle(f'Predictions for {model_name} on '
                 f'{data_type} data', fontsize=40)
    fig.set_size_inches(20, 40)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.subplots_adjust(top=0.93)
    SIZE_SUBPLOT = 30

    incorrect_predictions_all = np.where(predictions != labels)[0]
    correct_predictions_all = np.where(predictions == labels)[0]

    for class_label in range(8):

        incorrect_predictions = [
            i for i in incorrect_predictions_all if labels[i] == class_label]
        correct_predictions = [
            i for i in correct_predictions_all if labels[i] == class_label]

        try:
            incorrect_predictions = np.random.choice(
                incorrect_predictions, 2, replace=True)
        # Ensures if there are not enough incorrect predictions, it will randomly sample from the correct predictions
        except:
            incorrect_predictions = np.random.choice(
                correct_predictions, 2, replace=True)

        try:
            correct_predictions = np.random.choice(
                correct_predictions, 2, replace=True)
        # Ensures if there are not enough correct predictions, it will randomly sample from the incorrect predictions
        except:
            correct_predictions = np.random.choice(
                incorrect_predictions, 2, replace=True)

        try:
            for j, jdx in enumerate(correct_predictions):
                ax[class_label][j].imshow(
                    data[jdx].transpose(1, 2, 0), cmap='gray')
                ax[class_label][j].set_title(f"True: {labels[jdx]}\nPredicted: {
                                             predictions[jdx]}", fontsize=SIZE_SUBPLOT)
                ax[class_label][j].axis('off')
                ax[class_label][j].set_aspect('auto')
        except:
            for j, jdx in enumerate(correct_predictions):
                ax[class_label][j].imshow(data[jdx], cmap='gray')
                ax[class_label][j].set_title(f"True: {labels[jdx]}\nPredicted: {
                                             predictions[jdx]}", fontsize=SIZE_SUBPLOT)
                ax[class_label][j].axis('off')
                ax[class_label][j].set_aspect('auto')

        try:
            for i, idx in enumerate(incorrect_predictions):
                ax[class_label][i +
                                2].imshow(data[idx].transpose(1, 2, 0), cmap='gray')
                ax[class_label][i + 2].set_title(f"True: {labels[idx]}\nPredicted: {
                                                 predictions[idx]}", fontsize=SIZE_SUBPLOT)
                ax[class_label][i + 2].axis('off')
                ax[class_label][i + 2].set_aspect('auto')
        except:
            for i, idx in enumerate(incorrect_predictions):
                ax[class_label][i + 2].imshow(data[idx], cmap='gray')
                ax[class_label][i + 2].set_title(f"True: {labels[idx]}\nPredicted: {
                                                 predictions[idx]}", fontsize=SIZE_SUBPLOT)
                ax[class_label][i + 2].axis('off')
                ax[class_label][i + 2].set_aspect('auto')

    plt.savefig(f"Figures/Incorrect Predictions for {
                model_name} on {data_type} data.png")
    plt.close(fig)

    return None


def plot_roc_curve_multi(labels, predictions, model_name, data_type):
    """Plot the ROC curve for the model using one-vs-all strategy for multiclass classification."""
    # Binarize the labels for multiclass ROC
    labels = label_binarize(labels, classes=np.unique(labels))
    n_classes = labels.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(10, 10))
    TITLE_FONT_SIZE = {"size": "30"}
    LABEL_FONT_SIZE = {"size": "30"}
    LABEL_SIZE = 20

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                   'red', 'green', 'blue', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel('True Positive Rate', fontdict=LABEL_FONT_SIZE)
    ax.set_title(f'ROC Curve for {model_name}\n'
                 f'{data_type} data', fontdict=TITLE_FONT_SIZE)
    ax.legend(loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    plt.savefig(f"Figures/ROC Curve for {model_name} on {data_type} data.png")
    plt.close(fig)

    return None


def train(model, device, train_loader, optimizer, criterion):
    """Train the model on the training data."""
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        loss = loss.item()

    return loss, model


def test(model, device, test_loader, criterion):
    """Test the model on the validation or test data."""
    model.eval()
    test_losses = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)
            test_loss = criterion(output, target.view(-1)).item()
            test_losses.append(test_loss)

            predictions = output.argmax(
                dim=1, keepdim=True).view(-1).cpu().numpy()
            labels = target.view(-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

        avg_f1_score = f1_score(
            all_labels, all_predictions, average='weighted')

    return np.mean(test_losses), avg_f1_score


def initialize_weights(model):
    """Initialize the weights of the model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')

    return model


def train_validation(model, device, train_loader, val_loader, optimizer, criterion, n_epochs=20):
    """Plot the training and validation curve for the model."""

    # Train the model and record the losses
    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []
    model = initialize_weights(model)

    # Train the model and record the losses
    for _ in range(n_epochs):
        train_loss, _ = train(model, device, train_loader,
                              optimizer, criterion)
        val_loss, f1_score = test(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses, f1_score, model


def plot_train_validation(model, device, val_loader, train_losses, val_losses, n_epochs):

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig('Figures/Training and Validation Loss per Epoch.png')
    plt.close()

    # Plot the confusion matrix for the validation data
    val_data, val_labels = next(iter(val_loader))
    val_data = val_data.to(device)
    val_labels = val_labels.to(device)
    val_predictions = model.forward(val_data, ).detach().cpu().numpy()
    val_predictions = np.argmax(val_predictions, axis=1)

    # Check the data type of
    if isinstance(val_data, torch.Tensor):
        val_data = val_data.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()
    if isinstance(val_predictions, torch.Tensor):
        val_predictions = val_predictions.cpu().numpy()

    model_evaluation(val_labels, val_predictions, 'CNN', 'Validation')


def test_and_plot(model, device, test_loader):

    # Plot the confusion matrix for the validation data
    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    test_pred_prob = model.forward(test_data).detach().cpu().numpy()
    test_predictions = np.argmax(test_pred_prob, axis=1)

    # Check the data type of
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    if isinstance(test_predictions, torch.Tensor):
        test_predictions = test_predictions.cpu().numpy()

    model_evaluation(test_labels, test_predictions, 'CNN (Tuned)', 'Test')

    plot_roc_curve_multi(test_labels, test_pred_prob, 'CNN (Tuned)', 'Test')

    display_incorrect_images(test_data, test_labels,
                             test_predictions, 'CNN (Tuned)', 'Test')


class BloodMnistCNN(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64, fc1_out=128):
        super(BloodMnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, conv1_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(conv2_out * 7 * 7, fc1_out)
        self.fc2 = nn.Linear(fc1_out, 8)
        self.conv2_out = conv2_out

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, self.conv2_out * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


class BloodMnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
