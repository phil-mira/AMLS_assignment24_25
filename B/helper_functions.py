import numpy as np
import pandas as pd
import seaborn as sns
import torch

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
    print(f'Classification report for {model_name} on {dataset_type} data:\n{report}')

    fig, ax = plt.subplots(figsize=(10, 10))
    TITLE_FONT_SIZE = {"size":"40"}
    LABEL_FONT_SIZE = {"size":"40"}
    LABEL_SIZE = 20

    conf_matrix = np.array(confusion_matrix(predictions, labels))
    
    conf_matrix_norm = np.array(confusion_matrix(predictions, labels, normalize='true'))

    group_counts = ['Count: {0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in conf_matrix_norm.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]

    num_classes = 8
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix_norm, annot=labels, annot_kws={'size': 10}, fmt='', cmap='Blues', vmax=1.0, vmin=0.0)
    
    # Titles, axis labels, etc.
    title = f"Confusion Matrix for {model_name}\n({dataset_type} data)\n"
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(title, fontdict=TITLE_FONT_SIZE)
    ax.set_xlabel("Actual", fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel("Predicted", fontdict=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    #plt.savefig(f"Confusion Matrix for {model_name} on {dataset_type} data.png")

    return report


def display_incorrect_images(data, labels, predictions, model_name, data_type):
    """Display 2 images from each class the results that were correctly predicted
       and display 2 images from each class that were incorrectly predicted."""

    fig, ax = plt.subplots(8, 4)
    fig.suptitle(f"Predictions for {model_name} on {data_type} data\n", fontsize=40)
    fig.set_size_inches(20, 40)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.subplots_adjust(top=0.95)
    SIZE_SUBPLOT = 20

    incorrect_predictions_all = np.where(predictions != labels.T)[1]
    correct_predictions_all = np.where(predictions == labels.T)[1]

    for class_label in range(8):
        
        incorrect_predictions = [i for i in incorrect_predictions_all if labels[i] == class_label]
        correct_predictions = [i for i in correct_predictions_all if labels[i] == class_label]

        incorrect_predictions = np.random.choice(incorrect_predictions, 2, replace=True)
        correct_predictions = np.random.choice(correct_predictions, 2, replace=True)

        for j, jdx in enumerate(correct_predictions):
            ax[class_label][j].imshow(data[jdx], cmap='gray')
            ax[class_label][j].set_title(f"True: {labels[jdx]}\nCorrect Predicted: {predictions[jdx]}", fontsize=SIZE_SUBPLOT)
            ax[class_label][j].axis('off')
            ax[class_label][j].set_aspect('auto')

        for i, idx in enumerate(incorrect_predictions):
            ax[class_label][i + 2].imshow(data[idx], cmap='gray')
            ax[class_label][i + 2].set_title(f"True: {labels[idx]}\nIncorrect Predicted: {predictions[idx]}", fontsize=SIZE_SUBPLOT)
            ax[class_label][i + 2].axis('off')
            ax[class_label][i + 2].set_aspect('auto')

    #plt.savefig(f"Incorrect Predictions for {model_name} on {data_type} data.png")

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

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel('True Positive Rate', fontdict=LABEL_FONT_SIZE)
    ax.set_title(f'ROC Curve for {model_name} on {data_type} data \n', fontdict=TITLE_FONT_SIZE)
    ax.legend(loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    #plt.savefig(f"ROC Curve for {model_name} on {data_type} data.png")

    return None


def train(model, device, train_loader, optimizer, criterion):
    """Train the model on the training data."""

    for data, target in train_loader:

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        loss = loss.item()

        return loss
        
            

def test(model, device, test_loader, criterion, mode='Validation'):
    """Test the model on the validation or test data."""
    model.eval()
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target.view(-1)).item()
            test_losses.append(test_loss)
           

    return np.mean(test_losses)
            


def train_validation(model, device, train_loader, val_loader, optimizer, criterion, n_epochs=20):
    """Plot the training and validation curve for the model."""

    # Train the model and record the losses
    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Train the model and record the losses
    for epoch in range(n_epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss = test(model, device, val_loader, criterion)    

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses

def plot_train_validation(model, device, val_loader, train_losses, val_losses, n_epochs=20):

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.show()

    # Plot the confusion matrix for the validation data
    val_data, val_labels = next(iter(val_loader))
    val_data = val_data.to(device)
    val_labels = val_labels.to(device)
    val_predictions = model.forward(val_data).detach().cpu().numpy()
    val_predictions = np.argmax(val_predictions, axis=1)

    # Check the data type of 
    if isinstance(val_data, torch.Tensor):
        val_data = val_data.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()
    if isinstance(val_predictions, torch.Tensor):
        val_predictions = val_predictions.cpu().numpy()


    model_evaluation(val_labels, val_predictions, 'Model', 'Validation')

 

    # Display the ROC curve for the model
    #plot_roc_curve_multi(val_labels, val_predictions, 'Model', 'Validation')

        
  
