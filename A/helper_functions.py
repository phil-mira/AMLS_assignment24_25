import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import seaborn as sns


def model_evaluation(labels, predictions, model_name, dataset_type):
    """Evaluate the model using the sklearn classification report and produce a binary
    confusion matrix of the results for visualisation."""

    report = classification_report(labels, predictions)
    with open(f'Reports/classification_report_{model_name}_{dataset_type}.txt', 'w') as f:
        f.write(f'Classification report for {model_name} on {
                dataset_type} data:\n{report}')

    fig, ax = plt.subplots(figsize=(10, 10))
    TITLE_FONT_SIZE = {"size": "20"}
    LABEL_FONT_SIZE = {"size": "20"}
    LABEL_SIZE = 40

    conf_matrix = np.array(confusion_matrix(predictions, labels))

    conf_matrix_norm = np.array(confusion_matrix(
        predictions, labels, normalize='true'))

    group_counts = ['Count: {0:0.0f}'.format(
        value) for value in conf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value)
                         for value in conf_matrix_norm.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix_norm, annot=labels, fmt='',
                cmap='Blues', vmax=1.0, vmin=0.0)

    # Titles, axis labels, etc.
    title = (f'Task A: Confusion Matrix for {model_name} \n'
             f'({dataset_type} data)\n')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(title, fontdict=TITLE_FONT_SIZE)
    ax.set_xlabel("Actual", fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel("Predicted", fontdict=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    plt.savefig(
        f'Figures/Confusion Matrix for {model_name} on {dataset_type} data.png')
    plt.close(fig)

    return report


def display_incorrect_images(data, labels, predictions, model_name, data_type):
    """Display 2 images from each class the results that were correctly predicted
       and display 2 images from each class that were incorrectly predicted."""
    incorrect_predictions_0 = np.where((predictions != labels) & (labels == 0))
    correct_predictions_0 = np.where((predictions == labels) & (labels == 0))
    incorrect_predictions_1 = np.where((predictions != labels) & (labels == 1))
    correct_predictions_1 = np.where((predictions == labels) & (labels == 1))

    incorrect_predictions_0 = np.random.choice(
        incorrect_predictions_0[0], min(4, len(incorrect_predictions_0[0])))
    correct_predictions_0 = np.random.choice(
        correct_predictions_0[0], min(4, len(correct_predictions_0[0])))
    incorrect_predictions_1 = np.random.choice(
        incorrect_predictions_1[0], min(4, len(incorrect_predictions_1[0])))
    correct_predictions_1 = np.random.choice(
        correct_predictions_1[0], min(4, len(correct_predictions_1[0])))

    fig, ax = plt.subplots(4, 4)
    fig.suptitle(f'Task A: Predictions for {model_name} on {data_type} data\n',
                 fontsize=40)
    fig.set_size_inches(20, 20)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.subplots_adjust(top=0.85)
    SIZE_SUBPLOT = 30

    for j, jdx in enumerate(correct_predictions_0):
        ax[0][j].imshow(data.iloc[jdx].values.reshape(28, 28), cmap='gray')
        ax[0][j].set_title(f'True: {labels[jdx]}\nPredicted: {predictions[jdx]}',
                           fontsize=SIZE_SUBPLOT)
        ax[0][j].axis('off')
        ax[0][j].set_aspect('auto')

    for i, idx in enumerate(incorrect_predictions_0):
        ax[1][i].imshow(data.iloc[idx].values.reshape(28, 28), cmap='gray')
        ax[1][i].set_title(f'True: {labels[idx]}\nPredicted: {predictions[idx]}',
                           fontsize=SIZE_SUBPLOT)
        ax[1][i].axis('off')
        ax[1][i].set_aspect('auto')

    for j, jdx in enumerate(correct_predictions_1):
        ax[2][j].imshow(data.iloc[jdx].values.reshape(28, 28), cmap='gray')
        ax[2][j].set_title(f'True: {labels[jdx]}\nPredicted: {predictions[jdx]}',
                           fontsize=SIZE_SUBPLOT)
        ax[2][j].axis('off')
        ax[2][j].set_aspect('auto')

    for i, idx in enumerate(incorrect_predictions_1):
        ax[3][i].imshow(data.iloc[idx].values.reshape(28, 28), cmap='gray')
        ax[3][i].set_title(f'True: {labels[idx]}\nPredicted: {predictions[idx]}',
                           fontsize=SIZE_SUBPLOT)
        ax[3][i].axis('off')
        ax[3][i].set_aspect('auto')
    plt.savefig(
        f'Figures/Incorrect Predictions for {model_name} on {data_type} data.png')
    plt.close(fig)

    return None


def plot_roc_curve(labels, predictions, model_name, data_type):
    """Plot the ROC curve for the model."""
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 10))
    TITLE_FONT_SIZE = {'size': "20"}
    LABEL_FONT_SIZE = {'size': "20"}
    LABEL_SIZE = 20

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel('True Positive Rate', fontdict=LABEL_FONT_SIZE)
    ax.set_title(f'Task A\n ROC Curve for {model_name} on {data_type} data \n',
                 fontdict=TITLE_FONT_SIZE)
    ax.legend(loc='lower right')
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    plt.savefig(f'Figures/ROC Curve for {model_name} on {data_type} data.png')
    plt.close(fig)

    return None
