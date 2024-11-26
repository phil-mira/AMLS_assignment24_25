import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns



def model_evaluation(labels, predictions, model_name, dataset_type):
    """Evaluate the model using the sklearn classification report and produce a binary
    confusion matrix of the results for visualisation."""



    report = classification_report(labels, predictions)
    print(f'Classification report for {model_name} on {dataset_type} data:\n{report}')

    fig, ax = plt.subplots(figsize=(10, 10))
    TITLE_FONT_SIZE = {"size":"40"}
    LABEL_FONT_SIZE = {"size":"40"}
    LABEL_SIZE = 40

    conf_matrix = np.array(confusion_matrix(predictions, labels))
    
    conf_matrix_norm = np.array(confusion_matrix(predictions, labels, normalize='true'))

    group_counts = ['Count: {0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in conf_matrix_norm.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix_norm, annot=labels, fmt='', cmap='Blues', vmax=1.0, vmin=0.0)
    
    # Titles, axis labels, etc.
    title = f"Confusion Matrix for {model_name}\n({dataset_type} data)\n"
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(title, fontdict=TITLE_FONT_SIZE)
    ax.set_xlabel("Actual", fontdict=LABEL_FONT_SIZE)
    ax.set_ylabel("Predicted", fontdict=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    

    
    