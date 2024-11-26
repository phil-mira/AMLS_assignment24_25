import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def model_evaluation_new(labels, predictions, model_name):
    """Evaluate the model using the sklearn classification report and produce a binary
    confusion matrix of the results for visualisation."""


    report = classification_report(labels, predictions)
    print(f'Classification report for {model_name}:\n{report}')
    
    plt.figure()
    plt.matshow(confusion_matrix(labels, predictions))
    plt.title('Confusion matrix for ' + model_name)
    
    # Add values to the boxes of the confusion matrix
    conf_matrix = confusion_matrix(labels,predictions)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')

    plt.gcf().set_size_inches(6, 6)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar()
    plt.show()