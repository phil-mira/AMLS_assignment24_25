import pickle
import helper_functions
import importlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as accuracy
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from helper_functions import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# Set the working directory to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the tuning parameter to True to tune the models
tuning = False
# Set the base parameter to True to run the baseline models
base = False

# Load the breastmnist dataset
dfile = os.path.join(os.path.dirname(__file__), '../Datasets/breastmnist.npz')
if not os.path.exists(dfile):
    raise FileNotFoundError(
        f"The file {dfile} does not exist. Please check the file path.")
data_breastmnist = np.load(dfile)


# Reformat the data
breastmnist_train_images = data_breastmnist['train_images']
breastmnist_train_labels = data_breastmnist['train_labels']

breastmnist_val_images = data_breastmnist['val_images']
breastmnist_val_labels = data_breastmnist['val_labels']

breastmnist_test_images = data_breastmnist['test_images']
breastmnist_test_labels = data_breastmnist['test_labels']


# Summary statistics of the labels
unique, counts = np.unique(breastmnist_train_labels, return_counts=True)
summary_statistics_train_data = dict(zip(unique, counts))
ax, fig = plt.subplots()
plt.bar(summary_statistics_train_data.keys(),
        summary_statistics_train_data.values())
plt.title("Summary Statistics of the Train Data")
plt.xlabel("Labels")
plt.xticks(list(summary_statistics_train_data.keys()))
for i in range(len(summary_statistics_train_data)):
    plt.text(i, list(summary_statistics_train_data.values())[i], list(
        summary_statistics_train_data.values())[i], ha='center', size=9)
plt.ylabel("Number of Images")
plt.savefig("Figures/Summary Statistics of the Train Data.png")

# Flatten the target labels
breastmnist_train_labels = breastmnist_train_labels.ravel()
breastmnist_val_labels = breastmnist_val_labels.ravel()
breastmnist_test_labels = breastmnist_test_labels.ravel()

# Normalize the images
breastmnist_train_images = breastmnist_train_images.astype('float32') / 255
breastmnist_val_images = breastmnist_val_images.astype('float32') / 255
breastmnist_test_images = breastmnist_test_images.astype('float32') / 255

# Flatten the images for the model
breastmnist_train_images = breastmnist_train_images.reshape((-1, 28*28))
breastmnist_val_images = breastmnist_val_images.reshape((-1, 28*28))
breastmnist_test_images = breastmnist_test_images.reshape((-1, 28*28))

# Convert to a pandas dataframe
breastmnist_train_images = pd.DataFrame(breastmnist_train_images)
breastmnist_val_images = pd.DataFrame(breastmnist_val_images)
breastmnist_test_images = pd.DataFrame(breastmnist_test_images)


if base == True:
    # Define file paths for the models
    abc_model_path = 'Models/abc_model.pkl'
    rfc_model_path = 'Models/rfc_model.pkl'

    # Check if the AdaBoost Classifier model exists
    if os.path.exists(abc_model_path):
        with open(abc_model_path, 'rb') as file:
            abc = pickle.load(file)
    else:
        abc = AdaBoostClassifier(algorithm='SAMME')
        abc.fit(breastmnist_train_images,
                breastmnist_train_labels)
        with open(abc_model_path, 'wb') as file:
            pickle.dump(abc, file)

    # Check if the Random Forest Classifier model exists
    if os.path.exists(rfc_model_path):
        with open(rfc_model_path, 'rb') as file:
            rfc = pickle.load(file)
    else:
        rfc = RandomForestClassifier()
        rfc.fit(breastmnist_train_images,
                breastmnist_train_labels)
        with open(rfc_model_path, 'wb') as file:
            pickle.dump(rfc, file)

    # Make predictions on the validation set
    abc_predictions_val = abc.predict(breastmnist_val_images)
    rfc_predictions_val = rfc.predict(breastmnist_val_images)

    # Evaluate the initial models
    model_evaluation(breastmnist_val_labels, abc_predictions_val,
                     'AdaBoost Classifier (Untuned)', 'Validation')
    model_evaluation(breastmnist_val_labels, rfc_predictions_val,
                     'Random Forest Classifier (Untuned)', 'Validation')


# Tune the Decision Tree Classifiers
# Define file paths for the models
abc_model_tuned_path = 'Models/abc_model_tuned.pkl'
rfc_model_tuned_path = 'Models/rfc_model_tuned.pkl'

# Parameters for the best model from previous tuning
abc_n_estimators = 125
abc_learning_rate = 1

rfc_n_estimators = 75
rfc_min_samples_split = 6

if tuning == True:
    abc_param_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'learning_rate': [0.1, 0.5, 1, 5]
    }
    abc = AdaBoostClassifier(algorithm='SAMME')
    abc_grid_search = GridSearchCV(
        abc, abc_param_grid, cv=3, scoring='f1_weighted')
    abc_grid_search.fit(breastmnist_train_images, breastmnist_train_labels)
    print(f'Best parameters for the AdaBoost Classifier:'
          f'{abc_grid_search.best_params_}')
    model_evaluation(breastmnist_val_labels, abc_grid_search.predict(
        breastmnist_val_images), 'AdaBoost Classifier Tuned', 'Validation')
    with open(abc_model_tuned_path, 'wb') as file:
        pickle.dump(abc_grid_search, file)

    rfc_param_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'min_samples_split': [2, 4, 6, 8, 10]
    }
    rfc = RandomForestClassifier()
    rfc_grid_search = GridSearchCV(
        rfc, rfc_param_grid, cv=3, scoring='f1_weighted')
    rfc_grid_search.fit(breastmnist_train_images, breastmnist_train_labels)
    print(f'Best parameters for the Random Forest Classifier:'
          f'{rfc_grid_search.best_params_}')
    model_evaluation(breastmnist_val_labels, rfc_grid_search.predict(
        breastmnist_val_images), 'Random Forest Classifier Tuned', 'Validation')
    with open(rfc_model_tuned_path, 'wb') as file:
        pickle.dump(rfc_grid_search, file)


# Check if the AdaBoost Classifier model exists
if os.path.exists(abc_model_tuned_path):
    with open(abc_model_tuned_path, 'rb') as file:
        abc = pickle.load(file)
else:
    abc = AdaBoostClassifier(
        n_estimators=abc_n_estimators,
        learning_rate=abc_learning_rate,
        algorithm='SAMME')
    abc.fit(breastmnist_train_images,
            breastmnist_train_labels)
    with open(abc_model_tuned_path, 'wb') as file:
        pickle.dump(abc, file)

# Check if the Random Forest Classifier model exists
if os.path.exists(rfc_model_tuned_path):
    with open(rfc_model_tuned_path, 'rb') as file:
        rfc = pickle.load(file)
else:
    rfc = RandomForestClassifier(
        n_estimators=rfc_n_estimators, min_samples_split=rfc_min_samples_split)
    rfc.fit(breastmnist_train_images,
            breastmnist_train_labels)
    with open(rfc_model_tuned_path, 'wb') as file:
        pickle.dump(rfc, file)


# Evaluate the models on the test sets
abc_test_predictions = abc.predict(
    breastmnist_test_images)
rfc_test_predictions = rfc.predict(
    breastmnist_test_images)

# Evaluate the probability of the models on the test sets
abc_test_probabilities = abc.predict_proba(
    breastmnist_test_images)
rfc_test_probabilities = rfc.predict_proba(
    breastmnist_test_images)


# Evaluate the RoC-AUC graphs of the models on the test sets
plot_roc_curve(breastmnist_test_labels,
               abc_test_probabilities[:, 1], 'AdaBoost Classifier', 'Test')
plot_roc_curve(breastmnist_test_labels,
               rfc_test_probabilities[:, 1], 'Random Forest Classifier', 'Test')

display_incorrect_images(breastmnist_test_images, breastmnist_test_labels,
                         abc_test_predictions,  'AdaBoost Classifier', 'Test')
display_incorrect_images(breastmnist_test_images, breastmnist_test_labels,
                         rfc_test_predictions, 'Random Forest Classifier', 'Test')

model_evaluation(breastmnist_test_labels,
                 abc_test_predictions, 'AdaBoost Classifier', 'Test')

model_evaluation(breastmnist_test_labels,
                 rfc_test_predictions, 'Random Forest Classifier', 'Test')


# Define the AdaBoost Classifier model used in the Easy Ensemble Classifier
ABC_estimator = AdaBoostClassifier(
    n_estimators=abc_n_estimators, learning_rate=abc_learning_rate, algorithm='SAMME')
if base == True:
    # Define file path for the Easy Ensemble Classifier model
    eec_model_path = 'Models/eec_model.pkl'

    # Check if the Easy Ensemble Classifier model exists
    if os.path.exists(eec_model_path):
        with open(eec_model_path, 'rb') as file:
            eec = pickle.load(file)
    else:
        # Train a AdaBoost Classifier with additional bagging of balanced learners

        eec = EasyEnsembleClassifier(
            estimator=ABC_estimator, sampling_strategy=1, random_state=0)
        eec.fit(breastmnist_train_images,
                breastmnist_train_labels)
        with open(eec_model_path, 'wb') as file:
            pickle.dump(eec, file)
    eec_val_predictions = eec.predict(
        breastmnist_val_images)

    model_evaluation(breastmnist_val_labels,
                     eec_val_predictions, 'Easy Ensemble Classifier', 'Validation')

# Parameters for the best model from previous tuning
eec_n_estimators = 20

if tuning == True:
    param_grid = {
        'n_estimators': [5, 10, 20],
    }

    eec = EasyEnsembleClassifier(
        estimator=ABC_estimator, random_state=0)
    eec_grid_search = GridSearchCV(
        eec, param_grid, cv=3, scoring='f1_weighted')
    eec_grid_search.fit(breastmnist_train_images,
                        breastmnist_train_labels)
    print(f'Best parameters for the Easy Ensemble Classifier:'
          f'{eec_grid_search.best_params_}')
    eec_n_estimators = eec_grid_search.best_params_['n_estimators']
    model_evaluation(breastmnist_val_labels, eec_grid_search.predict(
        breastmnist_val_images), 'Easy Ensemble Classifier Tuned', 'Validation')

# Adjust the EEC model with the best parameters
eec = EasyEnsembleClassifier(
    estimator=ABC_estimator, n_estimators=eec_n_estimators)
eec.fit(breastmnist_train_images,
        breastmnist_train_labels)

# Define file path for the Easy Ensemble Classifier model
eec_model_tuned_path = 'Models/eec_model_tuned.pkl'

# Check if the Easy Ensemble Classifier model exists
if os.path.exists(eec_model_tuned_path):
    with open(eec_model_tuned_path, 'rb') as file:
        eec = pickle.load(file)
else:
    eec = EasyEnsembleClassifier(
        estimator=ABC_estimator, n_estimators=eec_n_estimators,
        sampling_strategy=1, random_state=0)
    eec.fit(breastmnist_train_images,
            breastmnist_train_labels)
    with open(eec_model_tuned_path, 'wb') as file:
        pickle.dump(eec, file)

# Evaluate the EEC model on the test set
eec_test_predictions = eec.predict(
    breastmnist_test_images)

eec_test_probabilities = eec.predict_proba(
    breastmnist_test_images)

plot_roc_curve(breastmnist_test_labels,
               eec_test_probabilities[:, 1], 'Easy Ensemble Classifier', 'Test')

display_incorrect_images(breastmnist_test_images,
                         breastmnist_test_labels,
                         eec_test_predictions, 'Easy Ensemble Classifier', 'Test')

model_evaluation(breastmnist_test_labels,
                 eec_test_predictions, 'Easy Ensemble Classifier', 'Test')

if base == True:
    # Define file path for the SVC model
    svc_model_path = 'Models/svc_model.pkl'

    # Check if the SVC model exists
    if os.path.exists(svc_model_path):
        with open(svc_model_path, 'rb') as file:
            svc = pickle.load(file)
    else:
        svc = SVC(kernel='linear')
        svc.fit(breastmnist_train_images,
                breastmnist_train_labels)
        with open(svc_model_path, 'wb') as file:
            pickle.dump(svc, file)
    svc_predictions = svc.predict(breastmnist_val_images)
    model_evaluation(breastmnist_val_labels,
                     svc_predictions, 'SVC with linear kernel', 'Validation')

# Define file path for the SVC model with the best kernel
svc_model_tuned_path = f'Models/svc_model_tuned.pkl'

# Parameters for the best model from previous tuning
best_C = 10
best_gamma = 0.1
best_kernel = 'rbf'

if tuning == True:
    param_grid = {
        'C': [0.1, 1, 10, 30],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    SVC_grid = GridSearchCV(SVC(probability=True),
                            param_grid, scoring='f1_weighted')
    SVC_grid.fit(breastmnist_train_images,
                 breastmnist_train_labels)
    best_C = SVC_grid.best_params_['C']
    best_gamma = SVC_grid.best_params_['gamma']
    best_kernel = SVC_grid.best_params_['kernel']
    print(f'Best parameters for the SVC model:'
          f'{SVC_grid.best_params_}')
    model_evaluation(breastmnist_val_labels, SVC_grid.predict(breastmnist_val_images),
                     f'SVC Tuned with {best_kernel} kernel', 'Validation')
    # Save the model
    with open(svc_model_tuned_path, 'wb') as file:
        pickle.dump(SVC_grid.best_estimator_, file)


# Check if the SVC model with the best kernel exists
if os.path.exists(svc_model_tuned_path):
    with open(svc_model_tuned_path, 'rb') as file:
        svc = pickle.load(file)
else:
    svc = SVC(kernel=best_kernel, C=best_C,  probability=True)
    svc.fit(breastmnist_train_images,
            breastmnist_train_labels)
    with open(svc_model_tuned_path, 'wb') as file:
        pickle.dump(svc, file)


svc_test_predictions = svc.predict(
    breastmnist_test_images)
model_evaluation(breastmnist_test_labels,
                 svc_test_predictions, f'SVC Tuned with {best_kernel} kernel', 'Test')

# Evaluate the probability of the model on the test set
svc_test_probabilities = svc.predict_proba(
    breastmnist_test_images)

# Evaluate the RoC-AUC graph of the model on the test set
plot_roc_curve(breastmnist_test_labels,
               svc_test_probabilities[:, 1], f'SVC with {best_kernel}', 'Test')

# Display images that the model predicted incorrectly
display_incorrect_images(breastmnist_test_images,
                         breastmnist_test_labels,
                         svc_test_predictions, f'SVC with {best_kernel}', 'Test')
