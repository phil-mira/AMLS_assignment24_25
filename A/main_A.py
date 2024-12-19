import pickle
import helper_functions
import importlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as accuracy
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from helper_functions import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os


# Set the working directory to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the tuning parameter to True to tune the models
tuning = False

# Set the validation parameter to True to run the validation set
validation = False


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
plt.savefig("Summary Statistics of the Train Data.png")

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

# Add the labels to the dataframes
breastmnist_train_images['label'] = breastmnist_train_labels
breastmnist_val_images['label'] = breastmnist_val_labels
breastmnist_test_images['label'] = breastmnist_test_labels

if validation == True:
    # Define file paths for the models
    gbc_model_path = 'Models/gbc_model.pkl'
    rfc_model_path = 'Models/rfc_model.pkl'

    # Check if the Gradient Boosting Classifier model exists
    if os.path.exists(gbc_model_path):
        with open(gbc_model_path, 'rb') as file:
            gbc = pickle.load(file)
    else:
        gbc = GradientBoostingClassifier()
        gbc.fit(breastmnist_train_images.drop(columns='label'),
                breastmnist_train_images['label'])
        with open(gbc_model_path, 'wb') as file:
            pickle.dump(gbc, file)

    # Check if the Random Forest Classifier model exists
    if os.path.exists(rfc_model_path):
        with open(rfc_model_path, 'rb') as file:
            rfc = pickle.load(file)
    else:
        rfc = RandomForestClassifier()
        rfc.fit(breastmnist_train_images.drop(columns='label'),
                breastmnist_train_images['label'])
        with open(rfc_model_path, 'wb') as file:
            pickle.dump(rfc, file)

    # Make predictions on the validation set
    gbc_predictions_val = gbc.predict(
        breastmnist_val_images.drop(columns='label'))
    rfc_predictions_val = rfc.predict(
        breastmnist_val_images.drop(columns='label'))

    # Evaluate the initial models
    model_evaluation(breastmnist_val_images['label'], gbc_predictions_val,
                     'Gradient Boosting Classifier (Untuned)', 'Validation')
    model_evaluation(breastmnist_val_images['label'], rfc_predictions_val,
                     'Random Forest Classifier (Untuned)', 'Validation')


# Tune the Gradient Boosting Classifier - left commented due to long runtime
if tuning == True:
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5]
    }
    grid_search = GridSearchCV(gbc, param_grid, cv=3, scoring='f1_weighted')
    grid_search.fit(breastmnist_train_images.drop(
        columns='label'), breastmnist_train_images['label'])
    print(f'Best parameters for the Gradient Boosting Classifier: {
          grid_search.best_params_}')

    param_grid = {
        'n_estimators': [50, 100, 150]
    }
    grid_search = GridSearchCV(rfc, param_grid, cv=3, scoring='f1_weighted')
    grid_search.fit(breastmnist_train_images.drop(
        columns='label'), breastmnist_train_images['label'])
    print(f'Best parameters for the Random Forest Classifier: {
          grid_search.best_params_}')


# Define file paths for the models
gbc_model_tuned_path = 'Models/gbc_model_tuned.pkl'
rfc_model_tuned_path = 'Models/rfc_model_tuned.pkl'

# Check if the Gradient Boosting Classifier model exists
if os.path.exists(gbc_model_tuned_path):
    with open(gbc_model_tuned_path, 'rb') as file:
        gbc = pickle.load(file)
else:
    gbc = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    gbc.fit(breastmnist_train_images.drop(columns='label'),
            breastmnist_train_images['label'])
    with open(gbc_model_tuned_path, 'wb') as file:
        pickle.dump(gbc, file)

# Check if the Random Forest Classifier model exists
if os.path.exists(rfc_model_tuned_path):
    with open(rfc_model_tuned_path, 'rb') as file:
        rfc = pickle.load(file)
else:
    rfc = RandomForestClassifier(n_estimators=150)
    rfc.fit(breastmnist_train_images.drop(columns='label'),
            breastmnist_train_images['label'])
    with open(rfc_model_tuned_path, 'wb') as file:
        pickle.dump(rfc, file)


# Evaluate the models on the test sets
gbc_test_predictions = gbc.predict(
    breastmnist_test_images.drop(columns='label'))
rfc_test_predictions = rfc.predict(
    breastmnist_test_images.drop(columns='label'))

# Evaluate the models
gbc_tuned_test = model_evaluation(
    breastmnist_test_images['label'],
    gbc_test_predictions, 'Gradient Boosting Classifier', 'Test')
rfc_tuned_test = model_evaluation(
    breastmnist_test_images['label'],
    rfc_test_predictions, 'Random Forest Classifier', 'Test')

# Evaluate the probability of the models on the test sets
gbc_test_probabilities = gbc.predict_proba(
    breastmnist_test_images.drop(columns='label'))
rfc_test_probabilities = rfc.predict_proba(
    breastmnist_test_images.drop(columns='label'))


# Evaluate the RoC-AUC graphs of the models on the test sets
plot_roc_curve(breastmnist_test_images['label'],
               gbc_test_probabilities[:, 1], 'Gradient Boosting Classifier', 'Test')
plot_roc_curve(breastmnist_test_images['label'],
               rfc_test_probabilities[:, 1], 'Random Forest Classifier', 'Test')

display_incorrect_images(breastmnist_test_images.drop(
    columns='label'), breastmnist_test_images['label'],
    gbc_test_predictions,  'Gradient Boosting Classifier', 'Test')
display_incorrect_images(breastmnist_test_images.drop(
    columns='label'), breastmnist_test_images['label'],
    rfc_test_predictions, 'Random Forest Classifier', 'Test')

model_evaluation(breastmnist_test_images['label'],
                 gbc_test_predictions, 'Gradient Boosting Classifier', 'Test')

model_evaluation(breastmnist_test_images['label'],
                 rfc_test_predictions, 'Random Forest Classifier', 'Test')


if validation == True:
    # Define file path for the Easy Ensemble Classifier model
    eec_model_path = 'Models/eec_model.pkl'

    # Check if the Easy Ensemble Classifier model exists
    if os.path.exists(eec_model_path):
        with open(eec_model_path, 'rb') as file:
            eec = pickle.load(file)
    else:
        # Train a Gradient Boosting Classifier with additional bagging of balanced learners
        GBC_estimator = GradientBoostingClassifier(
            n_estimators=100, max_depth=3)
        eec = EasyEnsembleClassifier(
            estimator=GBC_estimator, sampling_strategy=1, random_state=0)
        eec.fit(breastmnist_train_images.drop(columns='label'),
                breastmnist_train_images['label'])
        with open(eec_model_path, 'wb') as file:
            pickle.dump(eec, file)
    eec_val_predictions = eec.predict(
        breastmnist_val_images.drop(columns='label'))

    model_evaluation(breastmnist_val_images['label'],
                     eec_val_predictions, 'Easy Ensemble Classifier', 'Validation')

else:
    # Define the Gradient Boosting Classifier model for the Easy Ensemble Classifier
    GBC_estimator = GradientBoostingClassifier(n_estimators=100, max_depth=3)


best_n_estimators = 20

if tuning == True:
    param_grid = {
        'n_estimators': [5, 10, 20],
    }

    grid_search = GridSearchCV(eec, param_grid, cv=3, scoring='f1_weighted')
    grid_search.fit(breastmnist_train_images.drop(columns='label'),
                    breastmnist_train_images['label'])
    print(f'Best parameters for the Easy Ensemble Classifier: {
          grid_search.best_params_}')
    best_n_estimators = grid_search.best_params_['n_estimators']

# Adjust the EEC model with the best parameters
eec = EasyEnsembleClassifier(
    estimator=GBC_estimator, n_estimators=best_n_estimators)
eec.fit(breastmnist_train_images.drop(columns='label'),
        breastmnist_train_images['label'])

# Define file path for the Easy Ensemble Classifier model
eec_model_tuned_path = 'Models/eec_model_tuned.pkl'

# Check if the Easy Ensemble Classifier model exists
if os.path.exists(eec_model_tuned_path):
    with open(eec_model_tuned_path, 'rb') as file:
        eec = pickle.load(file)
else:
    # Train a Gradient Boosting Classifier with additional bagging of balanced learners
    GBC_estimator = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    eec = EasyEnsembleClassifier(
        estimator=GBC_estimator, n_estimators=best_n_estimators,
        sampling_strategy=1, random_state=0)
    eec.fit(breastmnist_train_images.drop(columns='label'),
            breastmnist_train_images['label'])
    with open(eec_model_path, 'wb') as file:
        pickle.dump(eec, file)

# Evaluate the EEC model on the test set
eec_test_predictions = eec.predict(
    breastmnist_test_images.drop(columns='label'))

eec_test_probabilities = eec.predict_proba(
    breastmnist_test_images.drop(columns='label'))

plot_roc_curve(breastmnist_test_images['label'],
               eec_test_probabilities[:, 1], 'Easy Ensemble Classifier', 'Test')

display_incorrect_images(breastmnist_test_images.drop(columns='label'),
                         breastmnist_test_images['label'],
                         eec_test_predictions, 'Easy Ensemble Classifier', 'Test')

model_evaluation(breastmnist_test_images['label'],
                 eec_test_predictions, 'Easy Ensemble Classifier', 'Test')

if validation == True:
    # Define file path for the SVC model
    svc_model_path = 'Models/svc_model.pkl'

    # Check if the SVC model exists
    if os.path.exists(svc_model_path):
        with open(svc_model_path, 'rb') as file:
            svc = pickle.load(file)
    else:
        svc = SVC(kernel='linear')
        svc.fit(breastmnist_train_images.drop(columns='label'),
                breastmnist_train_images['label'])
        with open(svc_model_path, 'wb') as file:
            pickle.dump(svc, file)
    svc_predictions = svc.predict(breastmnist_val_images.drop(columns='label'))
    model_evaluation(breastmnist_val_images['label'],
                     svc_predictions, 'SVC with linear kernel', 'Validation')

# Tune the SVC model by using different kernels
param_grid = ['linear', 'poly', 'rbf', 'sigmoid']
f1_score = {}
svc_pred_dict = {}
best_kernel = 'linear'

if tuning == True:
    for k in param_grid:
        svc_tune = SVC(kernel=k)
        svc_tune.fit(breastmnist_train_images.drop(
            columns='label'), breastmnist_train_images['label'])
        svc_pred_dict[k] = svc_tune.predict(
            breastmnist_val_images.drop(columns='label'))
        f1_score[k] = classification_report(
            breastmnist_val_images['label'], svc_pred_dict[k],
            output_dict=True)['weighted avg']['f1-score']

    best_kernel = max(f1_score, key=f1_score.get)
    model_evaluation(breastmnist_val_images['label'], svc_pred_dict[best_kernel],
                     f'SVC with {best_kernel}', 'Validation')


# Define file path for the SVC model with the best kernel
svc_model_best_kernel_path = f'svc_model_{best_kernel}.pkl'

# Check if the SVC model with the best kernel exists
if os.path.exists(svc_model_best_kernel_path):
    with open(svc_model_best_kernel_path, 'rb') as file:
        svc = pickle.load(file)
else:
    svc = SVC(kernel=best_kernel, probability=True)
    svc.fit(breastmnist_train_images.drop(columns='label'),
            breastmnist_train_images['label'])
    with open(svc_model_best_kernel_path, 'wb') as file:
        pickle.dump(svc, file)
svc_test_predictions = svc.predict(
    breastmnist_test_images.drop(columns='label'))
model_evaluation(breastmnist_test_images['label'],
                 svc_test_predictions, f'SVC with {best_kernel}', 'Test')

# Evaluate the probability of the model on the test set
svc_test_probabilities = svc.predict_proba(
    breastmnist_test_images.drop(columns='label'))

# Evaluate the RoC-AUC graph of the model on the test set
plot_roc_curve(breastmnist_test_images['label'],
               svc_test_probabilities[:, 1], f'SVC with {best_kernel}', 'Test')

# Display images that the model predicted incorrectly
display_incorrect_images(breastmnist_test_images.drop(columns='label'),
                         breastmnist_test_images['label'],
                         svc_test_predictions, f'SVC with {best_kernel}', 'Test')


# Test PCA on the breastmnist dataset for the Gradient Boosting Classifier model
n_of_components = [5, 10, 20, 50, 100]
f1_score_PCA = {}
gbc_pred_dict_PCA = {}
best_n = 20

if tuning == True:
    for n in n_of_components:
        pca = PCA(n_components=n)
        breastmnist_train_images_pca = pca.fit_transform(
            breastmnist_train_images.drop(columns='label'))
        breastmnist_val_images_pca = pca.transform(
            breastmnist_val_images.drop(columns='label'))
        breastmnist_test_images_pca = pca.transform(
            breastmnist_test_images.drop(columns='label'))

        gbc_PCA = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        gbc_PCA.fit(breastmnist_train_images_pca,
                    breastmnist_train_images['label'])
        gbc_PCA_predictions = gbc_PCA.predict(breastmnist_val_images_pca)
        gbc_pred_dict_PCA[n] = gbc_PCA_predictions
        f1_score_PCA[n] = classification_report(
            breastmnist_val_images['label'], gbc_PCA_predictions,
            output_dict=True)['weighted avg']['f1-score']

    best_n = max(f1_score_PCA, key=f1_score_PCA.get)
    model_evaluation(breastmnist_val_images['label'], gbc_pred_dict_PCA[best_n],
                     f'Gradient Boosting Classifier with PCA (n={best_n})', 'Validation')

# Define file path for the PCA model
pca_model_path = f'pca_model_{best_n}.pkl'

# Check if the PCA model exists
if os.path.exists(pca_model_path):
    with open(pca_model_path, 'rb') as file:
        pca = pickle.load(file)
else:
    pca = PCA(n_components=best_n)
    breastmnist_train_images_pca = pca.fit_transform(
        breastmnist_train_images.drop(columns='label'))
    with open(pca_model_path, 'wb') as file:
        pickle.dump(pca, file)

# Transform the test images using the PCA model
breastmnist_test_images_pca = pca.transform(
    breastmnist_test_images.drop(columns='label'))

# Define file path for the Gradient Boosting Classifier with PCA model
gbc_pca_model_path = f'Models/gbc_pca_model_{best_n}.pkl'

# Check if the Gradient Boosting Classifier with PCA model exists
if os.path.exists(gbc_pca_model_path):
    with open(gbc_pca_model_path, 'rb') as file:
        gbc_PCA = pickle.load(file)
else:
    gbc_PCA = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    gbc_PCA.fit(breastmnist_train_images_pca,
                breastmnist_train_images['label'])
    with open(gbc_pca_model_path, 'wb') as file:
        pickle.dump(gbc_PCA, file)

# Make predictions on the test set using the Gradient Boosting Classifier with PCA model
gbc_PCA_predictions = gbc_PCA.predict(breastmnist_test_images_pca)
gbc_PCA_probabilities = gbc_PCA.predict_proba(breastmnist_test_images_pca)

# Evaluate the model
plot_roc_curve(breastmnist_test_images['label'], gbc_PCA_probabilities[:, 1],
               f'Gradient Boosting Classifier with PCA (n={best_n})', 'Test')
display_incorrect_images(breastmnist_test_images.drop(columns='label'),
                         breastmnist_test_images['label'],
                         gbc_PCA_predictions, f'Gradient Boosting Classifier with PCA (n={best_n})', 'Test')
model_evaluation(breastmnist_test_images['label'], gbc_PCA_predictions,
                 f'Gradient Boosting Classifier with PCA (n={best_n})', 'Test')
