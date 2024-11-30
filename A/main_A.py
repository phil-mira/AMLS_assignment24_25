import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import importlib
import helper_functions
importlib.reload(helper_functions)
from helper_functions import *

from sklearn.model_selection import GridSearchCV


# Load the breastmnist dataset
dfile = '../Datasets/BreastMNIST.npz'
if not os.path.exists(dfile):
	raise FileNotFoundError(f"The file {dfile} does not exist.")
data_breastmnist = np.load(dfile)


# Reformat the data 
breastmnist_train_images = data_breastmnist['train_images']
breastmnist_train_labels = data_breastmnist['train_labels']

breastmnist_val_images = data_breastmnist['val_images']
breastmnist_val_labels = data_breastmnist['val_labels']

breastmnist_test_images = data_breastmnist['test_images']
breastmnist_test_labels = data_breastmnist['test_labels']


# Inspect the data
print(f"Train images shape breastmnist: {breastmnist_train_images.shape}")
print(f"Train labels shape breastmnist: {breastmnist_train_labels.shape}")

# Summary statistics of the labels
unique, counts = np.unique(breastmnist_train_labels, return_counts=True)
summary_statistics_train_data = dict(zip(unique, counts))
ax,fig = plt.subplots()
plt.bar(summary_statistics_train_data.keys(), summary_statistics_train_data.values())
plt.title("Summary Statistics of the Train Data")
plt.xlabel("Labels")
plt.xticks(list(summary_statistics_train_data.keys()))
for i in range(len(summary_statistics_train_data)):
    plt.text(i, list(summary_statistics_train_data.values())[i], list(summary_statistics_train_data.values())[i], ha = 'center', size = 9)
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
	  

# Dictionary to store the F1-scores of the models
model_evaulation_dict = {}

# Train a Gradient Boosting Classifier and a Random Forest Classifier on the breastmnist dataset
# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(breastmnist_train_images.drop(columns='label'), breastmnist_train_images['label'])
gbc_predictions_val = gbc.predict(breastmnist_val_images.drop(columns='label'))

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(breastmnist_train_images.drop(columns='label'), breastmnist_train_images['label'])
rfc_predictions_val = rfc.predict(breastmnist_val_images.drop(columns='label'))

# Evaluate the initial models
model_evaluation(breastmnist_val_images['label'], gbc_predictions_val, 'Gradient Boosting Classifier (Untuned)', 'Validation')
model_evaluation(breastmnist_val_images['label'], rfc_predictions_val, 'Random Forest Classifier (Untuned)', 'Validation')


# Tune the Gradient Boosting Classifier - left commented due to long runtime
"""param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5]
}
grid_search = GridSearchCV(gbc, param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(breastmnist_train_images.drop(columns='label'), breastmnist_train_images['label'])
print(f'Best parameters for the Gradient Boosting Classifier: {grid_search.best_params_}')
"""

# Tune the Random Forest Classifier - left commented due to long runtime
"""param_grid = {
    'n_estimators': [50, 100, 150]
}
grid_search = GridSearchCV(rfc, param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(breastmnist_train_images.drop(columns='label'), breastmnist_train_images['label'])
print(f'Best parameters for the Random Forest Classifier: {grid_search.best_params_}')
"""


# Adjust the models with the best parameters
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=3)
gbc.fit(breastmnist_train_images.drop(columns='label'), breastmnist_train_images['label'])

rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(breastmnist_train_images.drop(columns='label'), breastmnist_train_images['label'])


# Evaluate the models on the test sets
gbc_test_predictions = gbc.predict(breastmnist_test_images.drop(columns='label'))
rfc_test_predictions = rfc.predict(breastmnist_test_images.drop(columns='label'))

# Evaluate the models
gbc_tuned_test = model_evaluation(breastmnist_test_images['label'], gbc_test_predictions, 'Gradient Boosting Classifier', 'Test')
rfc_tuned_test = model_evaluation(breastmnist_test_images['label'], rfc_test_predictions, 'Random Forest Classifier', 'Test')

# Add the F1-scores to the dictionary for comparison
model_evaulation_dict['Gradient Boosting Classifier'] = gbc_tuned_test['weighted avg']['f1-score']
model_evaulation_dict['Random Forest Classifier'] = rfc_tuned_test['weighted avg']['f1-score']




