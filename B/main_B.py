
import numpy as np
import os

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


from helper_functions import *
import pickle


# Set the working directory to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the parameters for whether to perform hyperparameter tuning and validation
tuning = False
validation = False

dfile = '../Datasets/bloodmnist.npz'
if not os.path.exists(dfile):
    raise FileNotFoundError(f"The file {dfile} does not exist.")
data_bloodmnist = np.load(dfile)

images = data_bloodmnist['train_images']


# Exploring the bloodmnist dataset
bloodmnist_train_images = data_bloodmnist['train_images']
bloodmnist_train_labels = data_bloodmnist['train_labels']

bloodmnist_val_images = data_bloodmnist['val_images']
bloodmnist_val_labels = data_bloodmnist['val_labels']

bloodmnist_test_images = data_bloodmnist['test_images']
bloodmnist_test_labels = data_bloodmnist['test_labels']

# Summary statistics of the labels
unique, counts = np.unique(bloodmnist_train_labels, return_counts=True)
plt.bar(unique, counts)
plt.xticks(unique)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.title('Distribution of the labels')
plt.savefig('Figures/Distribution_of_labels.png')
plt.close()

# Visualize each class of the dataset
n_classes = len(np.unique(bloodmnist_train_labels))
fig, axes = plt.subplots(3, n_classes, figsize=(20, 10))
for i in range(n_classes):
    # Select the first 3 images of each class
    class_idx = np.where(bloodmnist_train_labels == i)[0][:3]
    for j, idx in enumerate(class_idx):
        axes[j, i].imshow(bloodmnist_train_images[idx])
        axes[j, i].set_title(f'Class {i}', fontsize=25)
        axes[j, i].axis('off')
plt.suptitle('BloodMNIST Dataset', fontsize=40)
plt.savefig('Figures/BloodMNIST_Dataset.png')
plt.close()

# Flatten the images
bloodmnist_train_images = bloodmnist_train_images.reshape(-1, 28*28*3)
bloodmnist_val_images = bloodmnist_val_images.reshape(-1, 28*28*3)
bloodmnist_test_images = bloodmnist_test_images.reshape(-1, 28*28*3)

# Normalize the images
bloodmnist_train_images = bloodmnist_train_images / 255.0
bloodmnist_val_images = bloodmnist_val_images / 255.0
bloodmnist_test_images = bloodmnist_test_images / 255.0


C_regularization = 10
gamma = 0.01
if tuning == True:

    # Train the model on various kernels
    svc_linear = SVC(kernel='linear')
    svc_poly = SVC(kernel='poly')
    svc_rbf = SVC(kernel='rbf')
    svc_sigmoid = SVC(kernel='sigmoid')

    kernels = {'Linear': [svc_linear], 'Poly': [svc_poly],
               'RBF': [svc_rbf], 'Sigmoid': [svc_sigmoid]}

    for n in kernels:
        model_filename = f'models/svc_{n.lower()}.pkl'
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as file:
                model = pickle.load(file)
        else:
            model = kernels[n][0]
            model.fit(bloodmnist_train_images, bloodmnist_train_labels)
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
        kernels[n].append(model.predict(bloodmnist_val_images))

    for n in kernels:
        model_evaluation(bloodmnist_val_labels, kernels[n][1], f'SVC with {
                         n} kernel', 'Validation')

    param_grid = {'C': [0.1, 1, 10,], 'gamma': [0.01, 0.001]}

    grid = GridSearchCV(SVC(kernel='rbf'), param_grid,
                        refit=True, verbose=2, scoring='f1_weighted')
    grid.fit(bloodmnist_train_images, bloodmnist_train_labels)

    C_regularization = grid.best_params_['C']
    gamma = grid.best_params_['gamma']


model_filename = 'models/svc_rbf_tuned.pkl'
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as file:
        svc_rbf_tuned = pickle.load(file)
else:
    svc_rbf_tuned = SVC(kernel='rbf', C=C_regularization,
                        gamma=gamma, probability=True)
    svc_rbf_tuned.fit(bloodmnist_train_images, bloodmnist_train_labels)
    with open(model_filename, 'wb') as file:
        pickle.dump(svc_rbf_tuned, file)

if validation == True:
    # Predict the probabilities of the validation set
    svc_rbf_tuned_pred_val = svc_rbf_tuned.predict(bloodmnist_val_images)

    svc_rbf_tuned_pred_val_prob = svc_rbf_tuned.predict_proba(
        bloodmnist_val_images)

    model_evaluation(bloodmnist_val_labels, svc_rbf_tuned_pred_val,
                     'SVC with RBF kernel (Tuned)', 'Validation')
    plot_roc_curve_multi(bloodmnist_val_labels, svc_rbf_tuned_pred_val_prob,
                         'SVC with RBF kernel (Tuned)', 'Validation')
    display_incorrect_images(data_bloodmnist['val_images'], bloodmnist_val_labels,
                             svc_rbf_tuned_pred_val, 'SVC with RBF kernel (Tuned)', 'Validation')


svc_rbf_tuned_test = svc_rbf_tuned.predict(bloodmnist_test_images)
svc_rbf_tuned_test_prob = svc_rbf_tuned.predict_proba(bloodmnist_test_images)

model_evaluation(bloodmnist_test_labels, svc_rbf_tuned_test,
                 'SVC with RBF kernel (Tuned)', 'Test')
plot_roc_curve_multi(bloodmnist_test_labels, svc_rbf_tuned_test_prob,
                     'SVC with RBF kernel (Tuned)', 'Test')
display_incorrect_images(data_bloodmnist['test_images'], bloodmnist_test_labels,
                         svc_rbf_tuned_test, 'SVC with RBF kernel (Tuned)', 'Test')


# Define the transformations
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((28, 28)),
                                transforms.ToTensor()])

# Create the dataset
train_dataset = BloodMnistDataset(
    data_bloodmnist['train_images'], data_bloodmnist['train_labels'], transform)
val_dataset = BloodMnistDataset(
    data_bloodmnist['val_images'], data_bloodmnist['val_labels'], transform)
test_dataset = BloodMnistDataset(
    data_bloodmnist['test_images'], data_bloodmnist['test_labels'], transform)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
model = BloodMnistCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 500

if validation == True:
    # Initialize lists to store training and validation losses
    train_losses, val_losses, _ = train_validation(
        model, device, train_loader, val_loader, optimizer, criterion, num_epochs)
    plot_train_validation(model, device, val_loader,
                          train_losses, val_losses, num_epochs)

if tuning == True:
    # Manually tune the hyperparameters of the CNN model
    # Define the hyperparameters
    conv1_out = [32, 64, 128]
    conv2_out = [32, 64, 128]
    fc1_out = [256, 512, 1024]
    learning_rates = [0.001, 0.01]

    # Initialize the best model and loss
    best_model = None
    best_f1 = 0
    best_lr = 0

    # Loop through the hyperparameters
    for c1 in conv1_out:
        for c2 in conv2_out:
            for f1 in fc1_out:
                for lr in learning_rates:
                    # Create the model
                    model = BloodMnistCNN(
                        conv1_out=c1, conv2_out=c2, fc1_out=f1).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()
                    num_epochs = 500

                    # Train the model
                    train_losses, val_losses, f1_score = train_validation(
                        model, device, train_loader, val_loader, optimizer, criterion, num_epochs)

                    # Update the best model and loss
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model = model
                        best_lr = lr
                        print(f'Current Best model found with conv1_out={c1}, conv2_out={
                              c2}, fc1_out={f1}, lr={lr}, f1_score={f1_score}')

else:
    conv1_out = 32
    conv2_out = 128
    fc1_out = 512
    best_lr = 0.001
    best_model = BloodMnistCNN(
        conv1_out=conv1_out, conv2_out=conv2_out, fc1_out=fc1_out).to(device)


model = best_model
optimizer = optim.Adam(model.parameters(), lr=best_lr)
criterion = nn.CrossEntropyLoss()
num_epochs = 1500

if validation == True:
    # Train and validate the model
    train_losses, val_losses, f1_score = train_validation(
        model, device, train_loader, val_loader, optimizer, criterion, num_epochs)

    # Plot the results
    plot_train_validation(model, device, val_loader,
                          train_losses, val_losses, num_epochs)
else:
    # Check if the model exists
    model_filename = 'models/best_CNN_model.pth'
    print("Checkpoint: CNN Best")
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename, weights_only=True))
        model.to(device)
    else:
        # Train the model
        for _ in range(num_epochs):
            train_loss, model = train(
                model, device, train_loader, optimizer, criterion)
        torch.save(model.state_dict(), model_filename)


test_and_plot(model, device, test_loader)
