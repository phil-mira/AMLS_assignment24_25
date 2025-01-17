# Applied Machine Learning Systems Project

This repository contains the implementation of machine learning models as part of the coursework for the Applied Machine Learning Systems project. The project is structured with separate modules for different model implementations, with pre-tuned models provided and the option to perform new tuning and validation. As required, folder A contains all modules and 

## Project Structure

```
project/
│
├── environment.yml        # Conda environment specification
├── main.py               # Main execution script
├── folder_A/            
│   ├── main_A.py             # Model A implementation
│   |── helper_functions.py   # Addtional functions
│   |── Models/               # Saved Model A variants
│   |── Reports/              # Performance reports
|   └── Figures/              
│
└── folder_B/
    ├── main_B.py             # Model B implementation
│   |── helper_functions.py   # Addtional functions
│   |── Models/               # Saved Model B variants
│   |── Reports/              # Performance reports
|   └── Figures/              
```

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [project-directory]
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate MLS_1
```

## Usage

### Running the Main Script

To execute the main program with default settings (using pre-tuned models):

```bash
python main.py
```


### Model Tuning and Validation

Each model implementation (main_A.py and main_B.py) includes options for retuning and validation. To enable these modes, modify the following parameters in the respective files:

In `folder_A/main_A.py`:
```python
tuning = False  # Enable hyperparameter tuning
base = False    # Enable validation phase
```

In `folder_B/main_B.py`:
```python
tuning = False  # Enable hyperparameter tuning
base = False    # Enable validation phase
```

Note: Running in tuning mode will take significantly longer than using the pre-tuned models. The pre-tuned models have been optimized and validated for the given datasets.

## Pre-trained Models

The repository includes pre-trained models in the following locations:
- `folder_A/models/`: Contains tuned models for implementation A
- `folder_B/models/`: Contains tuned models for implementation B

These models have been optimized using grid search and are ready for immediate use.


## Requirements

All required packages are specified in the `environment.yml` file. Key dependencies include:
- Python 3.8+
- scikit-learn
- pandas
- numpy
- pytorch




