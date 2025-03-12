# skin-project
This repository contains the codes for all the experiments performed in the paper [Machine learning and deep learning approaches for classifying keloid images in the context of malignant and benign skin disorders.](https://www...)


## Set up for running locally
1. Clone the repository by running
    ```bash
    git clone https://github.com/OEAdebayo/skin-project.git
    ```

1. This project requires python version 12 and above. Check your python version
    ```bash
    python -V
    ```
    If the python version is below python12 check [this link](https://github.com/pyenv/pyenv) to install setup your python12 environment using pyenv.
    Check the python version again using the command above and make sure its at least python 12.

1. Navigate to the root folder, i.e., `skin-project` and create a python virtual environment by running

    ```bash
    python3 -m venv .venv
    ``` 
1. Activate the virtual environment by running
    ```bash
    source .venv/bin/activate
    ```
1. Prepare all modules and required directories by running the following:
    ```bash
    make setup
    ```

1. You can then start running entry point (`run_training.py` for the model training and various experiment pipelines respectively) with their respective arguments as CLI. For instance to train the proposed VGG16 model without fine-tuning the it and with the original train data for 10 epochs for the binary classification task run:
    ```bash
    python run_training.py --classification_type bc --class_balance_type none
    ```
    The results will be found in the 'output' directory.

    To see all the available arguments or options to an entry point, e.g., for training pipeline entry point run:
    ```bash
    python run_training.py --help
    ```
If you use this work in your project, please reference:

    @article{OLUSEGUNETAL2025,
        title = {Deep Learning Approaches for the Classification of Keloid Images in the Context of Malignant and Benign Skin Disorders.},
        author = {Olusegun Ekundayo and Brice Chatelain and Dumitru Trucu and Raluca Eftimie},
        journal = {Diagnostics},
        volume = {15},
        year = {2025},
        issn = {6},
        pages={710},
        doi = {10.3390/diagnostics15060710},
    }