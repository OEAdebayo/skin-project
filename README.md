# Skin-Project

This repository contains the code for all the experiments performed in the paper: [Machine Learning and Deep Learning Approaches for Classifying Keloid Images in the Context of Malignant and Benign Skin Disorders](https://www.mdpi.com/2075-4418/15/6/710).

## About

The classical approach to diagnosing various skin disorders, including keloids, relies on dermatoscopy. However, this method is often considered complex and expensive. In this study, we propose a deep learning model based on transfer learning to identify non-dermatoscopic (clinical) images of keloids among other benign and malignant skin lesions.

## Setup for Running Locally

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/OEAdebayo/skin-project.git
    ```

2. **Ensure Python 3.12 or later is installed:**
    Check your Python version:
    ```bash
    python -V
    ```
    If the version is below 3.12, follow [this guide](https://github.com/pyenv/pyenv) to install Python 3.12 using `pyenv`.

3. **Navigate to the project root and create a virtual environment:**
    ```bash
    cd skin-project
    python3 -m venv .venv
    ```

4. **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate  # On macOS/Linux
    .venv\Scripts\activate    # On Windows
    ```

5. **Install dependencies and set up required directories:**
    ```bash
    make setup
    ```

## Usage

To train the model and run experiments, use the provided script `run_training.py` with the appropriate arguments.

### Example: Training the VGG16 Model
To train the default (i.e., VGG16) model without fine-tuning and using the original training dataset for binary classification over 10 epochs, run:
```bash
python run_training.py --classification_type bc --class_balance_type none
```
The results will be saved in the `output` directory.

### Viewing Available Arguments
To see all available options for `run_training.py`, run:
```bash
python run_training.py --help
```

## Citation
If you use this work in your project, please cite the following paper:

```bibtex
@article{OLUSEGUNETAL2025,
    title = {Deep Learning Approaches for the Classification of Keloid Images in the Context of Malignant and Benign Skin Disorders},
    author = {Olusegun Ekundayo Adebayo and Brice Chatelain and Dumitru Trucu and Raluca Eftimie},
    journal = {Diagnostics},
    volume = {15},
    year = {2025},
    issue = {6},
    pages = {710},
    doi = {10.3390/diagnostics15060710},
    url = {https://www.mdpi.com/2075-4418/15/6/710}
}
```

## License
This project is released under the [MIT License](LICENSE).

---

For any issues or contributions, please feel free to open an issue or submit a pull request.

