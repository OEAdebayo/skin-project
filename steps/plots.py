import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score
from keras import models



def accuracy_history(modelHist:dict, 
                     output_dir:str,
                     figsize:tuple=(8, 6), 
                     alpha:float=1.0,
                     fontsize=20, 
                     ) -> None:
    """
    Function to plot and save the model accuracy over epochs.

    Args:
    ----
        - modelHist:        Dictionary containing training accuracy and validation accuracy.
        - output_dir:       The directory where the accuracy plot will be saved. Defaults to 'output'.
        - figsize:          Tuple specifying the figure size.
        - alpha:            Float specifying the transparency of the plot lines.
        - fontsize:         Integer specifying the font size for plot labels.
        

    Returns:
    -------
        None
    """
    Acc_values = modelHist.get('accuracy')
    val_acc_values = modelHist.get('val_accuracy')

    if Acc_values is None or val_acc_values is None:
        raise KeyError("The model history dictionary does not contain the keys 'accuracy' and 'val_accuracy'.")

    Epochs = range(1, len(Acc_values) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(Epochs, Acc_values, 'bo-', label='Train-acc', linewidth=2, markersize=5, alpha=alpha)
    plt.plot(Epochs, val_acc_values, 'b--', label='Valid-acc', linewidth=2, alpha=alpha)
    
    plt.xlabel("Epochs", fontsize=fontsize, weight='bold')
    plt.ylabel("Accuracy", fontsize=fontsize, weight='bold')
    plt.ylim(0, 1)
    
    font_prop = fm.FontProperties(weight='bold', size=fontsize)
    plt.legend(prop=font_prop, fontsize=fontsize, loc='lower right', frameon=True).get_frame().set_edgecolor('black')
    plt.grid(True)
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  
    print(f"Model training history saved at {save_path}")

    return None


def conf_mat(model: models.Model, 
             output_dir:str,
             x_dat: np.ndarray, 
             y_dat: np.ndarray,
             classification_type: str,
             )-> None:
    """
    Function to generate and save a confusion matrix plot.

    Args:
    ----
        - model:                    The trained keras model used for predictions.
        - output_dir:               The directory where the confusion matrix plot will be saved. Defaults to 'output'.
        - x_dat:                    The input data for prediction.
        - y_dat:                    The true labels.
        - classification_type:      Type of classification ('mc' for multi-class, 'bc' for binary-class).
        

    Returns:
    -------
        None
    """

    if classification_type not in ("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given.")
    
    elif classification_type == 'bc': 
        y_pred_prob = model.predict(x_dat, verbose=0)
        y_pred = [1 if y > 0.5 else 0 for y in y_pred_prob]
        y_dat = y_dat.astype(int)
        cm = confusion_matrix(y_dat, y_pred)
        classes = ['Benign', 'Malignant'] 

    elif classification_type == 'mc': 
        y_pred = np.argmax(model.predict(x_dat, verbose=0), axis=1)
        y_dat = np.argmax(y_dat, axis=1)
        cm = confusion_matrix(y_dat, y_pred)
        classes = ['Benign', 'Malignant', 'Keloid'] 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    file_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(file_path)
    plt.close() 
    print(f"Confusion matrix saved at {file_path}")

    return None

def roc_curve_plot(model: models.Model, 
                   output_dir: str,
                   x_dat: np.ndarray, 
                   y_dat: np.ndarray,
                   classification_type: str) -> None:
    """
    Function to generate and save a ROC curve plot.

    Args:
    ----
        - model:                    The trained keras model used for predictions.
        - output_dir:               The directory where the ROC curve plot will be saved.
        - x_dat:                    The input data for prediction.
        - y_dat:                    The true labels.
        - classification_type:      Type of classification ('mc' for multi-class, 'bc' for binary-class).
        

    Returns:
    -------
        None
    """


    if classification_type not in ("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if classification_type == 'bc':
        
        y_pred_prob = model.predict(x_dat, verbose=0)
        
        fpr, tpr, _ = roc_curve(y_dat, y_pred_prob)
        
        auc_score = roc_auc_score(y_dat, y_pred_prob)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        file_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(file_path)
        plt.close()
        print(f"ROC curve saved at {file_path}")

    elif classification_type == 'mc':
        y_pred_prob = model.predict(x_dat, verbose=0)
        n_classes = y_dat.shape[1]
        classes = ['Benign', 'Malignant', 'Keloid']

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_dat[:, i], y_pred_prob[:, i])
            roc_auc[i] = roc_auc_score(y_dat[:, i], y_pred_prob[:, i])

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (area = {roc_auc[i]:.4f})')

        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        file_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(file_path)
        plt.close()
        print(f"ROC curve saved at {file_path}")

    return None