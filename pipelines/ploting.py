import logging
from steps.data_load import load_test_or_val_data
from utils.data_prep import standardize_test_data
from steps.plots import roc_curve_plot
from utils.save_log_data import config_logger
import os



def roc_curve_plot_pipeline(
    classification_type: str,
    model_dir:os.path, #FIXME change to model:models.Model after running the last simulations
    output_dir: str,
) -> None:
    
    """
    Pipeline to generate and save a ROC curve plot for a given model from a file.

    Parameters
    ----------
    classification_type : str
        The type of classification to perform. Must be one of 'mc' or 'bc'.
    model_dir : os.path
        The directory where the trained model is located.
    output_dir : str
        The directory where the ROC curve plot will be saved.

    Returns
    -------
    None
    """
    config_logger()
    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading test data...")
    
    X_test, y_test = load_test_or_val_data(
        classification_type=classification_type,
        resize_shape=(128, 128), 
        dir = "data/test_set1"
    )
    datacontainer = standardize_test_data(
        test_features=X_test,
        test_target=y_test,
        classification_type=classification_type
    )

    if classification_type not in ("mc", "bc", "kl"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc', but got {classification_type}.")

    # Plot and save confusion matrix
    logger.info("Plotting and auc roc curve...")
    roc_curve_plot(
        model_dir=model_dir, #FIXME
        x_dat=datacontainer.X_test,
        y_dat=datacontainer.y_test,
        classification_type=classification_type,
        output_dir=output_dir
    )
    logger.info("roc curve pipeline finished successfully.")

    return None