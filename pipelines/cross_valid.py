import logging
from steps.data_load import load_train_data, load_test_or_val_data
from utils.data_prep import standardize_and_merge
from utils.save_log_data import  config_logger
from steps.trainer import cross_val
from keras import models
import os



def cross_val_pipeline(
    classification_type: str,
    class_balance_type: str,
    basemodel:models.Model,
    fine_tune:bool,
    no_epochs:int,
    output_dir:int
)-> None:
    
    config_logger()
    logger = logging.getLogger(__name__)

    # load data
    logger.info("Loading data...")
    X_train, y_train = load_train_data(
        classification_type= classification_type,
        class_balance_type = class_balance_type,
        resize_shape=(128,128),
        train_dir="data/train_set1"
    )
    X_val, y_val = load_test_or_val_data(
        classification_type = classification_type,
        resize_shape=(128,128), 
         dir="data/validation_set1"
    )
    trainvaldata= standardize_and_merge(
        train_features=X_train,
        train_target=y_train,
        val_features=X_val,
        val_target=y_val,
        classification_type=classification_type
    )
    # evaluate model
    logger.info("Cross Validating the model...")
    crossvalresult = cross_val(
        X=trainvaldata.X_data,
        y=trainvaldata.y_data,
        basemodel=basemodel,
        classification_type=classification_type,
        fine_tune=fine_tune,
        no_epochs=no_epochs,
        input_shape=(128, 128, 3),
    )
    # Redirecting output to a file for cross-validation results
    cv_fname = 'cv_result.txt'

    # Create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cv_save_path = os.path.join(output_dir, cv_fname)
    logger.info(f"Saving cross validation result")
    with open(cv_save_path, 'w') as f:
        f.write(f"Average accuracy: {crossvalresult.avg_accuracy_score}\n")
        f.write(f"Average precision: {crossvalresult.avg_precision}\n")
        f.write(f"Average recall: {crossvalresult.avg_recall}\n")
        f.write(f"Average f1: {crossvalresult.avg_f1}\n")
        f.write(f"Average auc: {crossvalresult.avg_auc}\n")

    print(f"crossvalidation results saved at {cv_save_path}")
    
    logger.info("Cross-validation completed.")

    logger.info("Model cross validation pipeline finished successfully.")

    return None