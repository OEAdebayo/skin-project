import logging
from steps.data_load import load_train_data, load_test_or_val_data
from utils.data_prep import standardize_data
from utils.model_eval import model_metric
from steps.plots import accuracy_history, conf_mat, roc_curve_plot
from utils.save_log_data import save_keras_model, config_logger
from steps.trainer import get_fitted_model_bc, get_fitted_model_mc, cross_val
from keras import models
import os

def training_pipeline(
    classification_type: str,
    class_balance_type: str,
    basemodel: models.Model,
    fine_tune: bool,
    no_epochs: int,
    cv: bool,
    output_dir: str,
    custom=None,
) -> None:
    
    config_logger()
    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading data...")
    X_train, y_train = load_train_data(
        classification_type=classification_type,
        class_balance_type=class_balance_type,
        resize_shape=(128, 128),
        train_dir="data/train_set1"
    )
    X_val, y_val = load_test_or_val_data(
        classification_type=classification_type,
        resize_shape=(128, 128),
        dir="data/validation_set1")

    X_test, y_test = load_test_or_val_data(
        classification_type=classification_type,
        resize_shape=(128, 128), 
        dir = "data/test_set1"
    )

    datacontainer = standardize_data(
        train_features=X_train,
        train_target=y_train,
        val_features=X_val,
        val_target=y_val,
        test_features=X_test,
        test_target=y_test,
        classification_type=classification_type
    )

    if classification_type not in ("mc", "bc", "kl"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc', but got {classification_type}.")

    # Perform cross-validation if specified
    if cv:
        logger.info("Cross Validating the model...")
        crossvalresult = cross_val(
            X=datacontainer.X_train,
            y=datacontainer.y_train,
            basemodel=basemodel,
            classification_type=classification_type,
            fine_tune=fine_tune,
            no_epochs=no_epochs,
            input_shape=(128, 128, 3),
        )
        
        # Redirecting output to a file for cross-validation results
        cv_fname = 'cv_result.txt'
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

    # Train the final model
    logger.info(f"Training {classification_type} model...")
    if classification_type in ('mc', 'kl'):
        model, history = get_fitted_model_mc(
            input_shape=(128, 128, 3),
            X_train=datacontainer.X_train,
            y_train=datacontainer.y_train,
            X_val=datacontainer.X_val,
            y_val=datacontainer.y_val,
            custom = custom,
            fine_tune=fine_tune,
            basemodel=basemodel,
            no_epochs=no_epochs,
            cv=False,
        )
    elif classification_type == 'bc':
        model, history = get_fitted_model_bc(
            input_shape=(128, 128, 3),
            X_train=datacontainer.X_train,
            y_train=datacontainer.y_train,
            X_val=datacontainer.X_val,
            y_val=datacontainer.y_val,
            custom = custom,
            fine_tune=fine_tune,
            basemodel=basemodel,
            no_epochs=no_epochs,
            cv=False,
        )

    # Evaluate model
    logger.info("Evaluating model performance...")
    model_performance = model_metric(
        model=model,
        x_dat=datacontainer.X_test,
        y_dat=datacontainer.y_test,
        classification_type=classification_type
    )

    # Save accuracy history plot
    logger.info("Saving the history of the accuracy plots")
    accuracy_history(modelHist=history.history, output_dir=output_dir)

    # Plot and save confusion matrix
    logger.info("Plotting and saving the confusion matrix...")
    conf_mat(
        model=model,
        x_dat=datacontainer.X_test,
        y_dat=datacontainer.y_test,
        classification_type=classification_type,
        output_dir=output_dir,
    )
    
    # Save the trained model
    logger.info("Saving trained keras model...")
    save_keras_model(model=model, output_dir=output_dir)

    mp_fname = 'model_performance.txt'
    mp_save_path = os.path.join(output_dir, mp_fname)
    logger.info(f"Saving cross validation result")

    # Save model performance metrics to a file
    with open(mp_save_path, 'w') as f:
        f.write(f"Accuracy: {model_performance.accuracy_score}\n")
        f.write(f"Precision: {model_performance.precision}\n")
        f.write(f"Recall: {model_performance.recall}\n")
        f.write(f"F1 Score: {model_performance.f1}\n")
        f.write(f"AUC: {model_performance.auc}\n")

    print(f"Model performance results saved at {mp_save_path}")

    # Save the trained model
    logger.info("plotting the auc-roc curve...")
    roc_curve_plot(
        model=model, #FIXME seems correct
        x_dat=datacontainer.X_test,
        y_dat=datacontainer.y_test,
        classification_type=classification_type,
        output_dir=output_dir
    )
    
    logger.info("Model training pipeline finished successfully.")

    return None
