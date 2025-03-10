import os
from keras import models
import logging


def save_keras_model(model: models.Model, 
                     output_dir:str,
                     model_name='model.keras', 
                     ) -> None:
    """
    Saves a trained Keras model into a specified directory in the project's root directory.

    Args:
    ----
        model: The trained Keras model to be saved.
        model_name: The name to save the model under.
        output_dir: The directory where the trained model will be created. 
                  Defaults to the current script's directory.

    Returns:
    -------
        None
    """

    # Create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct the full path to save the model
    save_path = os.path.join(output_dir, model_name)
    
    # Save the model
    model.save(save_path)
    print(f"Model saved at {save_path}")

    return None


class CustomFormatter(logging.Formatter):

    green = "\x1b[0;32m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def config_logger() -> None:

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )