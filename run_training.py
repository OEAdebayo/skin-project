import click
import os
from pipelines.training import training_pipeline
from tensorflow.keras.applications import VGG16, MobileNet, DenseNet121, InceptionV3,EfficientNetB0, \
     Xception, InceptionResNetV2,EfficientNetV2L,NASNetLarge

# Define the available models and their mappings
MODEL_CHOICES = {
    'VG': VGG16,
    'MN': MobileNet,
    'DN': DenseNet121,
    'IC': InceptionV3,
    'EF': EfficientNetB0,
    'XC': Xception,
    'IR': InceptionResNetV2,
    'EV': EfficientNetV2L,
    'NN': NASNetLarge
}

@click.command(
    help="""
    Entry point for running the 'bc', 'mc' 'kl' models.
    """
)
@click.option(
    "--classification_type",
    default="mc",
    help="""Specify which classification type to run.
        Valid options are 'bc', 'mc' or 'kl'. Default to 'mc'.
        """
)
@click.option(
    "--cv",
    is_flag=True,
    default=False,
    help="""Specify whether to cross validate the model or not'. Default to False
        """
)
@click.option(
    "--class_balance_type",
    default="aug",
    help="""Specify which class balance type to run.
        Valid options are 'aug', 'ovs', or 'none'. Default to 'aug'.
        """
)
@click.option(
    "--output_dir",
    default="output",
    help="""Specify directory to save the outputs of the training.  Default to 'output'
        """
)
@click.option(
    "--custom",
    default=None,
    help="Specify whether to use the custom model or utilise Transfer learning. Defaults to None."
)
@click.option(
    "--fine_tune",
    is_flag=True,
    default=False,
    help="""Specify whether to fine-tune the base model or not. Default to False
        """
)
@click.option(
    "--basemodel",
    type=click.Choice(list(MODEL_CHOICES.keys())),  
    default='VG',
    help="""Specify which base model to train.
        Choices are 'VG', 'MN', 'DN', 'IC', 'EF','CN','XC','IR','EV', 'NN' . Default to 'VG'
        """
)
@click.option(
    "--no_epochs",
    default=10,
    help="""Specify the number of epochs to train for. Default to 10""", 
)
def main(
    classification_type: str = 'mc',
    class_balance_type: str = 'aug',
    custom=None,
    basemodel: str = 'VG',
    fine_tune: bool = False,
    no_epochs: int = 1,
    cv:bool=False,
    output_dir:str='output'
) -> None:
    selected_model = MODEL_CHOICES[basemodel]

    training_pipeline(
        classification_type=classification_type,
        class_balance_type=class_balance_type,
        basemodel=selected_model,
        no_epochs=no_epochs,
        custom=custom,
        fine_tune=fine_tune,
        cv=cv,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()