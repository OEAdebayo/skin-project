from utils.data_prep import (
    data_loading,
    data_class_restructuring,
    data_augmentation,
    data_oversample,
    features_target,
    )

def load_train_data(class_balance_type:str,
                    classification_type:str,
                    resize_shape:tuple,
                    train_dir:str,        
                    )->tuple:
    
    # load the data
    """
    Loads and processes training data with optional class balancing.

    This function loads image data from a specified directory, resizes the images 
    according to a given shape, and applies optional class balancing techniques 
    such as augmentation or oversampling based on the specified class balance type.

    Parameters
    ----------
    class_balance_type : str
        Specifies the type of class balancing to apply. Must be one of 'aug', 'ovs', or 'none'.
        'aug' applies data augmentation, 'ovs' applies oversampling, and 'none' applies no balancing.
    classification_type : str
        Specifies the type of classification to perform. Generally, it should be 'mc' (multi-class) 
        or 'bc' (binary classification).
    resize_shape : tuple
        The target size for resizing images, specified as (width, height).
    train_dir : str
        The directory path containing the training images.

    Returns
    -------
    tuple
        A tuple containing the processed features and target arrays. The features are image data 
        and the target is the associated labels.
    
    Raises
    ------
    ValueError
        If the `class_balance_type` is not one of 'aug', 'ovs', or 'none'.
    """

    loaded_train_data = data_loading(data_dir = train_dir)

    # resize the data
    resized_train_data = data_class_restructuring(df = loaded_train_data,
                                        classification_type= classification_type,
                                        resize=resize_shape)
    
    # check class balance type
    if class_balance_type not in ("aug", "ovs", "none"):
        raise ValueError(f"The 'class_balance_type' parameter only takes values 'aug','ovs', 'none' but class_balance_type: {class_balance_type} was given ")
    
    elif class_balance_type == 'aug':
        augmented_data = data_augmentation(resized_train_data, 
                            class_image_limit=7000)
        features, target = features_target(augmented_data)
        return features, target
    
    elif class_balance_type =='ovs':
        features, target = features_target(resized_train_data,)

        oversampled_data = data_oversample(features,
                                            target)
        return oversampled_data["X_resampled"], oversampled_data["y_resampled"]
        
    elif class_balance_type =='none':
        features, target  = features_target(resized_train_data)
        return features, target
    

    
def load_test_or_val_data(classification_type:str,
                            resize_shape: tuple, 
                            dir:str,              
                            )->tuple:
    
    
    # load test data
    """
    Loads and resizes the test or validation data.

    This function loads image data from a specified directory.

    Parameters
    ----------
    classification_type : str
        Specifies the type of classification to perform. Generally, it should be 'mc' (multi-class) 
        or 'bc' (binary classification).
    resize_shape : tuple
        The target size for resizing images, specified as (width, height).
    dir : str
        The directory path containing the test or validation images.

    Returns
    -------
    tuple
        A tuple containing the resized features and target arrays. The features are image data 
        and the target is the associated labels.
    """
    loaded_test_data = data_loading(data_dir = dir)

    # resize test data
    resized_test_data =  data_class_restructuring(df = loaded_test_data,
                                        classification_type= classification_type,
                                        resize=resize_shape)
    features, target  = features_target(resized_test_data)
        
    return features, target