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
    """
    load train data
    
    
    """
    # load the data
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
    """
    classification_type:        takes a str 'aug', 'oversampling', or 'none'
    resize_shape:               takes a tuple of int
    """
    
    # load test data
    loaded_test_data = data_loading(data_dir = dir)

    # resize test data
    resized_test_data =  data_class_restructuring(df = loaded_test_data,
                                        classification_type= classification_type,
                                        resize=resize_shape)
    features, target  = features_target(resized_test_data)
        
    return features, target