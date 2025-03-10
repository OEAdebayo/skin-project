import os
import pandas as pd
import numpy as np
import pathlib as Path
import multiprocessing
from PIL import Image
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from keras import utils
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out WARNING, 3 = filter out ERROR


def data_loading(data_dir: Path)-> object:


    # Create an empty dataframes
    """
    Loads image data from a specified directory into a DataFrame.

    This function iterates through all subdirectories within a given directory,
    assuming each subdirectory represents a class label. It constructs a DataFrame
    where each row contains the path to an image file and its corresponding label.

    Parameters
    ----------
    data_dir : Path
        The path to the directory containing subdirectories of images, where each 
        subdirectory is named according to the class label of the images it contains.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: 'image_path' containing the full path to each 
        image file, and 'label' containing the class label of each image.
    """

    df = pd.DataFrame(columns=['image_path', 'label'])
   
    for directory in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, directory)):
            image_path = os.path.join(data_dir, directory, filename)
            row_to_append = pd.DataFrame({'image_path': [image_path], 'label': [directory]})

            # Use concat to append the row to the DataFrame
            df = pd.concat([df, row_to_append], ignore_index=True)
   
    return df

    
def data_class_restructuring(df,
                             resize:tuple, 
                             classification_type = str,
                             image = "image", 
                             image_path= "image_path",
                             class_label = "label", 
                             ) -> object:
    
    #if (resize and image and image_path):
       
    # Get the number of CPU cores available

    """
    Restructures a DataFrame by resizing images and modifying class labels based on classification type.

    This function resizes images specified in a DataFrame and updates class labels according to the
    specified classification type, which can be 'bc' (binary classification), 'mc' (multi-class classification),
    or 'kl' (keloid classification). The images are resized in parallel using multiple CPU cores.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing image paths and class labels.
    resize : tuple
        The target size for resizing images, specified as (width, height).
    classification_type : str
        The type of classification to perform. Must be one of 'bc', 'mc', or 'kl'.
    image : str, optional
        The column name in the DataFrame where resized images will be stored. Defaults to 'image'.
    image_path : str, optional
        The column name containing paths to the images. Defaults to 'image_path'.
    class_label : str, optional
        The column name containing class labels. Defaults to 'label'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with resized images and updated class labels.

    Raises
    ------
    ValueError
        If the `classification_type` is not one of 'bc', 'mc', or 'kl'.
    """

    max_workers = multiprocessing.cpu_count()
    import concurrent.futures

    def convert_rgba_to_rgb(image_array):
        if image_array.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
            image = Image.fromarray(image_array)
            image = image.convert("RGB")  # Convert to RGB
            return np.array(image)
        return image_array

    def resize_image_array(image_path):
        image = Image.open(image_path).resize(resize)
        image_array = np.asarray(image)
        
        # Convert RGBA to RGB if necessary
        image_array = convert_rgba_to_rgb(image_array)
        
        return image_array
    
    df_use = df.copy() # get a copy of the dataframe to use

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Use executor.map to apply the function to each image path in the DataFrame
        image_arrays = list(executor.map(resize_image_array,
                                         df_use[image_path].tolist()))
        

    df_use[image] = image_arrays
    if classification_type not in ("mc", "bc", "kl"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc', 'bc' or 'kl' but classification_type: {classification_type} was giving ")

    
    elif classification_type =='bc': 

        def malignant_benign(directory_name: str) -> int:

            """
            Converts directory name into a new class label (int)
            """
            malignant_classes = ['BCC', 'SCC', 'MEL']

            return 1 if directory_name in malignant_classes else 0
        
        df_use[class_label] = df_use[class_label].map(malignant_benign)

    elif classification_type =='mc': 
    
        def malignant_benign_keloid(directory_name:str) -> int:
            """
            Converts directory name into a new class label (int).

            Args:
            ----
                - directory_name (str): The name of the directory.

            Returns:
            -------
                int: New class label, 1 for malignant (BCC, SCC, MEL), 2 for KLD, and 0 for benign (others).
            """
            malignant_classes = ['BCC', 'SCC', 'MEL']
            if directory_name in malignant_classes:
                return 1
            elif directory_name == 'KLD':
                return 2
            else:
                return 0
           
        df_use[class_label] = df_use[class_label].map(malignant_benign_keloid)
        df_use[class_label] = df_use[class_label].astype(int) 

    elif classification_type =='kl': 
        
        def keloid_lookalikes(directory_name: str) -> int:
            needed_classes = {'SCC': 0, 'BCC': 1, 'KLD': 2}
    
            if directory_name not in needed_classes:
                return None  
    
            return needed_classes[directory_name]  

        # Apply the function and force integer type
        df_use[class_label] = df_use[class_label].map(keloid_lookalikes)
        df_use = df_use[df_use[class_label].notna()]  
        df_use[class_label] = df_use[class_label].astype(int) 
    
    return df_use    


def data_augmentation(df:pd.DataFrame, 
                      class_image_limit: int,  
                      class_label:str='label', 
                      column_list:list = ['image_path', 'label', 'image'])-> object:
    
    """
    Function to perform data augmentation on a given dataframe of image arrays.
    
    The function takes in a dataframe of image arrays, a class image limit, a class label column name, and column names for the image path, label and image. It then performs data augmentation on the images in each class until the number of images in each class is equal to the class image limit. The augmented images are then added to a new dataframe, which is shuffled randomly before being returned.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of image arrays.
    class_image_limit : int
        The number of images each class should have after augmentation.
    class_label : str, optional
        The name of the class label column in the dataframe. Defaults to 'label'.
    column_list : list, optional
        A list of column names for the image path, label and image in the dataframe. Defaults to ['image_path', 'label', 'image'].
    
    Returns
    -------
    pd.DataFrame
        The dataframe with the augmented images.
    """
    new_augment_df = pd.DataFrame(columns=column_list)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for code in df[class_label].unique():
        df_ = df.copy()
        # Get the image arrays for the current class
        image_arrays = df_.loc[df[class_label] == code, 
                               column_list[2]].values
        
        # Calculate the number of additional images needed for the current class
        num_images_needed = class_image_limit - len(image_arrays)
       
        if num_images_needed > 0:

            selected_images = np.random.choice(image_arrays, 
                                               size=num_images_needed)
            
            for image_array in selected_images:

                
                image_tensor = np.expand_dims(image_array,
                                              axis=0)
                
               
                augmented_images = datagen.flow(image_tensor, 
                                                batch_size=1)
        
                # Extract the augmented image arrays and add them to the augmented dataframe
                for i in range(augmented_images.n):
                    
                    augmented_image_array = augmented_images.__next__()[0].astype('float32')
                    
                    row_to_append = pd.DataFrame({column_list[0]: None, 
                                                  column_list[1]: [code], 
                                                  column_list[2]: [augmented_image_array]})

                    new_augment_df = pd.concat([new_augment_df, 
                                                row_to_append], 
                                                ignore_index=True)
            
        original_images_df = df_.loc[df_[class_label] == code, 
                                     column_list]

        new_augment_df = pd.concat([original_images_df, 
                                    new_augment_df], 
                                    axis = 0, 
                                    ignore_index=True)

        df_shuffled= new_augment_df.sample(frac=1, 
                                           random_state=42, 
                                           ignore_index=True)
    return df_shuffled
 
def features_target(df:pd.DataFrame):
    
    """
    Splits a DataFrame into features and target arrays.

    Args:
    ----
        - df: A DataFrame containing the data to be split into features and target arrays.

    Returns:
    -------
        A tuple containing two elements: features and target. The features array is a DataFrame
        containing all columns of the input DataFrame except 'label' and 'image_path'. The target
        array is a Series containing the class labels of the input DataFrame.
    """
    features = df.drop(columns=['label','image_path'],axis=1)
    target = df['label']
    return features, target

def data_oversample(X: np.ndarray, y: np.ndarray) -> dict:

    """
    Randomly oversample the minority class in a dataset.

    Parameters
    ----------
    X : numpy array
        The features of the dataset. It should be either a 2D array
        (e.g. for tabular data) or a 4D array (e.g. for images).
    y : numpy array
        The class labels of the dataset.

    Returns
    -------
    A dictionary containing the resampled features and labels.
    The keys of the dictionary are 'X_resampled' and 'y_resampled'.
    The values are the resampled features and labels, respectively.
    The resampled features are returned as a 4D array if the input X was 4D;
    otherwise, it is returned as a 2D array.
    """
    if len(X.shape) == 2:
        # If X is already a 2D array, no need to reshape
        X_train_reshaped = X
    elif len(X.shape) == 4:
        # Reshape images to 2D arrays
        num_samples, height, width, channels = X.shape
        X_train_reshaped = X.reshape(num_samples, height * width * channels)
    else:
        raise ValueError("Input X must be either 2D or 4D numpy array")

    # Random oversampling
    random_oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = random_oversampler.fit_resample(X_train_reshaped, y)

    # After oversampling, reshape X_resampled back to 4D tensor if needed
    if len(X.shape) == 4:
        X_resampled_4d = X_resampled.reshape(-1, height, width, channels)
    else:
        X_resampled_4d = X_resampled  # No need to reshape if it was originally 2D

    resamp_data = {
        "X_resampled": X_resampled_4d,
        "y_resampled": y_resampled
    }

    return resamp_data


@dataclass(frozen=True)
class TrainTestData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

def standardize_data(
        train_features:np.ndarray,
        train_target: np.ndarray,
        val_features:np.ndarray,
        val_target: np.ndarray,
        test_features:np.ndarray,
        test_target: np.ndarray,
        classification_type:str,
        )->TrainTestData:
    

    """
    Standardizes the data by subtracting the mean and dividing by the standard deviation.
    
    Parameters
    ----------
    train_features : np.ndarray
        The array of training features.
    train_target : np.ndarray
        The array of training target values.
    val_features : np.ndarray
        The array of validation features.
    val_target : np.ndarray
        The array of validation target values.
    test_features : np.ndarray
        The array of testing features.
    test_target : np.ndarray
        The array of testing target values.
    classification_type : str
        The type of classification to perform. Must be one of 'mc', 'kl', or 'bc'.
    
    Returns
    -------
    TrainTestData
        A class instance with the following attributes: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    X_train = np.asarray(train_features['image'].tolist())
    y_train = np.asarray(train_target.tolist())
    X_val = np.asarray(val_features['image'].tolist())
    y_val = np.asarray(val_target.tolist())
    X_test = np.asarray(test_features['image'].tolist())
    y_test = np.asarray(test_target.tolist())

    X_train_mean = np.mean(X_train)
    X_train_std = np.std(X_train)
    std_X_train = (X_train - X_train_mean)/X_train_std

    X_val_mean = np.mean(X_val)
    X_val_std = np.std(X_val)
    std_X_val = (X_val - X_val_mean)/X_val_std

    X_test_mean = np.mean(X_test)
    X_test_std = np.std(X_test)
    std_X_test = (X_test - X_test_mean)/X_test_std

    if classification_type not in("mc", "bc", "kl"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' 'kl' or 'bc' but classification_type: {classification_type} was given ")

    elif classification_type in ('mc', 'kl'): 
        y_train_cat = utils.to_categorical(y_train, 
                                           num_classes = 3)
        y_val_cat = utils.to_categorical(y_val, 
                                          num_classes = 3)
        y_test_cat = utils.to_categorical(y_test, 
                                          num_classes = 3)

        return TrainTestData(X_train=std_X_train, 
                             y_train=y_train_cat,
                             X_val=std_X_val, 
                             y_val=y_val_cat,
                             X_test=std_X_test , 
                             y_test=y_test_cat)
    
    elif classification_type =='bc': 

        return TrainTestData(X_train=std_X_train, 
                             y_train=y_train,
                             X_val=std_X_val, 
                             y_val=y_val,
                             X_test=std_X_test, 
                             y_test=y_test)
@dataclass(frozen=True)
class TestData:
    X_test: np.ndarray
    y_test: np.ndarray    
def standardize_test_data(
        test_features:np.ndarray,
        test_target: np.ndarray,
        classification_type:str,
        )->TestData:
    

    """
    Standardizes test feature data and encodes the target based on classification type.

    This function takes in test features and target arrays, standardizes the feature data,
    and encodes the target labels according to the specified classification type. It supports
    multi-class ('mc'), keloid ('kl'), and binary classification ('bc') types.

    Parameters
    ----------
    test_features : np.ndarray
        An array containing the test features, expected to have an 'image' column.
    test_target : np.ndarray
        An array containing the test target labels.
    classification_type : str
        The type of classification to perform. Must be one of 'mc', 'kl', or 'bc'.

    Returns
    -------
    TestData
        An instance containing standardized test features and appropriately encoded target labels.

    Raises
    ------
    ValueError
        If the `classification_type` is not one of 'mc', 'bc', or 'kl'.
    """

    X_test = np.asarray(test_features['image'].tolist())
    y_test = np.asarray(test_target.tolist())


    X_test_mean = np.mean(X_test)
    X_test_std = np.std(X_test)
    std_X_test = (X_test - X_test_mean)/X_test_std

    if classification_type not in("mc", "bc", "kl"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given ")

    elif classification_type in ('mc', 'kl'): 
        y_test_cat = utils.to_categorical(y_test, 
                                          num_classes = 3)

        return TestData(
                        X_test=std_X_test , 
                        y_test=y_test_cat
                        )
    
    elif classification_type =='bc': 

        return TestData(
                        X_test=std_X_test, 
                        y_test=y_test
                        )

@dataclass(frozen=True)    
class MergeData:
    X_data:np.ndarray
    y_data:np.ndarray

def standardize_and_merge(
        train_features:np.ndarray,
        train_target: np.ndarray,
        val_features:np.ndarray,
        val_target: np.ndarray,
        classification_type:str,
        )-> MergeData:
    

    """
    Standardizes feature data and encodes the target based on classification type.

    This function takes in feature and target arrays for training and validation data,
    standardizes the feature data, and encodes the target labels according to the specified
    classification type. It supports multi-class ('mc'), keloid ('kl'), and binary classification
    ('bc') types.

    Parameters
    ----------
    train_features : np.ndarray
        An array containing the training features, expected to have an 'image' column.
    train_target : np.ndarray
        An array containing the training target labels.
    val_features : np.ndarray
        An array containing the validation features, expected to have an 'image' column.
    val_target : np.ndarray
        An array containing the validation target labels.
    classification_type : str
        The type of classification to perform. Must be one of 'mc', 'kl', or 'bc'.

    Returns
    -------
    MergeData
        An instance containing standardized feature data and appropriately encoded target labels.

    Raises
    ------
    ValueError
        If the `classification_type` is not one of 'mc', 'bc', or 'kl'.
    """
    X_train = np.asarray(train_features['image'].tolist())
    y_train = np.asarray(train_target.tolist())
    X_val = np.asarray(val_features['image'].tolist())
    y_val = np.asarray(val_target.tolist())

    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)

    X_combined_mean = np.mean(X_combined)
    X_combined_std = np.std(X_combined)
    std_X_combined = (X_combined - X_combined_mean)/X_combined_std


    if classification_type not in("mc", "bc", "kl"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc','bc' or 'kl' but classification_type: {classification_type} was given ")

    elif classification_type in ('mc', 'kl'): 
        y_combined_cat = utils.to_categorical(y_combined, 
                                           num_classes = 3)

        return MergeData(
            X_data=std_X_combined, 
            y_data=y_combined_cat)
    
    elif classification_type =='bc': 
        return MergeData(
            X_data = std_X_combined, 
            y_data = y_combined)