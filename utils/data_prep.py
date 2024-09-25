import os
import pandas as pd
import numpy as np
import pathlib as Path
import multiprocessing
from PIL import Image
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from keras import utils
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out WARNING, 3 = filter out ERROR


def data_loading(data_dir: Path)-> object:

    """
    Function that converts the train and test data into a dataframe.

    Args:
    ----
        -data:   a directory to the image data for training

    Returns:
    -------
           A DataFrames object
    """
    
    # Create an empty dataframes
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
    """
    Function converts the classes in the dataframe to either malignant or benign.

    Args:
    ----
        -df:                     a DataFrame object with 10 classes
        -classification_type     a str of maliangnat vs benign--'bc' or maliagnant vs benign vs keloids--'mc'
        -resize(tuple):          a tuple of integers (a(int), b(int)) containing the new size we wish to have
        -image(str):             a column in df that takes exactly the string 'image_path' if given. Should be given if and only if arg: resize is given
        -image_path(str):        a column in df that takes exactly the string 'image_path' if given. Should be given if and only if arg: resize is given
        -class_label(str):       the old label to drop after reclassifying df 
        -new_column(str):        The name of the new label 
    
    Returns:
    -------
           A DataFrame object with two classes: 1 -> malignant, 0 -> benign or three classes 2-> Keloid, 1 -> malignant, 0 -> benign


    """
    #if (resize and image and image_path):
       
    # Get the number of CPU cores available
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
    if classification_type not in ("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was giving ")

    
    elif classification_type =='bc': 

        def malignant_benign(directory_name: str) -> int:

            """
            Converts directory name into a new class label (int)
            """
            malignant_classes = ['BCC', 'SCC', 'MEL']

            return 1 if directory_name in malignant_classes else 0
        
        df_use[class_label] = df_use[class_label].map(malignant_benign)

    elif classification_type =='mc': 
    
        def malignant_benign_keloid(directory_name:int) -> int:
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
    
    return df_use    


def data_augmentation(df:pd.DataFrame, 
                      class_image_limit: int,  
                      class_label:str='label', 
                      column_list:list = ['image_path', 'label', 'image'])-> object:
    """
    Function to augment a dataframe.

    Args:
    ----
        -df:                    a DataFrame object
        -class_image_limit:     expected maximum number of images  per class
        -column_names:          a list of strings containing the column names for the augmented dataframe in this order ["image_path", "classcode", "image"]
    
    Returns:
    -------
           A DataFrame object with two classes containing "class_image_limit" number of images per class


    """
    new_augment_df = pd.DataFrame(columns=column_list)

        # Create an ImageDataGenerator object with the desired transformations

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
       
        # Generate augmented images for the current class
        if num_images_needed > 0:

            # Select (size e.g, 5) random subset of the original images (i.e., image_arrays) 
            selected_images = np.random.choice(image_arrays, 
                                               size=num_images_needed)
            
            # Apply transformations to the selected images and add them to the augmented dataframe
            for image_array in selected_images:

                # Reshape the image array to a 4D tensor with a batch size of 1
                image_tensor = np.expand_dims(image_array,
                                              axis=0)
                
                # Generate the augmented images
                augmented_images = datagen.flow(image_tensor, 
                                                batch_size=1)
        
                # Extract the augmented image arrays and add them to the augmented dataframe
                for i in range(augmented_images.n):
                    #augmented_image_array = augmented_images.next()[0].astype('uint8')
                    augmented_image_array = augmented_images.__next__()[0].astype('float32')
                    
                    #augmented_df = augmented_df.append({'image_path': None, 'label': class_label, 'image': augmented_image_array}, ignore_index=True)
                    row_to_append = pd.DataFrame({column_list[0]: None, 
                                                  column_list[1]: [code], 
                                                  column_list[2]: [augmented_image_array]})

                    # Concatenate the new row with the original DataFrame
                    new_augment_df = pd.concat([new_augment_df, 
                                                row_to_append], 
                                                ignore_index=True)
            #num_images_needed-=1
                    # Add the original images for the current class to the augmented dataframe
        original_images_df = df_.loc[df_[class_label] == code, 
                                     column_list]

        #print(augmented_df.shape, original_images_df.shape)
        new_augment_df = pd.concat([original_images_df, 
                                    new_augment_df], 
                                    axis = 0, 
                                    ignore_index=True)

        # Shuffle the dataframe for further processing
        df_shuffled= new_augment_df.sample(frac=1, 
                                           random_state=42, 
                                           ignore_index=True)
    return df_shuffled

# Function to... 
def features_target(df:pd.DataFrame):
    """
    Function to extract features and target from a dataframe
    
    """
    features = df.drop(columns=['label','image_path'],axis=1)
    target = df['label']
    return features, target

def data_oversample(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Function to oversample data.

    Args:
    ----
        - X: A multi-dimensional numpy array representing the pixels of an image.
        - y: A 1-dimensional numpy array of integers containing the class of the features in X.

    Returns:
    -------
        A dictionary containing the resampled X--features and y--target values.
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
    Function to standardize the dataset.
    Args:
    ----

        train_features (array):               an array of train feautures
        train_target (array):                 an array of train target values
        val_features (array):                 an array of validation feautures
        val_target (array):                   an array of validation target values
        test_features (array):                an array of test feautures
        test_target (array):                  an array of test target values
        classification_type (str):            default("mc") set to "bc" for binary classification 

    Returns:
    -------
            A class instance with these attributes: X_train, y_train, X_val, y_val, X_test, y_test.

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

    if classification_type not in("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given ")

    elif classification_type =='mc': 
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
    Function to standardize the dataset.
    Args:
    ----
        test_features (array):                an array of test feautures
        test_target  (array):                 an array of test target values
        classification_type (str):            default("mc") set to "bc" for binary classification 

    Returns:
    -------
            A class instance with these attributes: X_train, y_train, X_val, y_val, X_test, y_test.

    """

    X_test = np.asarray(test_features['image'].tolist())
    y_test = np.asarray(test_target.tolist())


    X_test_mean = np.mean(X_test)
    X_test_std = np.std(X_test)
    std_X_test = (X_test - X_test_mean)/X_test_std

    if classification_type not in("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given ")

    elif classification_type =='mc': 
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
    Function to split the data into train and validation set.
    Args:
    ----

        train_features:               an array of train feautures
        train_target:                 an array of train target values
        val_features:                 an array of validation feautures
        val_target:                   an array of validation target values
        classification_type:          default("mc") set to "bc" for binary classification

    Returns:
    -------
            A class instance with these attributes: X_data, y_data.

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


    if classification_type not in("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given ")

    elif classification_type =='mc': 
        y_combined_cat = utils.to_categorical(y_combined, 
                                           num_classes = 3)

        return MergeData(
            X_data=std_X_combined, 
            y_data=y_combined_cat)
    
    elif classification_type =='bc': 
        return MergeData(
            X_data = std_X_combined, 
            y_data = y_combined)