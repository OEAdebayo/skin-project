from keras import layers, models, optimizers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def get_fitted_model_bc(input_shape: tuple, 
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray | None,
                        y_val: np.ndarray | None,
                        fine_tune: bool = False,
                        custom=None,
                        no_epochs: int = 50,
                        cv: bool = False,
                        basemodel=None, 
                        batch_size: int = 32):
    if custom:
        model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
])
    else:
        pretrained_model = basemodel(weights='imagenet',
                                    include_top=False, 
                                    input_shape=input_shape)

        # Freeze the weights of the pre-trained model
        for layer in pretrained_model.layers:
            layer.trainable = fine_tune

        # Define the Sequential model with a list of layers
        model = models.Sequential([
            pretrained_model,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    if fine_tune==True:
        optimizer = optimizers.Adam(learning_rate=0.00001)
    else:
        optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    if cv:
        return model
    else:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided when cv is False.")
        
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=no_epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=0)
        return model, history

def get_fitted_model_mc(input_shape: tuple, 
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray | None,
                        y_val: np.ndarray | None,
                        fine_tune: bool = False,
                        custom=None,
                        no_epochs: int = 50,
                        cv: bool = False,
                        basemodel=None, 
                        batch_size: int = 32):
    if custom:
        model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax'),
        ])
    else:
        pretrained_model = basemodel(weights='imagenet',
                                    include_top=False, 
                                    input_shape=input_shape)

        # Freeze the weights of the pre-trained model
        for layer in pretrained_model.layers:
            layer.trainable = fine_tune

        # Define the Sequential model with a list of layers
        model = models.Sequential([
            pretrained_model,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])

    if fine_tune==True:
        optimizer = optimizers.Adam(learning_rate=0.00001)
    else:
        optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if cv:
        return model
    else:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided when cv is False.")
        
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=no_epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=0)
        return model, history

num_folds = 5
Skf = StratifiedKFold(n_splits=num_folds)

@dataclass
class CrossValidation:
    avg_accuracy_score: float
    avg_recall: float
    avg_f1: float
    avg_precision: float
    avg_auc: float
    cv_model: models.Model

def cross_val(X: np.ndarray, 
              y: np.ndarray, 
              basemodel, 
              classification_type: str,
              input_shape: tuple,
              fine_tune: bool,
              no_epochs: int) -> CrossValidation:

    num_folds = 5
    Skf = StratifiedKFold(n_splits=num_folds)

    # Initialize the metrics
    acc = []
    prec = []
    rec = []
    f1s = []
    aucs = []

    for train_index, val_index in Skf.split(X, y):
        X_train_c, X_val_c = X[train_index], X[val_index]
        Y_train_c, Y_val_c = y[train_index], y[val_index]

        # Create and fit the model
        if classification_type in ('mc' or 'kl'):
            model = get_fitted_model_mc(input_shape=input_shape,
                                        X_train=X_train_c,
                                        y_train=Y_train_c,
                                        X_val=None,
                                        y_val=None,
                                        basemodel=basemodel,
                                        cv=True,
                                        fine_tune=fine_tune,
                                        no_epochs=no_epochs)
        
            model.fit(X_train_c, 
                      Y_train_c, 
                      epochs=no_epochs, 
                      batch_size=32, 
                      verbose=0)

            Y_pred_prob = model.predict(X_val_c, verbose=0)
            Y_pred = np.argmax(Y_pred_prob, axis=1)
            Y_val_c = np.argmax(Y_val_c, axis=1)

            acc_ = accuracy_score(Y_val_c, Y_pred)
            prec_ = precision_score(Y_val_c, Y_pred, average='weighted')
            rec_ = recall_score(Y_val_c, Y_pred, average='weighted')
            f1_ = f1_score(Y_val_c, Y_pred, average='weighted')
            
            # Binarize the true labels for AUC calculation
            Y_val_c_binarized = label_binarize(Y_val_c, classes=np.arange(Y_pred_prob.shape[1]))
            auc_ = roc_auc_score(Y_val_c_binarized, Y_pred_prob, multi_class='ovr')

        elif classification_type == 'bc':
            model = get_fitted_model_bc(input_shape=input_shape,
                                        X_train=X_train_c,
                                        y_train=Y_train_c,
                                        X_val=None,
                                        y_val=None,
                                        basemodel=basemodel,
                                        cv=True,
                                        fine_tune=fine_tune,
                                        no_epochs=no_epochs)
        
            model.fit(X_train_c, 
                      Y_train_c, 
                      epochs=no_epochs, 
                      batch_size=32, 
                      verbose=0)
            
            Y_pred_prob = model.predict(X_val_c, verbose=0)
            Y_pred = [1 if y > 0.5 else 0 for y in Y_pred_prob]

            Y_val_c = Y_val_c.astype(int)

            acc_ = accuracy_score(Y_val_c, Y_pred)
            prec_ = precision_score(Y_val_c, Y_pred, average='binary')
            rec_ = recall_score(Y_val_c, Y_pred, average='binary')
            f1_ = f1_score(Y_val_c, Y_pred, average='binary')
            auc_ = roc_auc_score(Y_val_c, Y_pred_prob)

        # Collect metrics for this fold
        acc.append(acc_)
        prec.append(prec_)
        rec.append(rec_)
        f1s.append(f1_)
        aucs.append(auc_)
    
    # Compute average metrics
    avg_prec = np.mean(prec)
    avg_rec = np.mean(rec)
    avg_f1 = np.mean(f1s)
    avg_acc = np.mean(acc)
    avg_auc = np.mean(aucs)

    return CrossValidation(avg_accuracy_score=avg_acc,
                           avg_recall=avg_rec,
                           avg_f1=avg_f1,
                           avg_precision=avg_prec,
                           avg_auc=avg_auc,
                           cv_model=model)