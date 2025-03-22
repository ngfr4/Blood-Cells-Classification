import numpy as np

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl


class Model:
    def __init__(self):
        """
        Initialize the internal state of the model. Note that the __init__
        method cannot accept any arguments.

        The following is an example loading the weights of a pre-trained
        model.
        """
        self.ensemble = [
            tfk.models.load_model('convnext_ft.keras'),
            tfk.models.load_model('efficientnet.keras')
        ]

    def predict(self, X):
        """
        Predict the labels corresponding to the input X. Note that X is a numpy
        array of shape (n_samples, 96, 96, 3) and the output should be a numpy
        array of shape (n_samples,). Therefore, outputs must no be one-hot
        encoded.

        The following is an example of a prediction from the pre-trained model
        loaded in the __init__ method.
        """
        preds = [model.predict(X) for model in self.ensemble]
        preds = np.stack(preds, axis=0)
        preds = np.mean(preds, axis=0)
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        return preds
