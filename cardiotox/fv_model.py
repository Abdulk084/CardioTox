from .model import CardioTox

import numpy as np
import tensorflow as tf
import keras

from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.metrics import roc_auc_score

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class FVModel(CardioTox):

    def __init__(self, checkpoint_path="cardiotox/models/training_fv/cp_fv.ckpt"):
        CardioTox.__init__(self, checkpoint_path)
    
    def _convert_to_fv(self, smiles):
        bit_size=1024
        Max_len=93
        dataX = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bit_size)
            fp = np.array(fp)
            dataX.append(fp)

        dataX = np.array(dataX)

        data_x = []

        for i in range(len(dataX)):
            fp = [0] * Max_len
            n_ones = 0
            for j in range(bit_size):
                if dataX[i][j] == 1:
                    fp[n_ones] = j+1
                    n_ones += 1
            data_x.append(fp)

        data_x = np.array(data_x, dtype=np.int32)


        return data_x
        
    def preprocess_smile(self, smiles):
        X = self._convert_to_fv(smiles)
        return X

    def model_definition(self):
        n_x_new=93
        inputs = tf.keras.Input(shape=(n_x_new,))


        x = Embedding(1025, 200, input_length=n_x_new)(inputs)
        x = keras.layers.Conv1D(192, 10, activation='relu')(x)
        x = keras.layers.Conv1D(192, 5, activation='relu')(x)
        x = keras.layers.Conv1D(192, 3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(100, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        def auc(y_true, y_pred):
            return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)




        # Build model
        model = Model(inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)


        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc])

        return model
