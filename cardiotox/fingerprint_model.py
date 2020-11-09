from .model import CardioTox

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.layers import Dense, Dropout
from rdkit import Chem, DataStructs

from PyBioMed.PyMolecule.fingerprint import CalculateECFP2Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint


class FingerprintModel(CardioTox):

    def __init__(self, checkpoint_path="cardiotox/models/training_fp/cp_fp.ckpt"):
        CardioTox.__init__(self, checkpoint_path)
    
    def _calculate_fingerprint(self, smiles):
        features = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            mol_fingerprint = CalculateECFP2Fingerprint(mol)
            pubchem_mol_fingerprint = CalculatePubChemFingerprint(mol)
            feature1 = mol_fingerprint[0]
            feature2 = pubchem_mol_fingerprint
            feature = list(feature1)+list(feature2)
            features.append(feature)
        return np.asarray(features)
        
    def preprocess_smile(self, smiles):
        X = self._calculate_fingerprint(smiles)
        return X

    def model_definition(self):
        n_x_new=1905
        inputs = tf.keras.Input(shape=(n_x_new,))

        x = Dense(200, activation='relu')(inputs)
        x = Dense(200, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(100, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)


        def auc(y_true, y_pred):
            return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)



        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr


        # Build model
        model = Model(inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
        lr_metric = get_lr_metric(optimizer)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc, lr_metric])

        return model
