from .model import CardioTox

import tensorflow as tf
import numpy as np
import pickle

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.layers import Dense, Dropout
from rdkit import Chem, DataStructs
from mordred import Calculator, descriptors


class DescModel(CardioTox):

    def __init__(self, checkpoint_path="cardiotox/models/training_desc/cp_desc.ckpt", desc_file="cardiotox/models/training_desc/des_file.txt"):
        CardioTox.__init__(self, checkpoint_path)
        self.desc_file = desc_file
        self.descriptor_names = []
        with open(self.desc_file, 'r') as fp:
            for line in fp:
                self.descriptor_names.append(line.strip())
        self._load_scaler()

    def _normalize_data(self, X):
        return self.normalizer.transform(X)

    def _generate_scalar(self):
        pass

    def _load_scaler(self):
        try:
            pickle_in = open("cardiotox/models/training_desc/normalizer.pickle", "rb")
            self.normalizer = pickle.load(pickle_in)
        except:
            self.normalizer = self._generate_scalar()

    def _calculate_mordred_descriptors(self, smiles_list, desc_file):
        calc = Calculator(descriptors, ignore_3D = True)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        df = calc.pandas(mols)

        new_df = df[self.descriptor_names]
        return new_df

    def _calculate_desc(self, file_name, desc_file): 
        trfile = open(str(file_name), 'r')
        line = trfile.readline()
        
        
        mols_train=[]
        dataY_train=[]
        smiles_train=[]

        for i, line in enumerate(trfile):
            line = line.rstrip().split(',')
            smiles = str(line[1])
            smiles_train.append(smiles)
            Activity = str(line[0])
            mol = Chem.MolFromSmiles(smiles)
            mols_train.append(smiles)
            dataY_train.append(Activity)

        trfile.close()

        dataY_train = np.array(dataY_train)
        
        Y = (np.array(dataY_train, dtype=np.float32)).reshape(dataY_train.shape[0],1)
       
        smi_total= smiles_train
        data_features  = self._calculate_mordred_descriptors(smi_total, desc_file)
        return data_features, Y


    def preprocess_smile(self, smiles):
        smile_df = self._calculate_mordred_descriptors(smiles, self.desc_file)
        smile_df = self._normalize_data(smile_df)
        smile_df = np.nan_to_num(smile_df)
        return smile_df
    
    def model_definition(self):
        inputs = tf.keras.Input(shape=(995,))
        
        x = Dense(2000, activation='relu')(inputs)
        x = Dense(2000, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.0001), activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1000, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.0001), activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(200, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)        

        def auc(y_true, y_pred):
                return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr

        # Build model
        model = Model(inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(lr=10e-4)
        lr_metric = get_lr_metric(optimizer)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc, lr_metric])
        
        return model
