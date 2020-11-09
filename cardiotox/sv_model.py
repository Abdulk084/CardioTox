from .model import CardioTox

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.metrics import roc_auc_score
from itertools import chain, repeat, islice
import re


class SVModel(CardioTox):

    def __init__(self, checkpoint_path="cardiotox/models/training_sv/cp_sv.ckpt"):
        CardioTox.__init__(self, checkpoint_path)
        self.items_list=[ '$', '^', '#', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '=', 'Br', 
               'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[2H]', '[Br-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', 
               '[Cl-]', '[H]', '[I-]', '[N+]', '[N-]', '[N@+]', '[N@@+]', '[NH+]', '[NH2+]', '[NH3+]', '[N]', 
               '[Na+]', '[O-]', '[P+]', '[S+]', '[S-]', '[S@+]', '[S@@+]', '[SH]', '[Si]', '[n+]', '[n-]', 
               '[nH+]', '[nH]', '[o+]', '[se]', '\\', 'c', 'n', 'o', 's', '!', 'E']
        self.charset = list(set(self.items_list))
        self.charset.sort()
        self.char_to_int = dict((c,i) for i,c in enumerate(self.charset))
        self.int_to_char = dict((i,c) for i,c in enumerate(self.charset))
        self.pattern = '|'.join(re.escape(item) for item in self.items_list)
    
    def _convert_to_vectors(self, smiles):
    
        X_smiles_array=np.asarray(smiles)

        
        

        def pad_infinite(iterable, padding=None):
            return chain(iterable, repeat(padding))

        def pad(iterable, size, padding=None):
            return islice(pad_infinite(iterable, padding), size)


        token_list=[]
        X=[]
        for smiles in X_smiles_array:
            tokens = re.findall(self.pattern, smiles)
            tokens =list(pad(tokens, 97, 'E' ))

            x=[self.char_to_int[k] for k in tokens]

            token_list.append(tokens)
            X.append(x)

        X=np.asarray(X)

        return X
        
    def preprocess_smile(self, smiles):
        X = self._convert_to_vectors(smiles)
        return X

    def model_definition(self):
        l2_reg = 5e-5            # Regularization rate for l2
        learning_rate = 1e-4     # Learning rate for Adam
        epochs = 100           # Number of training epochs
        batch_size = 32          # Batch size



        embed = 97  # this is length of the longest smile

        vocab_size = 64
        n_x_new=97
        inputs = tf.keras.Input(shape=(n_x_new,))


        x = Embedding(vocab_size, 200, input_length=embed-1)(inputs)
        x = keras.layers.Conv1D(192,10,activation='relu')(x)
        x = keras.layers.Conv1D(192,5,activation='relu')(x)
        x = keras.layers.Conv1D(192,3,activation='relu')(x)
        x = Flatten()(x)
        x = Dense(100, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        def auc(y_true, y_pred):
            return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)




        # Build model
        model = Model(inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)


        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc])

        return model
    
    def index_to_tokens(self, index_lists):
        token_lists = []
        for n in index_lists:
            token_lists.append([self.int_to_char[k] for k in n])
        return token_lists
    
    def index_to_embedding(self, index_lists):
        embedding_weights = self.model.get_weights()[0]
        embeddings = []
        for indexes in index_lists:
            embed = [embedding_weights[x] for x in indexes]
            embeddings.append(np.array(embed))
        return np.array(embeddings)
    
    def embedding_to_index(self, embedding):
        embedding_weights = self.model.get_weights()[0]
        try:
            return [i for i, x in enumerate(embedding_weights) if (x == embedding).all()][0]
        except IndexError:
            print("Failed with embedding:", embedding)
            
    def embeddings_to_index(self, embeddings):
        x = []
        for e in embeddings:
            e_indexes = np.array(list(map(self.embedding_to_index, e)))
            x.append(e_indexes)
        x = np.array(x)
        return x
    
    def update_to_shap_model(self):
        l2_reg = 5e-5            # Regularization rate for l2
        learning_rate = 1e-4     # Learning rate for Adam
        epochs = 100           # Number of training epochs
        batch_size = 32          # Batch size



        embed = 97  # this is length of the longest smile

        vocab_size = 64
        n_x_new=97
        inputs = tf.keras.Input(shape=(97, 200))

        x = keras.layers.Conv1D(192,10,activation='relu')(inputs)
        x = keras.layers.Conv1D(192,5,activation='relu')(x)
        x = keras.layers.Conv1D(192,3,activation='relu')(x)
        x = Flatten()(x)
        x = Dense(100,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(100, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        def auc(y_true, y_pred):
            return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


        # Build model
        model = Model(inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)


        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc])
        
        model.set_weights(self.model.get_weights()[1:])
        
        self.model = model
        return model
    
    def shap_model_predict(self, embeddings):
        """
        SHAP can't handle the embedding layer of this model.
        The following code accepts embeddings, converts them back to indexes, 
        runs the model, and returns probabilities.
        
        Pass this function to create the SHAP explainer and pass embeddings instead of
        the indexes
        """
        
        
        x = []
        for e in embeddings:
            e_indexes = np.array(list(map(self.embedding_to_index, e)))
            x.append(e_indexes)
        x = np.array(x)
        out = self.model.predict(x)
        out = self.probabilities(out)
        return out
