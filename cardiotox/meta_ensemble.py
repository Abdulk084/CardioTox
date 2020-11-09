from .model import CardioTox

import tensorflow as tf

from tensorflow.python.keras.models import Model
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.merge import concatenate

from .desc_model import DescModel
from .fv_model import FVModel
from .sv_model import SVModel
from .fingerprint_model import FingerprintModel
from os.path import join


class EnsembleModel(CardioTox):

    def __init__(self, members, checkpoint_path="cardiotox/models/training_stack/cp_st.ckpt"):
        CardioTox.__init__(self, checkpoint_path, members)
        self.model_order = ["fp", "dm", "sv", "fv"]
    
    def get_model(self, model_name):
        if model_name in self.model_order:
            model_index = self.model_order.index(model_name)
            return self.get_model_by_index(model_index)
        else:
            raise ValueError("Model {} not in list of models {}".format(model_name, self.model_order))
        
    def get_model_by_index(self, index):
        return self.members[index]
    
    def _preprocess_per_member(self, smiles):
        preprocessed_smiles = []
        for member in self.members:
            s = member.preprocess_smile(smiles)
            preprocessed_smiles.append(s)
        return preprocessed_smiles
        
    def preprocess_smile(self, smiles):
        X = self._preprocess_per_member(smiles)
        return X

    def model_definition(self):
        for i in range(len(self.members)):
            model = self.members[i].model
            for layer in model.layers:

                layer.trainable = False
                #print(layer.name)
                layer._name = 'ensemble_' + str(i+1) + '_' + layer.name 
                #print(layer.name)

            
        ensemble_visible = [model.model.input for model in self.members]   

        ensemble_outputs = [model.model.output for model in self.members]

        #concat = tf.keras.layers.Concatenate(axis=1)


        merge = concatenate(ensemble_outputs)


        hidden_1 = Dense(100, activation='relu')(merge)
        hidden_2 = Dense(10, activation='relu')(hidden_1)

        output = Dense(1, activation='sigmoid')(hidden_2)

        def auc(y_true, y_pred):
            return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)



        model = Model(inputs=ensemble_visible, outputs=output)


        optimizer = tf.keras.optimizers.Adam(learning_rate=10e-3)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc])
        #plot_model(model, show_shapes=True, to_file='model_graph.png')
        return model

    
def load_ensemble():
    ROOT = "/"
    # Graph Model
    #gm = GraphModel()
    
    # Desc Model
    dm = DescModel()
    
    # Fingerprint Model
    fp = FingerprintModel()

    # FV Model
    fv = FVModel()
    
    # SV Model
    sv = SVModel()
    
    # Ensemble Model
    members=[fp, dm, sv, fv]
    ensemble = EnsembleModel(members)
    
    return ensemble
