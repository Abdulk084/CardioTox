from .model import CardioTox

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from spektral.layers import GraphAttention, GlobalAttentionPool,GraphConv
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.layers import Dense, Dropout
from rdkit import Chem, DataStructs
from spektral.layers import GraphConv


class GraphModel(CardioTox):

    def __init__(self, checkpoint_path="cardiotox/models/training_gc/cp_gc.ckpt"):
        CardioTox.__init__(self, checkpoint_path)
    
    def _convert_smile_to_graph(self, smiles):
        features = []
        adj = []
        maxNumAtoms = 50
        for smile in smiles:
            iMol = Chem.MolFromSmiles(smile)
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append(self.atom_feature(atom) )
            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp 

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            features.append(iFeature)
            adj.append(iAdj)
        features = np.asarray(features)
        adj = np.asarray(adj)
        return features, adj
    
    def convert_to_graph(file_name):
        trfile = open(str(file_name), 'r')
        line = trfile.readline()
        adj = []
        adj_norm = []
        features = []
        maxNumAtoms = 50
        cnt = 0
        new_smiles_list = []
        dataY_train=[]
        for i, line in enumerate(trfile):
            line = line.rstrip().split(',')
            smiles = str(line[1])
            cnt+=1
            iMol = Chem.MolFromSmiles(smiles)
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

            Activity = str(line[0])

            if( iAdjTmp.shape[0] <= maxNumAtoms):
                iFeature = np.zeros((maxNumAtoms, 65))
                iFeatureTmp = []
                for atom in iMol.GetAtoms():
                    iFeatureTmp.append( self.atom_feature(atom) ) ### atom features only
                iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp ### 0 padding for feature-set
                features.append(iFeature)

                # Adj-preprocessing
                iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
                iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
                adj.append(np.asarray(iAdj))
                new_smiles_list.append(smiles)
                dataY_train.append(Activity)

        features = np.asarray(features)
        adj = np.asarray(adj)
        dataY_train = np.array(dataY_train)
        Y = (np.array(dataY_train, dtype=np.float32)).reshape(dataY_train.shape[0],1)
        return features, adj, new_smiles_list, Y
    
    def atom_feature(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    self.one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                    [atom.GetIsAromatic()] + self.get_ring_info(atom))

    
    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))
  
    
    def get_ring_info(self, atom):
        ring_info_feature = []
        for i in range(3, 9):
            if atom.IsInRingSize(i):
                ring_info_feature.append(1)
            else:
                ring_info_feature.append(0)
        return ring_info_feature

        
    def preprocess_smile(self, smiles):
        X, A = self._convert_smile_to_graph(smiles)
        return [X, A]

    def model_definition(self):
        l2_reg = 5e-3            # Regularization rate for l2
        learning_rate = 1e-3     # Learning rate for Adam
        epochs = 100          # Number of training epochs
        batch_size = 32          # Batch size
        N = 50          # Number of nodes in the graphs
        F = 65        # Original feature dimensionality


        # Model definition
        X_in = Input(shape=(N, F))
        A_in = Input((N, N))

        gc1 = GraphConv(64, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
        gc2 = GraphConv(64, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in])

        pool = GlobalAttentionPool(1024)(gc2)


        dense1=Dense(1024, activation='relu')(pool)
        dense2=Dense(1024, activation='relu')(dense1)

        output = Dense(1, activation='sigmoid')(dense2)


        def auc(y_true, y_pred):
            return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)



        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr


        # Build model
        model = Model(inputs=[X_in, A_in], outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
        lr_metric = get_lr_metric(optimizer)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc, lr_metric])

        return model
