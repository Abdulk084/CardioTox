from abc import ABC, abstractmethod
import numpy as np


class CardioTox(ABC):
    """
    Abstract class

    This class is a base and all models extend from it. It has some common functions
    such as 'predict' and 'load_model', as well as some abstract functions such as 'preprocess_smile'
    that require each model to implement their own version of that function.
    """

    def __init__(self, checkpoint_path, members=None):
        """
        :param checkpoint_path: The models checkpoint path used for loading the model
        :param members: Only used for the ensemble model. It is a list of CardioTox objects
            that the ensemble loads when initiated.
        """
        self.members = members
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model(self.checkpoint_path)

    @abstractmethod
    def preprocess_smile(self, smile):
        """ Returns preprocessed smile for the model
        Each model preprocesses the SMILE string in their own way. This abstract
        function is to be implemented in each of the models to handle the requisite
        preprocessing that the model requires.
        """
        pass

    @abstractmethod
    def model_definition(self):
        """ Defines the model and returns it
        Each model is unique and requires its own implementation of the model definition.
        """
        pass

    def load_model(self, checkpoint_path):
        """ Loads the model and weights
        Each model is loaded the same way except for the model definition (which is defined individually
        for each model)
        """
        model = self.model_definition()
        model.load_weights(checkpoint_path)
        return model

    def predict(self, smiles, probabilities=False):
        """ Returns the models prediction for a list of smiles
        :param smiles: list of smiles to apply the model to
        :param probabilities: If True, the model produces classification probabilities
            [hERG prob, Non-hERG prob] -> [0.2, 0.8]
                instead of
            [hERG prob]
        """
        if isinstance(smiles, list) is False:
            smiles = [smiles]
        # Removes any spaces in the smile string
        smiles = [smile.replace(" ", "") for smile in smiles]
        smile_preprocessed = self.preprocess_smile(smiles)
        out = self.model.predict(smile_preprocessed)
        if probabilities:
            return self.probabilities(out)
        return out

    def predict_preprocessed(self, features, probabilities=False):
        """ Returns the models prediction for a list of preprocessed features
        :param features: Matrix of preprocessed smiles in numpy format
            [Num-of-instances, feature-size]
        :param probabilities: If True, the model produces classification probabilities
            [hERG prob, Non-hERG prob] -> [0.2, 0.8]
                instead of
            [hERG prob]

        You can retrieve the preprocessed features matrix by calling 'preprocess_smile' on a list of smiles
        """
        out = self.model.predict(features)
        if probabilities:
            return self.probabilities(out)
        return out

    def predict_probabilities(self, smiles):
        """
        Same as 'predict' if probabilities was True
        Redundant but left in as a helper function
        """
        out = self.predict(smiles)
        return self.probabilities(out)
    
    def probabilities(self, out):
        ''' Returns probabilities for each class
        :param out: The output from running predict. A list of probabilities for hERG

        While the model outputs a single probability (0 < x < 1), LIME
        expects the format to be two classes (With probabilties for each class).

        This is easily accomplished by taking the output as the probability for class HeRG
        and subtracting that same probability from 1 to assign the remainder to non-HeRG

            Non-hERG-probability = 1 - hERG-probability
        '''
        class_probs = []
        for x in out:
            herg = x
            nonherg = 1 - x 
            class_probs.append((nonherg, herg))
            
        # Non-herg, Herg
        return np.array(class_probs).squeeze(axis=2)

