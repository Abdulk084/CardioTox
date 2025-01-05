# CardioTox net: A robust predictor for hERG channel blockade via deep learning meta ensembling approaches
##### Abdul Karim, Matthew Lee, Thomas Balle, and Abdul Sattar

This is complementary code for running the models in the paper https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00541-z

## Installation

Tested on Ubuntu 20.04 with Python 3.7.7

1. Install conda dependency manager https://docs.conda.io/en/latest/ 
2. Restore environment.yml:
```
conda env create -f environment.yml 
```
3. Activate environment: 
```
conda activate cardiotox
```
4. Install pyBioMed:
```
cd PyBioMed
python setup.py install
cd ..
```
5. Test model: 
```
python test.py
```
This will test the model on two external data sets mentioned in the paper.

## Usage

### Run Ensemble
#### Single SMILE String
```python
import cardiotox

smile = "CC(=O)SC1CC2=CC(=O)CCC2(C)C2CCC3C(CCC34CCC(=O)O4)C12"

model = cardiotox.load_ensemble()

model.predict(smile)
``` 

#### Multiple SMILE Strings
```python
import cardiotox

smiles = [
    "CC(=O)SC1CC2=CC(=O)CCC2(C)C2CCC3C(CCC34CCC(=O)O4)C12",
    "CCCCCCCCCC[N+](CC)(CC)CC"
]

model = cardiotox.load_ensemble()

model.predict(smiles)
``` 

### Run Individual Models
#### Import the model you want
```python
from cardiotox import DescModel, SVModel, FVModel,  FingerprintModel
```

#### Run the model the same way as ensemble
```python
from cardiotox import SVModel

smile = "CCCCCCCCCC[N+](CC)(CC)CC"

model = SVModel()

model.predict(smile)
```

### Run Preprocessing
Each model performs its own preprocessing. When 'predict' is called, the preprocessing is 
performed before running the model. This can be accessed by calling the 'preprocess_smile' 
function.
```python
from cardiotox import SVModel

smile = "CCCCCCCCCC[N+](CC)(CC)CC"

model = SVModel()

preprocessed_smile = model.preprocess_smile([smile]) # Expects a list of smiles

model.predict_preprocessed(preprocessed_smile)

```

### Pairwise Tanimoto similarity 

We make sure that none of the molecule in both test sets ([test set-I](https://github.com/Abdulk084/CardioTox/blob/master/data/external_test_set_pos.csv), [test set-II](https://github.com/Abdulk084/CardioTox/blob/master/data/external_test_set_neg.csv)) are similar to trainining set ([training](https://github.com/Abdulk084/CardioTox/blob/master/data/train_validation_cardio_tox_data.tar.xz)) and to each other as well.




![Pairwise Tanimoto similarity bins](https://github.com/Abdulk084/CardioTox/blob/master/sim.png?raw=true "Title")




### Results

We compared our method using the [test set-I](https://github.com/Abdulk084/CardioTox/blob/master/data/external_test_set_pos.csv) and  [test set-II](https://github.com/Abdulk084/CardioTox/blob/master/data/external_test_set_neg.csv) with other state of the art methods as follows.



<table>
<tr><th>Test set-I  </th><th>Test set-II </th></tr>
<tr><td>


                                                                      
 Methods | MCC | NPV | ACC | PPV | SPE | SEN  
 ------- | --- | --- | --- | --- | --- | ---
 CardioTox | 0.599 | 0.688 | 0.810 | 0.893 | 0.786 | 0.833
 DeepHIT | 0.476 | 0.643 | 0.773 | 0.833 | 0.643 | 0.833
 CardPred | 0.193 | 0.643 | 0.614 | 0.760 | 0.571 | 0.633
 OCHEM Predictor-I | 0.149 | 0.333 | 0.364 | 1.000 | 1.000 | 0.067
 OCHEM Predictor-II | 0.164 | 0.351 | 0.432 | 0.857 | 0.929 | 0.200
 Pred-hERG 4.2 | 0.306 | 0.538 | 0.705 | 0.774 | 0.500 | 0.800
</td><td>

 Methods | MCC | NPV | ACC | PPV | SPE | SEN  
 ------- | --- | --- | --- | --- | --- | ---
 CardioTox | 0.469 | 0.947 | 0.758 | 0.478 | 0.600 | 0.917
 DeepHIT |0.398|0.941|0.721|0.417|0.533|0.909
 CardPred |0.049|0.750|0.527|0.294|0.600|0.454
 OCHEM Predictor-I |0.372|0.800|0.648|0.666|0.933|0.364
 OCHEM Predictor-II |0.310|0.794|0.632|0.571|0.900|0.364
 Pred-hERG 4.2 |0.146|0.813|0.580|0.320|0.433|0.727

</td></tr> </table>

#### Note: Only suitable for SMILES with Maximum number of 1's in MorganFingerprint <= 93.



