# CardioTox net: A robust predictor for hERG channel blockade via deep learning meta ensembling approaches
##### Abdul Karim, Matthew Lee, Thomas Balle, and Abdul Sattar

This is complementary code for running the models in the paper. Included are the trained models
and the code to load and run inference.

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
### Results

We compared our method using the [test set-I](https://github.com/Abdulk084/CardioTox/blob/master/data/external_test_set_pos.csv) and  [test set-II](https://github.com/Abdulk084/CardioTox/blob/master/data/external_test_set_neg.csv) with other state of the art methods as follows.

**Model** | **MCC** | **NPV** | **ACC** | **PPV** | **SPE** | **SEN** |
CardioTox | 0.599 | 0.688 | 0.810 | 0.893 |0.786 |0.833 |

