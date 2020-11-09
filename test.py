import cardiotox

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef

import numpy as np

model = cardiotox.load_ensemble()

test_set_pos = pd.read_csv("data/external_test_set_pos.csv")
test_set_neg = pd.read_csv("data/external_test_set_neg.csv")

pos_smiles = list(test_set_pos["smiles"])
y_test_ex_fp_pos = test_set_pos["ACTIVITY"]

neg_smiles = list(test_set_neg["smiles"])
y_test_ex_fp_neg = test_set_neg["ACTIVITY"]

###################### Stack ensemble for POSITIVELY BIASED TEST DATA ########################################

pred_test_external_stack_pos = model.predict(pos_smiles)

auc_test_external_stack_pos = roc_auc_score(y_test_ex_fp_pos, pred_test_external_stack_pos)
print("auc_test_external_stack_pos: " + str(auc_test_external_stack_pos))

pred_test_external_stack_pos = np.where(pred_test_external_stack_pos > 0.5, 1, 0)

tn, fp, fn, tp = confusion_matrix(y_test_ex_fp_pos, pred_test_external_stack_pos).ravel()

specificity_test_external_stack_pos = tn / (tn + fp)

sensitivity_test_external_stack_pos = tp / (tp + fn)

NPV_test_external_stack_pos = tn / (tn + fn)

PPV_test_external_stack_pos = tp / (tp + fp)

Accuracy_test_external_stack_pos = balanced_accuracy_score(y_test_ex_fp_pos, pred_test_external_stack_pos)

MCC_test_external_stack_pos = matthews_corrcoef(y_test_ex_fp_pos, pred_test_external_stack_pos)

print("specificity_test_external_stack_pos: " + str(specificity_test_external_stack_pos))
print("sensitivity_test_external_stack_pos: " + str(sensitivity_test_external_stack_pos))

print("NPV_test_external_stack_pos: " + str(NPV_test_external_stack_pos))
print("PPV_test_external_stack_pos: " + str(PPV_test_external_stack_pos))

print("Accuracy_test_external_stack_pos: " + str(Accuracy_test_external_stack_pos))

print("MCC_test_external_stack_pos: " + str(MCC_test_external_stack_pos))


###################### Stack ensemble for NEGATIVELY BIASED TEST DATA ########################################

pred_test_external_stack_neg = model.predict(neg_smiles)

auc_test_external_stack_neg = roc_auc_score(y_test_ex_fp_neg, pred_test_external_stack_neg)
print("auc_test_external_stack_neg: " + str(auc_test_external_stack_neg))

pred_test_external_stack_neg = np.where(pred_test_external_stack_neg > 0.5, 1, 0)

tn, fp, fn, tp = confusion_matrix(y_test_ex_fp_neg, pred_test_external_stack_neg).ravel()

specificity_test_external_stack_neg = tn / (tn + fp)

sensitivity_test_external_stack_neg = tp / (tp + fn)

NPV_test_external_stack_neg = tn / (tn + fn)

PPV_test_external_stack_neg = tp / (tp + fp)

Accuracy_test_external_stack_neg = balanced_accuracy_score(y_test_ex_fp_neg, pred_test_external_stack_neg)

MCC_test_external_stack_neg = matthews_corrcoef(y_test_ex_fp_neg, pred_test_external_stack_neg)

print("specificity_test_external_stack_neg: " + str(specificity_test_external_stack_neg))
print("sensitivity_test_external_stack_neg: " + str(sensitivity_test_external_stack_neg))

print("NPV_test_external_stack_neg: " + str(NPV_test_external_stack_neg))
print("PPV_test_external_stack_neg: " + str(PPV_test_external_stack_neg))

print("Accuracy_test_external_stack_neg: " + str(Accuracy_test_external_stack_neg))

print("MCC_test_external_stack_neg: " + str(MCC_test_external_stack_neg))

