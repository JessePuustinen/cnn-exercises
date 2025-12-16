

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# Load dataset
data = pd.read_csv("Imbalanced_data.csv", header=None)
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)



bsmote = BorderlineSMOTE(random_state=42)
X_bsmote, y_bsmote = bsmote.fit_resample(X_train, y_train)



plt.figure(figsize=(15,5))

# Original
plt.subplot(1,3,1)
plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], label='Class 0', alpha=0.5)
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], label='Class 1', alpha=0.5)
plt.title("Original Data")
plt.legend()

# SMOTE
plt.subplot(1,3,2)
plt.scatter(X_smote[y_smote==0][:,0], X_smote[y_smote==0][:,1], label='Class 0', alpha=0.5)
plt.scatter(X_smote[y_smote==1][:,0], X_smote[y_smote==1][:,1], label='Class 1', alpha=0.5)
plt.title("SMOTE Oversampled Data")
plt.legend()

# Borderline-SMOTE
plt.subplot(1,3,3)
plt.scatter(X_bsmote[y_bsmote==0][:,0], X_bsmote[y_bsmote==0][:,1], label='Class 0', alpha=0.5)
plt.scatter(X_bsmote[y_bsmote==1][:,0], X_bsmote[y_bsmote==1][:,1], label='Class 1', alpha=0.5)
plt.title("Borderline-SMOTE Oversampled Data")
plt.legend()

plt.tight_layout()
plt.show()
