import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# # Load dataset
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

# First, delete some trivial features
del df['type']
del df['nameOrig']
del df['nameDest']
del df['isFlaggedFraud']

# Assign labels and features for training
X = df[['step', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']]
y = df['isFraud']

# Split dataset to generate training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# PCA
X = np.array(X).astype(float)
# Subtract sample mean
X -= np.mean(X, axis=1)[:, np.newaxis]
# Get eigenValues and eigenVectors by using Singular Value Decomposition
U, W, _ = np.linalg.svd(X, full_matrices=False)
W = W ** 2 / (X.shape[1] - 1)
# Calculate cumulative sum of eigenValues
S = np.cumsum(W)
R = S / S[-1]
# Plot cumulative sum
plt.plot(R, 'ro-')
plt.show()
# Taking the first Kth eigenVectors to reduce dimensions, where their corresponding eigenValues' cumulative sum
# should be smaller than 0.9.
n = (R < 0.9).sum()
# Data projection to lower dimensions.
A = U[:, :n].T.dot(X)
print (X.shape, A.shape)
print (W)
print (U.shape)