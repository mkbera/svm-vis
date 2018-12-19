import numpy as np
from svm import SVM

EPSILON = 1e-6

C = 10.0

X = np.array([
		[1, 0],
		[-1, 0],
		[1000, 0],
		[-1000, 0],
	], dtype=float)


y = np.array([-1, 1, 1, -1], dtype=float)

clf = SVM()

clf.train(X, y, C)

print(clf.multipliers)
print('margin indices: ', clf.margin_indices)
print('support_indices: ', clf.support_indices)


