from qpsolvers import solve_qp
import numpy as np

EPSILON = 1e-6

class SVM:
	def train(self, X, y, C):
		self.X = X
		self.y = y
		self.C = C

		P = np.dot(X, X.T)
		for i in range(4):
			for j in range(4):
				P[i,j] = P[i,j] * y[i] * y[j]

		P +=  EPSILON * np.identity(4)
		q = -1.0 * np.ones(4)

		G = np.zeros((2*4, 4))
		G[0:4] = -1.0 * np.identity(4)
		G[4:8] = 1.0 * np.identity(4)

		h =  np.zeros(2*4)
		h[0:4] = 1.0 * np.zeros(4)
		h[4:8] = 1.0 * C


		A = y.reshape((1,4))

		b = np.zeros(1)

		multipliers = solve_qp(P, q, G, h, A, b)
		self.multipliers = multipliers

		support_indices = multipliers > EPSILON
		self.support_indices = support_indices
		support_multipliers = multipliers[support_indices]
		support_vectors = X[support_indices]

		weight = np.zeros_like(X[0])
		for i in range(support_vectors.shape[0]):
			weight += support_multipliers[i] * support_vectors[i]

		margin_indices = np.logical_and(multipliers > EPSILON,
								multipliers < C -EPSILON)
		self.margin_indices = margin_indices
