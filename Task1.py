import numpy as np


def SVD(matrix):
    AtA = np.dot(matrix.T, matrix)
    AAt = np.dot(matrix, matrix.T)

    eigenvals_AtA, eigenvec_AtA = np.linalg.eig(AtA)
    eigenvals_AAt, eigenvec_AAt = np.linalg.eig(AAt)

    sorted_indices_AtA = np.argsort(eigenvals_AtA)[::-1]
    sorted_indices_AAt = np.argsort(eigenvals_AAt)[::-1]

    # sorting values
    sorted_eigenvals_AtA = eigenvals_AtA[sorted_indices_AtA]
    sorted_eigenvals_AAt = eigenvals_AAt[sorted_indices_AAt]

    # sorting vectors
    sorted_eigenvecs_AtA = eigenvec_AtA[:, sorted_indices_AtA]
    sorted_eigenvecs_AAt = eigenvec_AAt[:, sorted_indices_AAt]

    singular_values = np.sqrt(sorted_eigenvals_AtA)

    sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
    np.fill_diagonal(sigma, singular_values)

    U = sorted_eigenvecs_AAt
    V = sorted_eigenvecs_AtA

    return U, sigma, V.T


A = np.array([[1, 2], [3, 4], [5, 6]])
U, Sigma, VT = SVD(A)

print("U:\n", U)
print("Sigma:\n", Sigma)
print("V^T:\n", VT)

print("Reconstructed A:\n", np.dot(np.dot(U, Sigma), VT))