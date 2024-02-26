import torch
import numpy as np

def identity(table: np.ndarray) -> int:
    """Returns the index of the identity element.
    Input:
        table: np.array of shape [n, n] where the entry at [i, j] is the index of the product of the ith and jth elements in the group.
    Output:
        Index of identity element.
    Raises:
        ValueError("No or multiple identities") if there is no or multiple identities.
    """
    n = table.shape[0]
    (i,) = np.nonzero(np.all((table == np.arange(n)), axis=1))
    if (len(i) != 1):
        raise ValueError('No or multiple identities')
    return i[0]

def inverses(table: np.ndarray) -> np.ndarray:
    """Returns the indices of the inverses of each element.
    Input:
        table: np.array of shape [n, n] where the entry at [i, j] is the index of the product of the ith and jth elements in the group.
    Output:
        np.array of shape [n] where the ith entry is the index of the inverse of the ith element.
    Raises:
        ValueError("Every element does not have one inverse") if there is no or multiple inverses.
    """
    n = table.shape[0]
    e = identity(table)
    (i, j) = np.nonzero((table == e))
    if ((len(i) != n) or (not np.all((i == np.arange(n))))):
        raise ValueError('Every element does not have one inverse')
    return j


def make_multiplication_table(matrices: np.ndarray, *, tol: float=1e-08) -> np.ndarray:
    """Makes multiplication table for group.
    Input:
        matrices: np.array of shape [n, d, d], n matrices of dimension d that form a group under matrix multiplication.
        tol: float numberical tolerance
    Output:
        Group multiplication table.
        np.array of shape [n, n] where entries correspond to indices of first dim of matrices.
    """
    (n, d, d2) = matrices.shape
    assert (d == d2)
    mtables = np.einsum('nij,mjk->nmik', matrices, matrices)
    result = (mtables.reshape(1, n, n, d, d) - matrices.reshape(n, 1, 1, d, d))
    indices = np.nonzero(np.all(np.all((np.abs(result) < tol), axis=(- 1)), axis=(- 1)))
    indices = np.stack(indices)
    table = np.zeros([n, n], dtype=np.int32)
    table[indices[1], indices[2]] = indices[0]
    return table


def regular_representation(table: np.array) -> np.array:
    """Returns regular representation for group represented by a multiplication table.
    Input:
        table: np.array [n, n] where table[i, j] = k means i * j = k.
    Output:
        Regular representation. array [n, n, n] where reg_rep[i, :, :] = D(i) and D(i)e_j = e_{ij}.
                                Equivalently, D(g) |h> = |gh>
    """
    (n, _) = table.shape
    (g, h) = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    gh = table
    reg_rep = np.zeros((n, n, n))
    reg_rep[g, gh, h] = 1
    return reg_rep
