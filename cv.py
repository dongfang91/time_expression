import numpy


def random_permutation_matrix(n):
    """
    Generate a permutation matrix: an NxN matrix in which each row
    and each column has only one 1, with 0's everywhere else.
    See: https://en.wikipedia.org/wiki/Permutation_matrix
    :param n: size of the square permutation matrix
    :return: NxN permutation matrix
    """
    rows = numpy.random.permutation(n)
    cols = numpy.random.permutation(n)
    m = numpy.zeros((n, n))
    for r, c in zip(rows, cols):
        m[r][c] = 1
    return m

def permute_rows(X, P=None):
    """
    Permute the rows of a 2-d array (matrix) according to
    permutation matrix P.
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with rows permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return numpy.dot(P, X)

def get_newdata(data_x,data_y,P=None):
    if P is None:
        P = random_permutation_matrix(data_x.size)
    x = permute_rows(data_x, P)
    t = permute_rows(data_y, P)
    return x,t