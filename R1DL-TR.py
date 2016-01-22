import argparse
import numpy as np
import numpy.linalg as sla

def op_selectTopR(vct_input, R):
    """
    Returns the Rth greatest elements indices
    in input vector and store them in idxs_n.
    Here, we're using this function instead of
    a complete sorting one, where it's more efficient
    than complete sorting function in real big data application

    parameters
    ----------
    vct_input : array, shape (T)
        indicating the input vector which is a
        vector we aimed to find the Rth greatest
        elements. After finding those elements we
        will store the indices of those specific
        elements in output vector.
    R : integer
        indicates Rth greatest elemnts which we
        are seeking for.

    Returns
    -------
    idxs_n : array, shape (R)
        a vector in which the Rth greatest elements
        indices will be stored and returned as major
        output of the function.
    """
    temp = np.argpartition(-vct_input, R)
    idxs_n = temp[:R]
    return idxs_n

def op_getResidual(S, u, v, idxs_n):
    """
    Returns the new S matrix by calculating :
        S =( S - uv )
    Here the product operation between u and v
    is an outer product operation.

    parameters
    ----------
    S : array, shape (T, P)
        The input matrix ( befor we stored the input
        file in this matrix at the main module of program)
        Here, we need to update this matrix for next iteration.
    u : array, shape (T)
        indicating 'u_new' vector (new vector
        of dictionary elements which will be used
        for updating the S matrix)
    v : array, shape (P)
        indicating 'v' vector ( which would be
        finally our output vector but here we are using
        this vector for updating S matrix by applying
        outer product of specific elements of v
        and u_new )
    idxs_n : array, shape (R)
        which is a vector encompassing Rth
        greatest elements indices.

    Returns
    -------
    S : array, shape (T, P)
        new S matrix based on above mentioned equation
        (updating S matrix for next iteration)
    """
    v_sparse = np.zeros(v.shape[0], dtype = np.float)
    v_sparse[idxs_n] = v[idxs_n]
    S = S - np.outer(v_sparse, u)
    return S

def r1dl(S, nonzero, atoms, epsilon):
    """
    R1DL dictionary method.

    Parameters
    ----------
    S : array, shape (T, P)
        Input data: P instances, T features.
    nonzero : float
        Sparsity of the resulting dictionary (percentage of nonzero elements).
    atoms : integer
        Number of atoms in the resulting dictionary.
    epsilon : float
        Convergence epsilon in determining each dictionary atom.

    Returns
    -------
    D : array, shape (M, T)
        Dictionary atoms.
    Z : array, shape (M, P)
        Loading matrix.
    """
    T, P = S.shape
    max_iteration = P * 10
    R = float(nonzero * P)

    # Normalize the data.
    S -= S.mean(axis = 0)
    S /= sla.norm(S, axis = 0)

    # Generate the atom vectors.
    u_old = np.zeros(P, dtype = np.float)
    u_new = np.zeros(P, dtype = np.float)
    v = np.zeros(T, dtype = np.float)
    Z = np.zeros((atoms, T), dtype = np.float)  # applied
    D = np.zeros((atoms, P), dtype = np.float)  # applied 
    idxs_n = np.zeros(int(R), dtype = np.int)

    epsilon *= epsilon
    for m in range(atoms):
        it = 0
        u_old = np.random.random(P)
        u_old -= u_old.mean()
        u_old /= sla.norm(u_old, axis = 0)
        while True:
            v = np.dot(S, u_old)
            idxs_n = op_selectTopR(v, R)
            u_new = np.dot(v[idxs_n], S[idxs_n, :])
            u_new /= sla.norm(u_new, axis = 0)
            diff = sla.norm(u_old - u_new)
            if (diff < epsilon):
                break
            it += 1
            if (it > max_iteration):
                print('WARNING: Max iteration reached; result may be unstable!\n')
                break
                # Copying the new vector on old one
            u_old = u_new
        S = op_getResidual(S, u_new, v, idxs_n)
        totoalResidual = np.sum(S ** 2)
        Z[m, :] = v
        D[m, :] = u_new

    # All done!
    return [D, Z]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Python Dictionary Learning',
        add_help = 'How to use', prog = 'python R1DL.py <args>')

    # Input arguments.
    parser.add_argument("-i", "--input", required = True,
        help = "Input filename containing matrix S.")
    parser.add_argument("-r", "--pnonzero", type = float, required = True,
        help = "Percentage of non-zero elements.")
    parser.add_argument("-m", "--mDicatom", type = int, required = True,
        help = "Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type = float, required = True,
        help = "The value of epsilon.")

    # Output arguments.
    parser.add_argument("-d", "--dictionary", required = True,
        help = "Dictionary (D) output file.")
    parser.add_argument("-z", "--zmatrix", required = True,
        help = "Loading matrix (Z) output file.")

    args = vars(parser.parse_args())

    # Parse out the command-line arguments.
    M = args['mDicatom']
    R = args['pnonzero']
    epsilon = args['epsilon']
    file_s = args['input']
    file_D = args['dictionary']
    file_Z = args['zmatrix']

    # Read the inputs and generate variables to pass to R1DL.
    S1 = np.loadtxt(file_s)
    S = S1.transpose()
    D, Z = r1dl(S, R, M, epsilon)
    #T, P = S.shape
    #counter = 0
    #for i in range( M ):
    #    for j in range( P ):
    #        if Z[i, j] == 0.00000:
    #            counter += 1
    #print("number of zeros in Z file :", counter )
    # Write the output to files.
    np.savetxt(file_D, D, fmt = '%.5lf\t')
    np.savetxt(file_Z, Z, fmt = '%.5lf\t')
