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
    S = S - np.outer(u, v_sparse)
    return S

def main():
    parser = argparse.ArgumentParser(description = 'PySpark Dictionary Learning',
        add_help = 'How to use', prog = 'python DictionaryLearning_spark <args>')
    parser.add_argument("-i", "--input", required = True,
        help="Input File name.(file_s)")
    parser.add_argument("-d", "--dictionary", required = True,
        help="Dictionary File name.(file_D)")
    parser.add_argument("-o", "--output", required = True,
        help="Output File name.(file_Z)")
    parser.add_argument("-n", "--pnonzero", type = float, required = True,
        help="Percentage of Non-zero elements.")
    parser.add_argument("-m", "--mDicatom", type = int, required = True,
        help="Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type = float, required = True,
        help="The value of epsilon.")

    args = vars(parser.parse_args())

    M = int(args['mDicatom'])
    PCT = float(args['pnonzero'])
    epsilon = float(args['epsilon'])
    file_s = str(args['input'])
    file_D = str(args['dictionary'])
    file_Z = str(args['output'])
    S = np.loadtxt(file_s)
    y = np.shape(S)
    T = y[0]
    P = y[1]
    max_iteration = P * 10
    R = float(PCT * P)
    print('Length of samples is:', T, '\n')
    print('Number of samples is:', P, '\n')
    print('Number of dictionaries is:', M, '\n')
    print('Convergence criteria is: ||u_new - u_old||<', epsilon, '\n')
    print('Number of maximum iteration is: ', max_iteration, '\n')
    print("Loading input file...")
    S = S - S.mean(axis = 0)
    S = S / sla.norm(S, axis = 0)
    print('Training .... \n')
    u_old = np.zeros(T, dtype = np.float)
    u_new = np.zeros(T, dtype = np.float)
    v = np.zeros(P, dtype = np.float)
    Z = np.zeros((M, P), dtype = np.float)
    D = np.zeros((M, T), dtype = np.float)
    idxs_n = np.zeros(R, dtype = np.int)
    print('Initalization is complete!')
    epsilon = epsilon * epsilon
    for m in range(M):
        it = 0
        u_old = np.random.random(T)
        u_old = (u_old - u_old.mean())
        u_old = u_old / sla.norm(u_old, axis = 0)
        print('Analyzing component ', (m + 1), '...')
        while True:
            v = np.dot(u_old, S)
            idxs_n = op_selectTopR(v, R)
            u_new = np.dot(S[:, idxs_n], v[idxs_n])
            u_new = u_new / sla.norm(u_new, axis = 0)
            diff = sla.norm(u_old - u_new)
            if (diff < epsilon):
                print('it: ', it)
                break
            it = it + 1
            if (it > max_iteration):
                print('WARNING: MAX ITERATION REACHED! RESULT MAY BE UNSTABLE!\n')
                break
                # Copying the new vector on old one
            u_old = u_new
        S = op_getResidual(S, u_new, v, idxs_n)
        totoalResidual = np.sum(S ** 2)
        Z[m, :] = v
        D[m, :] = u_new
    print('Training complete!')
    print('Writing output (D and z) files...\n')
    print('z =', Z)
    print('totoalResidual =', totoalResidual)
    np.savetxt(file_D, D, fmt = '%.5lf\t')
    np.savetxt(file_Z, Z, fmt = '%.5lf\t')

if __name__ == "__main__":
    main()
