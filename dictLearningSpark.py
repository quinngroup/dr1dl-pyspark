import argparse
import numpy as np
import numpy.linalg as sla

def op_selectTopR(vct_input, R):
    """
    Returns the Rth greatest elements indices
    in vct_input.

    parameters
    ----------
    vct_input : array, shape (T)
        indicating input vector
    R : integer
        indicates Rth greatest elemnts

    Returns
    -------
    idxs_n : array, shape (R)
        which is a vector indicating Rth
        greatest elements indices
    """
    temp = np.argpartition(-vct_input, R)
    idxs_n = temp[:R]
    return idxs_n

def op_getResidual(S, u, v, I, idxs_n, R):
    """
    Returns the new S matrix by calculating :
        S =( S - uv )

    parameters
    ----------
    u : array, shape (T)
        indicating 'u_new' vector
    v : array, shape (P)
        indicating 'v' vector
    I : integer
        indicating number of nonzero items
    idxs_n : array, shape (R)
        which is a vector indicating Rth
        greatest elements indices
    R : integer
        indicates Rth greatest elemnts

    Returns
    -------
    S : array, shape (T, P)
        new S matrix based on above mentioned equation
    """
    for i in range(I):
        for idx_r in range(int(R)):
            j = idxs_n[idx_r]
            S[i, j] = S[i, j] - (u[i] * v[j])
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
    u_old = np.zeros((1, T), dtype = np.float)
    u_new = np.zeros((1, T), dtype = np.float)
    v = np.zeros((1, P), dtype = np.float)
    Z = np.zeros((M, P), dtype = np.float)
    D = np.zeros((M, T), dtype = np.float)
    idxs_n = np.zeros((1, R), dtype = np.int)
    print('Initalization is complete!')
    epsilon = epsilon * epsilon
    for m in range(M):
        it = 0
        u_old = np.random.random(T)
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
            np.copyto(u_old, u_new, casting='same_kind')
        S = op_getResidual(S, u_new, v, T, idxs_n, R)
        totoalResidual = np.sum(S ** 2)
        Z[m, :] = v
        D[m, :] = u_new

    print('Training complete!')
    print('Writing output (D and z) files...\n')
    print('z =', Z)
    print('totoalResidual =', totoalResidual)
    np.savetxt(file_D, D, fmt='%.5lf\t')
    np.savetxt(file_Z, Z, fmt='%.5lf\t')

if __name__ == "__main__":
    main()
