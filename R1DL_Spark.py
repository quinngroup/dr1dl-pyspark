import argparse
import numpy as np
import os.path
import scipy.linalg as sla

from pyspark import SparkContext, SparkConf

###################################
# Utility functions
###################################

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

def input_to_rowmatrix(raw_rdd):
    """
    Utility function for reading the matrix data and converting it to a thunder
    RowMatrix.
    """
    # Parse each line of the input into a numpy array of floats. This requires
    # several steps.
    #  1: Split each string into a list of strings.
    #  2: Convert each string to a float.
    #  3: Convert each list to a numpy array.
    numpy_rdd = raw_rdd \
        .zipWithIndex() \
        .map(lambda x: (x[1], parse_and_normalize(x[0])))
   return numpy_rdd

###################################
# Spark helper functions
###################################

def parse_and_normalize(line):
    """
    Utility function. Parses a line of text into a floating point array, then
    whitens the array.
    """
    x = np.array(map(float, line.strip().split()))

    # x.strip() -- strips off trailing whitespace from the string
    # .split("\t") -- splits the string into a list of strings, splitting on tabs
    # map(float, list) -- converts each element of the list from strings to floats
    # np.array(list) -- converts the list of floats into a numpy array

    x -= x.mean()  # 0-mean.
    x /= sla.norm(x)  # Unit norm.
    return x

def vector_matrix(row):
    """
    Applies u * S by row-wise multiplication, followed by a reduction on
    each column into a single vector.
    """
    row_index, vector = row     # Split up the [key, value] pair.
    u = _U_.value       # Extract the broadcasted vector "v".

    # Generate a list of [key, value] output pairs, one for each nonzero
    # element of vector.
    out = []
    for i in range(vector.shape[0]):
        out.append([i, u[row_index] * vector[i]])
    return out

def matrix_vector(row):
    """
    Applies S * v by row-wise multiplication. No reduction needed, as all the
    summations are performed within this very function.
    """
    k, vector = row

    # Extract the broadcast variables.
    v = _V_.value
    indices = _I_.value

    # Perform the multiplication using the specified indices in both arrays.
    innerprod = np.dot(vector[indices], v)

    # That's it! Return the [row, inner product] tuple.
    return [k, innerprod]

def deflate(row):
    """
    Deflates the data matrix by subtracting off the outer product of the
    broadcasted vectors and returning the modified row.
    """
    k, vector = row

    # It's important to keep order of operations in mind: we are computing
    # (and subtracting from S) the outer product of u * v. As we are operating
    # on a row-distributed matrix, we therefore will only iterate over the
    # elements of v, and use the single element of u that corresponds to the
    # index of the current row of S.

    # Got all that? Good! Explain it to me.
    u, v = _U_.value, _V_.value
    return [k, vector - (u[k] * v)]

if __name__ == "__main__":
    # Set up the arguments here.
    parser = argparse.ArgumentParser(description = 'PySpark Dictionary Learning',
        add_help = 'How to use', prog = 'python R1DL_Spark.py <args>')

    # Inputs.
    parser.add_argument("-i", "--input", required = True,
        help = "Input file containing the matrix S.")
    parser.add_argument("-n", "--pnonzero", type = float, required = True,
        help = "Percentage of Non-zero elements.")
    parser.add_argument("-m", "--mDicatom", type = int, required = True,
        help = "Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type = float, required = True,
        help = "The value of epsilon.")

    # Optional arguments.
    parser.add_argument("--nrows", type = int, default = None,
        help = "Number of rows of data in S. [DEFAULT: None]")
    parser.add_argument("--ncols", type = int, default = None,
        help = "Number of columns of data in S. [DEFAULT: None]")

    # Outputs.
    parser.add_argument("-d", "--dictionary", required = True,
        help = "Output path to dictionary file.(file_D)")
    parser.add_argument("-o", "--output", required = True,
        help = "Output path to Z matrix.(file_Z)")

    args = vars(parser.parse_args())

    # Initialize the SparkContext. This is where you can create RDDs,
    # the Spark abstraction for distributed data sets.
    sc = SparkContext(conf = SparkConf())
    
    # Read the data and convert it into a thunder RowMatrix.
    raw_rdd = sc.textFile(args['input'])
    S = input_to_rowmatrix(raw_rdd, args['nrows'], args['ncols'])

    ##################################################################
    # Here's where the real fun begins.
    #
    # First, we're going to initialize some variables we'll need for the
    # following operations. Next, we'll start the optimization loops. Finally,
    # we'll perform the stepping and deflation operations until convergence.
    #
    # Sound like fun?
    ##################################################################

    # If the number of rows and columns are provided as command-line arguments,
    # this will save from having to compute it from the RDD!
    T = args['ncols'] if args['ncols'] is not None else S.first()[1].shape[0]
    P = args['nrows'] if args['nrows'] is not None else S.count()

    epsilon = args['epsilon']       # convergence stopping criterion
    M = args['mDicatom']            # dimensionality of the learned dictionary
    R = args['pnonzero'] * P        # enforces sparsity
    u_new = np.zeros(T)             # atom updates at each iteration
    v = np.zeros(P)
    Z = np.zeros((M, P))            # output variables
    D = np.zeros((M, T))

    indices = np.zeros(R)           # for top-R sorting
    max_iterations = P * 10

    # Start the loop!
    for m in range(M):

        # Generate a random vector, subtract off its mean, and normalize it.
        u_old = np.random.random(T)
        u_old -= u_old.mean()
        u_old /= sla.norm(u_old)

        num_iterations = 0
        delta = 2 * epsilon

        # Start the inner loop: this learns a single atom.
        while num_iterations < max_iterations and delta > epsilon:
            # P2: Vector-matrix multiplication step. Computes v.
            _U_ = sc.broadcast(u_old)
            v = S \
                .flatMap(vector_matrix) \
                .reduceByKey(lambda x, y: x + y) \
                .collect()
            v = np.take(sorted(v), indices = 1, axis = 1)

            # Use our previous method to select the top R.
            indices = op_selectTopR(v, R)
            temp_v = np.zeros(v.shape)
            temp_v[indices] = v[indices]
            v = temp_v

            # Broadcast the indices and the vector.
            _V_ = sc.broadcast(v[indices])
            _I_ = sc.broadcast(indices)

            # P1: Matrix-vector multiplication step. Computes u.
            u_new = S \
                .map(matrix_vector) \
                .collect()
            u_new = np.take(sorted(u_new), indices = 1, axis = 1)

            # Subtract off the mean and normalize.
            u_new -= u_new.mean()
            u_new /= sla.norm(u_new)

            # Update for the next iteration.
            delta = sla.norm(u_old - u_new)
            u_old = u_new
            num_iterations += 1

        # Add the newly-computed u and v to the output variables.
        D[m] = u_new
        Z[m] = v

        # P4: Deflation step. Update the primary data matrix S.
        _U_ = sc.broadcast(u_new)
        _V_ = sc.broadcast(v)
        print m
        S = S.map(deflate).reduceByKey(lambda x, y: x + y)

    # All done! Write out the matrices as tab-delimited text files, with
    # floating-point values to 6 decimal-point precision.
    np.savetxt(os.path.join(args['dictionary'], "D.txt"),
        D, fmt = "%.6f", delimiter = "\t")
    np.savetxt(os.path.join(args['output'], "Z.txt"),
        Z, fmt = "%.6f", delimiter = "\t")

