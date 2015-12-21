import argparse
import numpy as np

from pyspark import SparkContext, SparkConf
from thunder import ThunderContext, RowMatrix

def input_to_rowmatrix(raw_rdd, nrows, ncols):
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
        .map(lambda x: np.array(map(float, x.strip().split("\t")))) \
        .zipWithIndex() \
        .map(lambda x: ((x[1],), x[0]))  # Reverse the elements so the index is first.

    # x.strip() -- strips off trailing whitespace from the string
    # .split("\t") -- splits the string into a list of strings, splitting on tabs
    # map(float, list) -- converts each element of the list from strings to floats
    # np.array(list) -- converts the list of floats into a numpy array

    # Now, convert the RDD of (index, ndarray) tuples to a thunder RowMatrix.
    S = RowMatrix(numpy_rdd,
        dtype = np.float,
        nrows = nrows,
        ncols = ncols)
    return S

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
    tsc = ThunderContext(sc)

    # Read the data and convert it into a thunder RowMatrix.
    raw_rdd = sc.textFile(args['input'])
    S = input_to_rowmatrix(raw_rdd, args['nrows'], args['ncols'])

    # Column-wise whitening of S.
    S.zscore(axis = 1)
