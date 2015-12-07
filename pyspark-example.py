import argparse

from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    # Set up the arguments here.
    parser = argparse.ArgumentParser(description = 'PySpark Dictionary Learning',
        add_help = 'How to use', prog = 'python pyspark-example.py <args>')
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

    # Initialize the SparkContext. This is where you can create RDDs,
    # the Spark abstraction for distributed data sets.
    sc = SparkContext(conf = SparkConf())
