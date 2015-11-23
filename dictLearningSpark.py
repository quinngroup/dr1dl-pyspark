import argparse
import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object
import math
import random
import matplotlib.pyplot as plt

RAND_MAX = 2147483647

def op_selectTopR( vct_input, idxs_n, R):
	temp = np.argpartition(-vct_input, R)
	idxs_n = temp[:R]
	return (idxs_n)

def op_VCTl2diff( vct_input1, vct_input2, N):
	tmp_diff = 0
	for n in range(N):
		tmp_diff = np.power((vct_input1[n]-vct_input2[n]), 2) + tmp_diff
	return (tmp_diff)

def op_getResidual( S, u, v, I, idxs_n, R):
	
	for i in range (I):
		for idx_r in range(R):
			j = idxs_n[idx_r]
			S[[i],[j]] = S[[i],[j]] - u[i]*v[j]

def op_getl2NormMTX(mtx_input, I, J):
	double_result = 0
	for i in range(I):
		for j in range(J):
			double_result = mtx_input[[i],[j]]*mtx_input[[i],[j]] + double_result 
	return (double_result)

def op_vctCopy2MTX( vct_input, mtx_input, N, idx_copy):
	
	for n in range (N):
		mtx_input[[idx_copy],[n]] = vct_input[n]

def op_vctCopy2MTX2( vct_input, mtx_input, N, idx_copy, idxs_n, R):

	for r in range (R):
		n = idxs_n[r]
		mtx_input[idx_copy][n] = vct_input[n]

def stat_normalize2l2NormVCT(vct_input, T):
	double_l2norm = 0
	for t in range(T):
		double_l2norm = vct_input[t]*vct_input[t] + double_l2norm
	double_l2norm = np.sqrt(double_l2norm)

	for t in range (T):
		vct_input[t] = vct_input[t]/double_l2norm

# chek again !
def stat_normalize2zeroMeanMTX( mtx_input, T, P ):
	for p in range(P):
		double_mean = 0
		for t in range (T):
			double_mean = mtx_input[[t],[p]] + double_mean
		double_mean = double_mean/T
		for t in range (T):
			mtx_input[[t],[p]] = mtx_input[[t],[p]] - double_mean

def stat_normalize2l2NormMTX( mtx_input, T, P ):
	for p in range(P):
		double_l2norm = 0
		for t in range (T):
			double_l2norm = (mtx_input[[t],[p]]*mtx_input[[t],[p]]) + double_l2norm
		double_l2norm = math.sqrt(double_l2norm)
		for t in range (T):
			mtx_input[[t],[p]] = mtx_input[[t],[p]]/double_l2norm

def stat_randVCT(vct_input, count_row ):
	myseed = RAND_MAX * random.random()	
	for idx_row in range (count_row):
		vct_input[idx_row] = 2*(random.random() / (RAND_MAX + 1.0))-1	

#def op_VCTbyMTX(mtx_input, vct_input, vct_result, I, J):
#	for j in range(J):
#		tmp1 = 0
#		for i in range(I):
#			tmp1 = vct_input[i]*mtx_input[[i],[j]] + tmp1
#		vct_result[j] = tmp1


def main():
#parser = argparse.ArgumentParser(description='close bug')
	parser = argparse.ArgumentParser(description = 'PySpark Dictionary Learning',
        add_help = 'How to use', prog = 'python DictionaryLearning_spark <args>')
	parser.add_argument("-i", "--input", required = True,
        help = "Input File name.(file_s)")
	parser.add_argument("-d", "--dictionary", required = True,
        help = "Dictionary File name.(file_D)")
	parser.add_argument("-o", "--output", required = True,
        help = "Output File name.(file_Z)")
	parser.add_argument("-s", "--summary", required = True,
        help = "Summary File name.(file_summary)")
	parser.add_argument("-l", "--length", type = int, required = True,
        help = "Length of the samples.")
	parser.add_argument("-P", "--pnumber", type = int, required = True,
        help = "Number of the samples.")
	parser.add_argument("-n", "--pnonzero", type = int, required = True,
        help = "Percentage of Non-zero elements.")
	parser.add_argument("-m", "--mDicatom", type = int, required = True,
        help = "Number of the dictionary atoms.")
	parser.add_argument("-e", "--epsilon", type = float, required = True,
        help = "The value of epsilon.")


	args = vars(parser.parse_args())

	#print(args['input'])
	#print(args['dictionary'])
	#print(args['output'])
	#print(args['summary'])
# Intializing the variables ;
	P = int(args['pnumber'])
	T = int(args['length'])
	M = int(args['mDicatom'])
	PCT = float(args['pnonzero'])
	R = float(PCT * P)
	epsilon = float(args['epsilon'])
	file_summary = args['summary']
    file_s = args['input']
    file_D = args['dictionary']
    file_Z = args['output']

# setting the max number of iterations
	max_iteration = P*10
# Opening the file in Write mode & converting the TXT file to a matrix
	print("The Input file is loading...")
    #S = np.genfromtxt(file_s,delimiter='    ') 
    S = np.loadtxt(file_s)
# Normalizing the Data 
	stat_normalize2zeroMeanMTX( S, T, P )
	stat_normalize2l2NormMTX( S, T, P )
	print('Training .... \n')
# Initializing 4 vectors with zero    
    u_old = np.zeros((1,T), dtype=np.float)
    u_new = np.zeros((1,T), dtype=np.float)
    v = np.zeros((1,P), dtype=np.float)
    idxs_n = np.zeros((1,R), dtype=np.float)
    print ('Initalization is complete!')
    epsilon = epsilon * epsilon

	for m in range(M):
		it=0
		stat_randVCT( u_old, T )
		stat_normalize2zeroMeanVCT( u_old, T )
		stat_normalize2l2NormVCT( u_old, T )
		print('Analyzing component ',(m+1),'...')
		
		while True :
		
			# this instruction is equal with : op_VCTbyMTX( S, u_old, v, T, P );
			v = np.dot(S,u_old)
			idxs_n= op_selectTopR(vct_input,idxs_n,R)
			op_selectTopR( v, P, idxs_n, R )
			u_new = np.dot(S,v)	
			stat_normalize2l2NormVCT( u_new, T)
			diff = op_VCTl2diff( u_old, u_new, T)
			if ( diff < epsilon ):
				break
			it=it+1
			if (it > max_iteration):
				print('WARNING: MAX ITERATION REACHED! RESULT MAY BE UNSTABLE! \n')
				break
			# Copying the new vector on old one
			#u_old[:] = u_new
			np.copyto(u_old,u_new,casting='same_kind')
		op_getResidual( S, u_new, v, T, idxs_n, R )	
		totoalResidual = op_getl2NormMTX( S, T, P )
		op_vctCopy2MTX2( v, Z, P, m, idxs_n, R )
		op_vctCopy2MTX( u_new, D, T, m)

	print('Training complete!')
	print('Writing output (D and z) files...\n')
	np.savetxt(file_D, D, fmt='%.50lf\t')
	np.savetxt(file_Z, Z, fmt='%.50lf\t')

if __name__ == "__main__":
	main()
