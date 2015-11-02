/*
	rank-1 pursuit for Dictionary Learning, by Xiang Li (xiangli@uga.edu);
	created on 7/15/2015;
*/
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <functional>
#include <time.h>
#include <ctime>
#include <iomanip>
#include <string>
#include <limits>
#include <omp.h>
using namespace std;

double **memory_mtxAllocation( unsigned long int I, unsigned long int J )
{
	double **mtx2D_result = (double**)malloc(I*sizeof(double*));	
	for( unsigned long int i=0; i<I; i++ )
	{
		mtx2D_result[i] = (double*)malloc(J*sizeof(double));	
	}
	return mtx2D_result;	
}

void memory_mtxFree(double** mtx_input, unsigned long int N )
{
	for( unsigned int n=0; n<N; n++ )
	{
		free(mtx_input[n]);
	}
	free(mtx_input);
}

unsigned long int* memory_idxsAllocation( unsigned long int N )
{
	unsigned long int* mtx1D_result = (unsigned long int*)malloc(N*sizeof(unsigned long int));	
	return mtx1D_result;	
}

double *memory_vctAllocation( unsigned long int N )
{
	double *mtx1D_result = (double*)malloc(N*sizeof(double));	
	return mtx1D_result;	
}

void op_vctCopy2MTX( const vector<double>& vct_input, double** mtx_input, unsigned long int N, unsigned long int idx_copy)
{
	for (unsigned long int n=0; n<N; n++)
	{
		mtx_input[idx_copy][n] = vct_input[n];
	}
}

void op_vctCopy2MTX( const vector<double>& vct_input, double** mtx_input, unsigned long int N, unsigned long int idx_copy, const vector<unsigned long int>& idxs_n, unsigned long int R )
{
	for (unsigned long int r=0; r<R; r++)
	{
		unsigned long int n = idxs_n[r];
		mtx_input[idx_copy][n] = vct_input[n];
	}
}

void op_vctCopy2vct( const vector<double>& vct_input, vector<double>& vct_output, unsigned long int N )
{
	for (unsigned long int n=0; n<N; n++)
	{
		vct_output[n] = vct_input[n];
	}
}

double op_getl2NormMTX(double** mtx_input, unsigned long int I, unsigned long int J )
{
	double double_result = 0;
	for (unsigned long int i=0; i<I; i++)
	{
		for (unsigned long int j=0; j<J; j++)
		{
			double_result += mtx_input[i][j]*mtx_input[i][j];
		}
	}
	return double_result;
}

void op_getResidual(double** S, const vector<double>& u, const vector<double>& v, unsigned long int I, const vector<unsigned long int>& idxs_n, unsigned long int R )
{
	for (unsigned long int i=0; i<I; i++)
	{
		for (unsigned long int idx_r=0; idx_r<R; idx_r++)
		{
			unsigned long int j = idxs_n[idx_r];
			S[i][j] = S[i][j] - u[i]*v[j];
		}
	}
}

void op_getResidual(double** S, const vector<double>& u, const vector<double>& v, unsigned long int I, unsigned long int J )
{
	for (unsigned long int i=0; i<I; i++)
	{
		for (unsigned long int j=0; j<J; j++)
		{	
			S[i][j] = S[i][j] - u[i]*v[j];
		}
	}
}


void op_selectTopR( const vector<double> &vct_input, unsigned long int N, vector<unsigned long int>& idxs_n, unsigned long int R )
{
	vector<double> tmp ( N, 0 );
	tmp = vct_input;
	std::nth_element(tmp.begin(), tmp.begin()+R, tmp.end(), std::greater<double>());
	double threshold = tmp[R];
	unsigned long int r = 0;
	for (unsigned long int n=0; n<N; n++)
	{
		if (vct_input[n]>threshold)
		{
			idxs_n[r] = n;
			r++;
		}
	}
}


/* Slower method;
void op_selectTopR( vector<double>& vct_input, unsigned long int N, vector<unsigned long int>& idxs_n, unsigned long int R )
{
	vector<double> tmp_values ( R, 0 );
	for( unsigned long int idx_r=0; idx_r<R; idx_r++ )
	{
		double tmp_value = -1*std::numeric_limits<double>::infinity();
		long int tmp_idx=-1;
		for (unsigned long int n=0; n<N; n++ )
		{
			if (vct_input[n]>tmp_value)
			{
				tmp_value = vct_input[n];
				tmp_idx = n;
			}
		}
		tmp_values[idx_r] = tmp_value;
		idxs_n[idx_r] = tmp_idx;
		vct_input[tmp_idx] = -1*std::numeric_limits<double>::infinity();
	}
	for( unsigned long int idx_r=0; idx_r<R; idx_r++ )
	{
		vct_input[idxs_n[idx_r]] = tmp_values[idx_r];
	}
}
*/


/* void op_selectTopR(vector<double> vct_input, unsigned long int N, unsigned long int* idxs_n, unsigned long int R )
{
	vector<double> tmp_values = memory_vctAllocation(R);
	for( unsigned long int idx_r=0; idx_r<R; idx_r++ )
	{
		//double tmp_value = -1*std::numeric_limits<double>::infinity();
		double tmp_value = 0; //only select positive coefficients;
		long int tmp_idx=-1;
		for (unsigned long int n=0; n<N; n++ )
		{
			if (vct_input[n]>tmp_value)
			{
				tmp_value = vct_input[n];
				tmp_idx = n;
			}
		}
		if (tmp_idx==-1) //no more positive coefficients;
		{

		}
		tmp_values[idx_r] = tmp_value;
		idxs_n[idx_r] = tmp_idx;
		vct_input[tmp_idx] = -1*std::numeric_limits<double>::infinity();
	}
	for( unsigned long int idx_r=0; idx_r<R; idx_r++ )
	{
		vct_input[idxs_n[idx_r]] = tmp_values[idx_r];
	}
}
*/

void op_selectTopR( vector<unsigned long int>& idxs_n, unsigned long int R )
{
	for ( unsigned long int r=0; r<R; r++)
	{
		idxs_n[r] = r;
	}
}

double op_VCTl2diff( const vector<double>& vct_input1, const vector<double>& vct_input2, unsigned long int N )
{
	double tmp_diff = 0;
	for (unsigned long int n=0; n<N; n++)
	{
		tmp_diff += pow((vct_input1[n]-vct_input2[n]), 2);
	}
	return tmp_diff;
}

void op_VCTbyMTX(double** mtx_input, const vector<double>& vct_input, vector<double>& vct_result, unsigned long int I, unsigned long int J )
{
	for (unsigned long int j=0; j<J; j++)
	{
		double tmp1 = 0;
		for (unsigned long int i=0; i<I; i++)
		{
			tmp1 += vct_input[i]*mtx_input[i][j];
		}
		vct_result[j] = tmp1;
	}
}

void op_MTXbyVCT(double** mtx_input, const vector<double>& vct_input, vector<double>& vct_result, unsigned long int count_row, unsigned long int count_col, const vector<unsigned long int>& idxs_col, unsigned long int R )
{
	for (unsigned long int idx_row=0; idx_row<count_row; idx_row++)
	{
		double tmp1 = 0;
		for (unsigned long int idx_r=0; idx_r<R; idx_r++)
		{
			unsigned long int idx_col = idxs_col[idx_r];
			tmp1 += vct_input[idx_col]*mtx_input[idx_row][idx_col];
		}
		vct_result[idx_row] = tmp1;
	}
}

void op_MTXbyVCT(double** mtx_input, const vector<double>& vct_input, vector<double>& vct_result, unsigned long int count_row, unsigned long int count_col )
{
	for (unsigned long int idx_row=0; idx_row<count_row; idx_row++)
	{
		double tmp1 = 0;
		for (unsigned long int idx_col=0; idx_col<count_col; idx_col++)
		{
			tmp1 += vct_input[idx_col]*mtx_input[idx_row][idx_col];
		}
		vct_result[idx_row] = tmp1;
	}
}

void dataIO_TXT2MTX( char* file_input, double** mtx_input, unsigned long int count_row, unsigned long int count_col )
{
	FILE *fp;
	fp = fopen( file_input, "r" );
	if( fp == NULL )
	{
		cout<<"could not find file "<<file_input<<endl;
		exit(0);
	}
	for( unsigned int idx_row=0; idx_row<count_row; idx_row++ )
	{
		for( unsigned int idx_col=0; idx_col<count_col; idx_col++)
		{
			fscanf(fp, "%lf", &mtx_input[idx_row][idx_col]);		
		}
	}
	fclose(fp);
};

void dataIO_TXT2VCT( char* file_input, vector<double>& mtx_input, unsigned long int count_row )
{
	FILE *fp;
	fp = fopen( file_input, "r" );
	if( fp == NULL )
	{
		cout<<"could not find file "<<file_input<<endl;
		exit(0);
	}
	for( unsigned int idx_row=0; idx_row<count_row; idx_row++ )
	{
		fscanf(fp, "%lf", &mtx_input[idx_row]);		
	}
	fclose(fp);
};

void dataIO_VCT2TXT( char* file_output, const vector<double>& vct_input, unsigned long int count_row )
{
	FILE *fp;
	fp = fopen( file_output, "w" );
	if( fp == NULL )
	{
		cout<<"could not find file: %s\n"<<file_output<<endl;
		exit(0);
	}
	for( unsigned int idx_row=0; idx_row<count_row; idx_row++ )
	{	
		fprintf(fp, "%.50lf\n", vct_input[idx_row]);
	}
	fclose(fp);
}

void dataIO_MTX2TXT( char* file_output, double** mtx_input, unsigned long int count_row, unsigned long int count_col )
{
	FILE *fp;
	fp = fopen( file_output, "w" );
	if( fp == NULL )
	{
		cout<<"could not find file: %s\n"<<file_output<<endl;
		exit(0);
	}
	for( unsigned int idx_row=0; idx_row<count_row; idx_row++ )
	{	
		for( unsigned int idx_col=0; idx_col<count_col; idx_col++ )
		{
			fprintf(fp, "%.50lf\t", mtx_input[idx_row][idx_col]);
		}
		fprintf(fp, "\n");		
	}
	fclose(fp);
}

void stat_normalize2zeroMeanVCT( vector<double>& vct_input, unsigned long int N )
{
	double double_mean = 0;
	for( unsigned long int n=0; n<N; n++)
	{
		double_mean += vct_input[n];
	}
	double_mean = double_mean/N;
	for( unsigned long int n=0; n<N; n++)
	{
		vct_input[n] = vct_input[n] - double_mean;
	}
}

void stat_normalize2zeroMeanMTX( double** mtx_input, unsigned long int T, unsigned long int P )
{
	for( unsigned long int p=0; p<P; p++)
	{
		double double_mean = 0;
		for( unsigned long int t=0; t<T; t++ )
		{
			double_mean += mtx_input[t][p];
		}
		double_mean = double_mean/T;
		for( unsigned int t=0; t<T; t++ )
		{
			mtx_input[t][p] = mtx_input[t][p] - double_mean;
		}
	}
}

void stat_normalize2l2NormMTX( double** mtx_input, unsigned long int T, unsigned long int P )
{
	for( unsigned long int p=0; p<P; p++)
	{
		double double_l2norm = 0;
		for( unsigned long int t=0; t<T; t++ )
		{
			double_l2norm += mtx_input[t][p]*mtx_input[t][p];
		}
		double_l2norm = pow(double_l2norm, 0.5);
		for( unsigned int t=0; t<T; t++ )
		{
			mtx_input[t][p] = mtx_input[t][p]/double_l2norm;
		}
	}
};

void stat_normalize2l2NormVCT( vector<double>& vct_input, unsigned long int T )
{
	double double_l2norm = 0;
	for( unsigned long int t=0; t<T; t++ )
	{
		double_l2norm += vct_input[t]*vct_input[t];
	}
	double_l2norm = pow(double_l2norm, 0.5);
	for( unsigned int t=0; t<T; t++ )
	{
		vct_input[t] = vct_input[t]/double_l2norm;
	}
};

void stat_randMTX( double** mtx_input, unsigned long int count_row, unsigned long int count_col )
{
	srand(clock());
	unsigned int myseed = (unsigned int) RAND_MAX * rand();	
	for( unsigned long int idx_row=0; idx_row<count_row; idx_row++ )
	{
		for( unsigned long int idx_col=0; idx_col<count_col; idx_col++ )
		{
			mtx_input[idx_row][idx_col] = 2*(double) (rand() / (RAND_MAX + 1.0))-1;
		}
	}
	return;
};

void stat_randVCT( vector<double>& vct_input, unsigned long int count_row )
{
	srand(clock());
	unsigned int myseed = (unsigned int) RAND_MAX * rand();	
	for( unsigned long int idx_row=0; idx_row<count_row; idx_row++ )
	{
		vct_input[idx_row] = 2*(double) (rand() / (RAND_MAX + 1.0))-1;
	}
	return;
};

int main(int argc, char* argv[])
{
	std::clock_t start_global;
	start_global = std::clock();
	char* file_S =argv[1]; 
	unsigned long int T = atoi(argv[2]);
	unsigned long int P = atoi(argv[3]);
	char* file_D =argv[4];
	char* file_z = argv[5];
	char* file_summary = argv[6]; // including error and number of iterations;
	unsigned long int M =atoi(argv[7]);
	unsigned long int R = (unsigned long int) (atof(argv[8])*P);
	double epsilon = atof(argv[9]);
	
	double **D =  memory_mtxAllocation( M, T );
	double **z =  memory_mtxAllocation( M, P );
	double **S =  memory_mtxAllocation( T, P );;
	unsigned long int max_iteration = P*10;
	
	cout<<"Length of samples is: "<<T<<endl;
	cout<<"Number of samples is: "<<P<<endl;
	cout<<"Number of dictionaries is: "<<M<<endl;
	cout<<"R (number of non-zero elements) is "<<R<<endl;
	cout<<"Convergence criteria is: ||u_new-u_old||<"<<epsilon<<endl;
	cout<<"Number of maximum iteration is: "<<max_iteration<<endl;

	FILE *fp_log;
	fp_log = fopen( file_summary, "w" );
	
	if (argc!=10)
	{
		cout<<"1 file_S 2 T 3 P 4 file_D 5 file_z 6 file_summary 7 M (dictionary size) 8 R (number of non-zero elements in z) 9 epsilon (convergence criteria) \n";
		exit(1);
	}
	
	cout<<"Reading input file..."<<endl;
	dataIO_TXT2MTX( file_S, S, T, P );
	stat_normalize2zeroMeanMTX( S, T, P );
	stat_normalize2l2NormMTX( S, T, P );
	
	cout<<"Training... "<<endl;
	vector<double> u_old ( T, 0 );
	vector<double> u_new ( T, 0 );
	vector<double> v ( P, 0 );
	vector<unsigned long int> idxs_n ( R, 0 );
	cout<<"Initialization complete!"<<endl;
	
	epsilon = epsilon*epsilon;
	fprintf(fp_log, "op_VCTbyMTX \t op_selectTopR \t op_MTXbyVCT \t op_getResidual \t total_time \t residual \t total_iteration \n" );
	for( unsigned long int m=0; m<M; m++ )
	{
		unsigned long int it=0;
		stat_randVCT( u_old, T );
		stat_normalize2zeroMeanVCT( u_old, T );
		stat_normalize2l2NormVCT( u_old, T );
		cout<<"\tAnalyzing component "<<(m+1)<<"... "<<endl;
		double time1=0;
		double time2=0;
		double time3=0;
		std::clock_t start_iteration;
		start_iteration = std::clock();
		while (true)
		{
			std::clock_t start1 = std::clock();
			op_VCTbyMTX( S, u_old, v, T, P );	
			//cout<<"\t\t VCTbyMTX time cost: "<<( std::clock() - start_sub1 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
			time1 += std::clock() - start1;

			// Use all voxels, for debugging purpose only;
			//op_selectTopR( idxs_n, R );

			std::clock_t start2 = std::clock();
			op_selectTopR( v, P, idxs_n, R );
			//cout<<"        Sub2 time cost: "<<( std::clock() - start_sub2 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
			time2 += std::clock() - start2;
			

			// Use all voxels, for debugging purpose only;
			//op_MTXbyVCT( S, v, u_new, T, P );

			std::clock_t start3 = std::clock();
			op_MTXbyVCT( S, v, u_new, T, P , idxs_n, R);	
			//cout<<"        Sub3 time cost: "<<( std::clock() - start_sub3 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
			time3 += std::clock() - start3;

			stat_normalize2l2NormVCT( u_new, T );
			double diff = op_VCTl2diff( u_old, u_new, T );
			if ( diff < epsilon )
			{
				break;
			}
			it++;
			if (it > max_iteration)
			{
				cout<<"WARNING: MAX ITERATION REACHED! RESULT MAY BE UNSTABLE!"<<endl;
				break;
			}
			op_vctCopy2vct( u_new, u_old, T );
		}
		double timeIteration = ( std::clock() - start_iteration ) / (double) CLOCKS_PER_SEC;
		cout<<"\tTotal iterations: "<<it<<", total time cost: "<<timeIteration<<" second;"<<endl;
		
		// Use all voxels, for debugging purpose only;
		//op_getResidual( S, u_new, v, T, P );
		
		std::clock_t start4 = std::clock();
		op_getResidual( S, u_new, v, T, idxs_n, R );
		double totoalResidual = op_getl2NormMTX( S, T, P );
		double time4 = std::clock() - start4;
		//cout<<"    Sub6 time cost: "<<( std::clock() - start_sub6 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
		fprintf(fp_log, "%f\t", (time1/ (double) CLOCKS_PER_SEC) );
		fprintf(fp_log, "%f\t", (time2/ (double) CLOCKS_PER_SEC) );
		fprintf(fp_log, "%f\t", (time3/ (double) CLOCKS_PER_SEC) );
		fprintf(fp_log, "%f\t", (time4/ (double) CLOCKS_PER_SEC) );
		fprintf(fp_log, "%f\t", timeIteration );
		fprintf(fp_log, "%f\t", totoalResidual );
		fprintf(fp_log, "%d\n", it );
		
		// For benchmarking purpose only;
		//cout<<"    Object function value: "<<op_getl2NormMTX(S, T, P)<<endl;
		
		// Use all voxels, for debugging purpose only;
		//op_vctCopy2MTX( v, z, P, m );

		op_vctCopy2MTX( v, z, P, m, idxs_n, R );
		op_vctCopy2MTX( u_new, D, T, m);
	}
	
	cout<<"Training complete! Total time cost: "<<( std::clock() - start_global ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
	cout<<"Writing output (D and z) files..."<<endl;
	dataIO_MTX2TXT( file_D, D, M, T );	
	dataIO_MTX2TXT( file_z, z, M, P );
	memory_mtxFree( S, T );
	memory_mtxFree( z, M );
	memory_mtxFree( D, M );
	cout<<"All done!"<<endl;
	
	fclose(fp_log);
	return 0;
}
