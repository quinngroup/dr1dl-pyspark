/*
	rank-1 pursuit for Dictionary Learning, version 2, by Xiang Li (xiangli@uga.edu);
	created on 1/19/2016;
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

void memory_mtxFree(double** mtx_input, unsigned long int I )
{
	for( unsigned int i=0; i<I; i++ )
	{
		free(mtx_input[i]);
	}
	free(mtx_input);
}

unsigned long int* memory_idxsAllocation( unsigned long int I )
{
	unsigned long int* mtx1D_result = (unsigned long int*)malloc(I*sizeof(unsigned long int));
	return mtx1D_result;
}

void op_vctCopy2MTX( const vector<double>& u, double** D, unsigned long int T, unsigned long int k)
{
	for (unsigned long int t=0; t<T; t++)
	{
		D[k][t] = u[t];
	}
}

void op_vctCopy2MTX( const vector<double>& v, double** z, unsigned long int k, const vector<unsigned long int>& Id, unsigned long int R )
{
	for (unsigned long int r=0; r<R; r++)
	{
		unsigned long int p = Id[r];
		z[k][p] = v[p];
	}
}

void op_vctCopy2vct( const vector<double>& vct_input, vector<double>& vct_output, unsigned long int N )
{
	for (unsigned long int n=0; n<N; n++)
	{
		vct_output[n] = vct_input[n];
	}
}

double op_getl2NormMTX(double** S, unsigned long int T, unsigned long int P ) ////revised op_getl2NormMTX;
{
	double l2Norm = 0;
	for (unsigned long int t=0; t<T; t++)
	{
		for (unsigned long int p=0; p<P; p++)
		{
			l2Norm += S[p][t]*S[p][t];
		}
	}
	return l2Norm;
}

void op_getResidual(double** S, const vector<double>& u, const vector<double>& v, unsigned long int T, const vector<unsigned long int>& Id, unsigned long int R ) ////revised op_getResidual;
{
	for (unsigned long int t=0; t<T; t++)
	{
		for (unsigned long int r=0; r<R; r++)
		{
			unsigned long int p=Id[r];
			S[p][t] = S[p][t] - u[t]*v[p];
		}
	}
}

void op_selectTopR( const vector<double> &v, unsigned long int P, vector<unsigned long int>& Id, unsigned long int R )
{
	vector<double> tmp ( P, 0 );
	tmp = v;
	std::nth_element(tmp.begin(), tmp.begin()+R, tmp.end(), std::greater<double>());
	double threshold = tmp[R];
	unsigned long int r = 0;
	for (unsigned long int p=0; p<P; p++)
	{
		if (v[p]>threshold)
		{
			Id[r] = p;
			r++;
		}
	}
}

double op_VCTl2diff( const vector<double>& u1, const vector<double>& u2, unsigned long int T )
{
	double tmpDiff = 0;
	for (unsigned long int t=0;t<T;t++)
	{
		tmpDiff += (u1[t]-u2[t])*(u1[t]-u2[t]);
	}
	return tmpDiff;
}

void op_VCTbyMTX(double** S, const vector<double>& u, vector<double>& v, unsigned long int T, unsigned long int P ) ////revised VCTbyMTX;
{
	for (unsigned long int p=0; p<P; p++)
	{
		double tmp1 = 0;
		for (unsigned long int t=0; t<T; t++)
		{
			tmp1 += u[t]*S[p][t];
		}
		v[p] = tmp1;
	}
}

void op_MTXbyVCT(double** S, const vector<double>& v, vector<double>& u, unsigned long int T, const vector<unsigned long int>& Id, unsigned long int R ) ////revised op_MTXbyVCT;
{
	for (unsigned long int t=0; t<T; t++)
	{
		double tmp1 = 0;
		for (unsigned long int r=0; r<R; r++)
		{
			unsigned long int p=Id[r];
			tmp1 += v[p]*S[p][t];
		}
		u[t] = tmp1;
	}
}

void dataIO_TXT2MTX( char* file_S, double** S, unsigned long int T, unsigned long int P )
{
	FILE *fp;
	fp = fopen( file_S, "r" );
	if( fp == NULL )
	{
		cout<<"could not find file "<<file_S<<endl;
		exit(0);
	}
	for( unsigned int p=0; p<P; p++ )
	{
		for( unsigned int t=0; t<T; t++)
		{
			fscanf(fp, "%lf", &S[p][t]);
		}
	}
	fclose(fp);
};

void dataIO_MTX2TXT( char* file_output, double** mtx_input, unsigned long int I, unsigned long int J )
{
	FILE *fp;
	fp = fopen( file_output, "w" );
	if( fp == NULL )
	{
		cout<<"could not find file: %s\n"<<file_output<<endl;
		exit(0);
	}
	for( unsigned int i=0; i<I; i++ )
	{	
		for( unsigned int j=0; j<J; j++ )
		{
			fprintf(fp, "%.50lf\t", mtx_input[i][j]);
		}
		fprintf(fp, "\n");		
	}
	fclose(fp);
}

void stat_normalize2zeroMeanVCT( vector<double>& u, unsigned long int T )
{
	double double_mean = 0;
	for( unsigned long int t=0; t<T; t++)
	{
		double_mean+=u[t];
	}
	double_mean = double_mean/T;
	for( unsigned long int t=0; t<T; t++)
	{
		u[t]=u[t]-double_mean;
	}
}

void stat_normalize2zeroMeanMTX( double** S, unsigned long int T, unsigned long int P ) ////revised to make row-wise normalization by switching S[t][p] to S[p][t];
{
	for( unsigned long int p=0; p<P; p++)
	{
		double double_mean = 0;
		for( unsigned long int t=0; t<T; t++ )
		{
			double_mean += S[p][t];
		}
		double_mean = double_mean/T;
		for( unsigned int t=0; t<T; t++ )
		{
			S[p][t] = S[p][t] - double_mean;
		}
	}
}

void stat_normalize2l2NormMTX( double** S, unsigned long int T, unsigned long int P ) ////revised to make row-wise normalization by switching S[t][p] to S[p][t];
{
	for( unsigned long int p=0; p<P; p++)
	{
		double double_l2norm = 0;
		for( unsigned long int t=0; t<T; t++ )
		{
			double_l2norm += S[p][t]*S[p][t];
		}
		double_l2norm = pow(double_l2norm, 0.5);
		for( unsigned int t=0; t<T; t++ )
		{
			S[p][t] = S[p][t]/double_l2norm;
		}
	}
};

void stat_normalize2l2NormVCT( vector<double>& u, unsigned long int T )
{
	double double_l2norm = 0;
	for( unsigned long int t=0; t<T; t++ )
	{
		double_l2norm += u[t]*u[t];
	}
	double_l2norm = pow(double_l2norm, 0.5);
	
	for( unsigned int t=0; t<T; t++ )
	{
		u[t] = u[t]/double_l2norm;
	}
};

void stat_randVCT( vector<double>& u, unsigned long int T )
{
	srand (time(NULL));
	for( unsigned long int t=0; t<T; t++ )
	{
		u[t] = 2*(double) (rand() / (RAND_MAX + 1.0))-1;
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
	unsigned long int K =atoi(argv[7]);
	unsigned long int R = (unsigned long int) (atof(argv[8])*P);
	double epsilon = atof(argv[9]);
	
	double **D =  memory_mtxAllocation( K, T );
	double **z =  memory_mtxAllocation( K, P );
	double **S =  memory_mtxAllocation( P, T ); ////transposed input, P rows * T columns (thin matrix);
	unsigned long int max_iteration = P*100;
	
	cout<<"Length of samples is: "<<T<<endl;
	cout<<"Number of samples is: "<<P<<endl;
	cout<<"Number of dictionaries is: "<<K<<endl;
	cout<<"R (number of non-zero elements) is "<<R<<endl;
	cout<<"Convergence criteria is: ||u_new-u_old||<"<<epsilon<<endl;
	cout<<"Number of maximum iteration is: "<<max_iteration<<endl;

	FILE *fp_log;
	fp_log = fopen( file_summary, "w" );
	
	if (argc!=10)
	{
		cout<<"1 file_S 2 T 3 P 4 file_D 5 file_z 6 file_summary 7 K (dictionary size) 8 R (number of non-zero elements in z) 9 epsilon (convergence criteria) \n";
		exit(1);
	}
	
	cout<<"Reading input file..."<<endl;
	dataIO_TXT2MTX( file_S, S, T, P ); ////transposed input, P rows * T columns (thin matrix);
	stat_normalize2zeroMeanMTX( S, T, P );
	stat_normalize2l2NormMTX( S, T, P );
	
	cout<<"Training... "<<endl;
	vector<double> u_old ( T, 0 );
	vector<double> u_new ( T, 0 );
	vector<double> v ( P, 0 );
	vector<unsigned long int> Id ( R, 0 );
	cout<<"Initialization complete!"<<endl;
	
	epsilon = epsilon*epsilon;
	fprintf(fp_log, "op_VCTbyMTX \t op_selectTopR \t op_MTXbyVCT \t op_getResidual \t total_time \t residual \t total_iteration \n" );
	for( unsigned long int k=0; k<K; k++ )
	{
		unsigned long int it=0;
		stat_randVCT( u_old, T );
		stat_normalize2zeroMeanVCT( u_old, T );
		stat_normalize2l2NormVCT( u_old, T );
		cout<<"\tAnalyzing component "<<(k+1)<<"... "<<endl;
		double time1=0;
		double time2=0;
		double time3=0;
		std::clock_t start_iteration;
		start_iteration = std::clock();
		while (true)
		{
			std::clock_t start1 = std::clock();
			op_VCTbyMTX( S, u_old, v, T, P ); ////revised VCTbyMTX;
			//cout<<"        sub1 time cost: "<<( std::clock() - start1 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
			time1 += std::clock() - start1;

			std::clock_t start2 = std::clock();
			op_selectTopR( v, P, Id, R );
			//cout<<"        Sub2 time cost: "<<( std::clock() - start2 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
			time2 += std::clock() - start2;
			
			std::clock_t start3 = std::clock();
			op_MTXbyVCT( S, v, u_new, T, Id, R); ////revised op_MTXbyVCT;
			//cout<<"        Sub3 time cost: "<<( std::clock() - start3 ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
			time3 += std::clock() - start3;
			
			stat_normalize2l2NormVCT( u_new, T );
			
			double diff = op_VCTl2diff( u_old, u_new, T );
			//cout<<"        diff: "<<diff<<endl;
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
		
		std::clock_t start4 = std::clock();
		op_getResidual( S, u_new, v, T, Id, R ); ////revised op_getResidual;
		double totoalResidual = op_getl2NormMTX( S, T, P ); ////revised op_getl2NormMTX;
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
		
		op_vctCopy2MTX( v, z, k, Id, R );
		op_vctCopy2MTX( u_new, D, T, k);
	}
	
	cout<<"Training complete! Total time cost: "<<( std::clock() - start_global ) / (double) CLOCKS_PER_SEC<<" second;"<<endl;
	cout<<"Writing output (D and z) files..."<<endl;
	dataIO_MTX2TXT( file_D, D, K, T );	
	dataIO_MTX2TXT( file_z, z, K, P );
	memory_mtxFree( S, P ); ////S is transposed initialized, so it will be freed;
	memory_mtxFree( z, K );
	memory_mtxFree( D, K );
	cout<<"All done!"<<endl;
	
	fclose(fp_log);
	return 0;
}
