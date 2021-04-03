//#include "scale.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>       /* fabs */
#include <string.h>
#include <stdlib.h>

using namespace std;

int parallel_sk(int *xadj, int *adj, int *txadj, int* tadj,
	double *rv, double *cv, int nov, int num_iter, int siter) {

	int nt = siter;
	double start_time = omp_get_wtime();

	//TO DO: implement the algorithmwe
	//cout << "Number of rows: " << nov << endl;
	//cout << "Number of iters:" << num_iter << endl;
	//cout << "Number of threads " << nt << endl;

#pragma omp parallel for num_threads(nt) schedule(static,16384)
	for (int i = 0; i<nov; i++) { // initialize rv and rc
		rv[i] = 1.0;
		cv[i] = 1.0;
	}
	// perform row sum for the first iteration
#pragma omp parallel for num_threads(nt) schedule(static,16384)
	for (int i = 0; i<nov; i++) {
		int start = xadj[i];
		int end = xadj[i + 1];
		double rsum = 0.0;
		for (int j = 0; j < (end - start); j++) {
			rsum += cv[adj[start + j]];
		}
		rv[i] = 1.0 / rsum;
	}
	for (int x = 0; x < num_iter; x++) {
		// perform column sum
#pragma omp parallel for num_threads(nt) schedule(static,16384)
		for (int j = 0; j < nov; j++) {
			int start = txadj[j];
			int end = txadj[j + 1];
			double csum = 0.0;
			for (int i = 0; i < (end - start); i++) {
				csum += rv[tadj[start + i]];
			}
			cv[j] = 1.0 / csum;
		}
		// compute error
		double error = -1.0;
#pragma omp parallel for num_threads(nt) schedule(static,16384)
		for (int i = 0; i < nov; i++) {
			double cur_value = 0.0;
			int start = xadj[i];
			int end = xadj[i + 1];
			double rsum = 0.0;
			for (int j = 0; j < (end - start); j++) {
				cur_value += rv[i] * cv[adj[start + j]];
				rsum += cv[adj[start + j]]; // update row sum for next iteration
			}
			cur_value = fabs(1.0 - cur_value); // compute the local error
			if (cur_value > error) { // update the global maximum error
#pragma omp critical
				error = cur_value;
			}
			rv[i] = 1.0 / rsum; // update rv for next iteration
		}
		cout << "iter " << x << " - error " << error << endl;
	}
	double end_time = omp_get_wtime();
	std::cout << nt << " Threads  --  " << "Time: " << end_time - start_time << " s." << std::endl;
	return 0;
}


void* read_mtxbin(std::string bin_name, int num_iter, int num_threads) {

	const char* fname = bin_name.c_str();
	FILE* bp;
	bp = fopen(fname, "rb");

	int* nov = new int;
	int* nnz = new int;

	fread(nov, sizeof(int), 1, bp);
	fread(nnz, sizeof(int), 1, bp);

	int* adj = new int[*nnz];
	int* xadj = new int[*nov];
	int* tadj = new int[*nnz];
	int* txadj = new int[*nov];

	fread(adj, sizeof(int), *nnz, bp);
	fread(xadj, sizeof(int), *nov + 1, bp);

	fread(tadj, sizeof(int), *nnz, bp);
	fread(txadj, sizeof(int), *nov + 1, bp);


	int inov = *nov + 1;

	double* rv = new double[inov];
	double* cv = new double[inov];

	parallel_sk(xadj, adj, txadj, tadj, rv, cv, *nov, num_iter, num_threads); //or no_col
	return rv; // just return the adress of rv(arbitrarily selected) to avoid gcc warning
			   // instead of 20, i gave number of threads
}


int main(int argc, char* argv[]) {

	string fname = argv[1];
	int num_iter = atoi(argv[2]);
	int num_threads = atoi(argv[3]);

	read_mtxbin(fname, num_iter, num_threads);
	return 0;
}
