#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>       /* fabsf */
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define NO_THREADS 256
#define SHMEM_SIZE 1024 * 8

#define DEBUG 0

using namespace std;
//Error check-----
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
//Error check-----
//This is a very good idea to wrap your calls with that function.. Otherwise you will not be able to see what is the error.
//Moreover, you may also want to look at how to use cuda-memcheck and cuda-gdb for debugging.

__device__ void warpReduce(volatile int* shmem_ptr, int t){
  shmem_ptr[t] += shmem_ptr[t + 32];
  shmem_ptr[t] += shmem_ptr[t + 16];
  shmem_ptr[t] += shmem_ptr[t + 8];
  shmem_ptr[t] += shmem_ptr[t + 4];
  shmem_ptr[t] += shmem_ptr[t + 2];
  shmem_ptr[t] += shmem_ptr[t + 1];

}

__global__ void row_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows){

  //TO DO: GPU SCALE
  int start,end;
  double rsum;
  int i = blockIdx.y * blockDim.y + threadIdx.y; //n
  int j = blockIdx.x * blockDim.x + threadIdx.x; //i

  start = xadj[i];
  end   = xadj[i+1];
  if (i>=rows || j>=(end - start))
        return;
  rsum = 0.0;
  rsum += cv[adj[start + j]];
	rv[i] = 1.0/rsum;
}

__global__ void less_row_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows){

  //TO DO: GPU SCALE
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  if(i < rows){
    int start = xadj[i];
    int end   = xadj[i+1];
    double rsum = 0.0;
    for(int j = 0; j < (end - start); j++){
      rsum += cv[adj[start + j]];
    }
    /*if(rsum == 0.0){
      printf("%s\n", rsum);
    }*/
    rv[i] = 1.0/rsum;
    //printf("%s\n", rsum);
  }

}

__global__ void col_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows){

  //TO DO: GPU SCALE
  int start,end;
  double csum;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  start = txadj[j];
  end   = txadj[j+1];
  if (j>=rows || i>=(end - start))
        return;
  csum = 0.0;
  csum += rv[tadj[start + i]];
	cv[j] = 1.0/csum;

}

__global__ void less_col_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows){

  //TO DO: GPU SCALE
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("%d\n", j);
  if (j < rows){
    int start = txadj[j];
    int end = txadj[j+1];
    double csum = 0.0;
    for(int i = 0; i < (end - start); i++){
      csum += rv[tadj[start + i]];
    }
    cv[j] = 1.0/csum;
  }

}

__global__ void code_optimized_error_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double * temp){

  //TO DO: GPU SCALE
  __shared__ double partial_max[NO_THREADS];
  //double error = -1.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x; //i
  if(i < rows){
    double cur_value = 0.0;
    double rsum = 0.0;
    int start = xadj[i];
    int end = xadj[i+1];
    for(int j = 0; j < (end - start); j++){
      cur_value +=  rv[i] * cv[adj[start + j]];
      rsum += cv[adj[start + j]];
    }
    cur_value = fabs(1.0 - cur_value); // compute the local error
    partial_max[threadIdx.x] = cur_value; // store the local error in shared memory
    rv[i] = 1.0/rsum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){

      if(threadIdx.x < s){
        double a = partial_max[threadIdx.x];
        double b = partial_max[threadIdx.x + s];
        partial_max[threadIdx.x] = (a >= b) ? a : b;
      }
      __syncthreads();
    }
    if(threadIdx.x == 0){
      temp[blockIdx.x] = partial_max[0]; // store the maximum values of particular thread blocks
    }

  }
}

__global__ void error_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double &error){

  //TO DO: GPU SCALE
  int start,end;
  //double rsum;
  double cur_value = 0.0;
  int i = blockIdx.y * blockDim.y + threadIdx.y; //n
  int j = blockIdx.x * blockDim.x + threadIdx.x; //i

  start = xadj[i];
  end   = xadj[i+1];
  if (i>=rows || j>=(end - start))
        return;
  //rsum = 0.0;
  cur_value +=  rv[i] * cv[adj[start + j]];
  //rsum += cv[adj[start + j]];
	//rv[i] = 1.0/rsum;
}

__global__ void less_error_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double * temp){

  __shared__ double partial_max[NO_THREADS];
  //double error = -1.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x; //i
  if(i < rows){
    double cur_value = 0.0;
    double rsum = 0.0;
    int start = xadj[i];
    int end = xadj[i+1];
    for(int j = 0; j < (end - start); j++){
      cur_value +=  rv[i] * cv[adj[start + j]];
      rsum += cv[adj[start + j]];
    }
    cur_value = fabs(1.0 - cur_value); // compute the local error
    partial_max[threadIdx.x] = cur_value; // store the local error in shared memory
    rv[i] = 1.0/rsum;
    __syncthreads();

    for(int s = 1; s< blockDim.x; s*=2){
      if(threadIdx.x % (2*s) == 0){
        double a = partial_max[threadIdx.x];
        double b = partial_max[threadIdx.x + s];
        partial_max[threadIdx.x] = (a >= b) ? a : b;
      }
      __syncthreads();
    }
    if(threadIdx.x == 0){
      temp[blockIdx.x] = partial_max[0]; // store the maximum values of particular thread blocks
    }

  }
}

__global__ void non_optimized_error_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double * temp){

  __shared__ double partial_max[NO_THREADS];
  //double error = -1.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x; //i
  if(i < rows){
    double cur_value = 0.0;
    int start = xadj[i];
    int end = xadj[i+1];
    for(int j = 0; j < (end - start); j++){
      cur_value +=  rv[i] * cv[adj[start + j]];
    }
    cur_value = fabs(1.0 - cur_value); // compute the local error
    partial_max[threadIdx.x] = cur_value; // store the local error in shared memory
    __syncthreads();

    for(int s = 1; s< blockDim.x; s*=2){
      if(threadIdx.x % (2*s) == 0){
        double a = partial_max[threadIdx.x];
        double b = partial_max[threadIdx.x + s];
        partial_max[threadIdx.x] = (a >= b) ? a : b;
      }
      __syncthreads();
    }
    if(threadIdx.x == 0){
      temp[blockIdx.x] = partial_max[0]; // store the maximum values of particular thread blocks
    }

  }
}

__global__ void optimized_error_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double * temp){

  // using sequential thread ids and getting rid of modulos operation
  __shared__ double partial_max[NO_THREADS];
  //double error = -1.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x; //i
  if(i < rows){
    double cur_value = 0.0;
    double rsum = 0.0;
    int start = xadj[i];
    int end = xadj[i+1];
    for(int j = 0; j < (end - start); j++){
      cur_value +=  rv[i] * cv[adj[start + j]];
      rsum += cv[adj[start + j]];
    }
    cur_value = fabs(1.0 - cur_value); // compute the local error
    partial_max[threadIdx.x] = cur_value; // store the local error in shared memory
    rv[i] = 1.0/rsum;
    __syncthreads();

    for(int s = 1; s< blockDim.x; s*=2){

      int index = 2 * s * threadIdx.x;
      if(index < blockDim.x){
        double a = partial_max[index];
        double b = partial_max[index + s];
        partial_max[index] = (a >= b) ? a : b;
      }
      __syncthreads();
    }
    if(threadIdx.x == 0){
      temp[blockIdx.x] = partial_max[0]; // store the maximum values of particular thread blocks
    }

  }
}

__global__ void no_conflict_error_scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double * temp){

  __shared__ double partial_max[NO_THREADS];
  //double error = -1.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x; //i
  if(i < rows){
    double cur_value = 0.0;
    double rsum = 0.0;
    int start = xadj[i];
    int end = xadj[i+1];
    for(int j = 0; j < (end - start); j++){
      cur_value +=  rv[i] * cv[adj[start + j]];
      rsum += cv[adj[start + j]];
    }
    cur_value = fabs(1.0 - cur_value); // compute the local error
    partial_max[threadIdx.x] = cur_value; // store the local error in shared memory
    rv[i] = 1.0/rsum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){

      if(threadIdx.x < s){
        double a = partial_max[threadIdx.x];
        double b = partial_max[threadIdx.x + s];
        partial_max[threadIdx.x] = (a >= b) ? a : b;
      }
      __syncthreads();
    }
    if(threadIdx.x == 0){
      temp[blockIdx.x] = partial_max[0]; // store the maximum values of particular thread blocks
    }

  }
}

__global__ void scalesk(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int rows, double * temp){ // perform everything in single function

    // performing every iteration inside a single kernel
    int i = (blockIdx.x * blockDim.x) + threadIdx.x; //n
    __shared__ double partial_max[NO_THREADS];

    if(i < rows){
      int start_row = xadj[i];
      int end_row   = xadj[i+1];
      double rsum   = 0.0;
      for(int j = 0; j < (end_row - start_row); j++){
        rsum += cv[adj[start_row + j]];
      }
      rv[i] = 1.0/rsum;
      __syncthreads();
      int start_col = txadj[i];
      int end_col   = txadj[i+1];
      double csum = 0.0;
      for(int j = 0; j < (end_col - start_col); j++){
        csum += rv[tadj[start_col + j]];
      }
      cv[i] = 1.0/csum;
      __syncthreads();
      // compute max here
      int start = xadj[i];
      int end   = xadj[i+1];
      double cur_value = 0.0;
      for(int j = 0; j < (end - start); j++){
        cur_value +=  rv[i] * cv[adj[start + j]];
      }
      cur_value = fabs(1.0 - cur_value); // compute the local error
      partial_max[threadIdx.x] = cur_value; // store the local error in shared memory
      __syncthreads();

      for(int s = 1; s < blockDim.x; s*=2){
        if(threadIdx.x % (2*s) == 0){
          double a = partial_max[threadIdx.x];
          double b = partial_max[threadIdx.x + s];
          partial_max[threadIdx.x] = (a >= b) ? a : b;
        }
        __syncthreads();
      }
      if(threadIdx.x == 0){
        temp[blockIdx.x] = partial_max[0]; // store the maximum values of particular thread blocks
      }
    }
}

__global__ void final_reduction(double * temp){

  __shared__ double partial_max[NO_THREADS];
  int i = blockIdx.x * blockDim.x + threadIdx.x; //i
  partial_max[threadIdx.x] = temp[i];

  for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
      double a = partial_max[threadIdx.x];
      double b = partial_max[threadIdx.x + s];
      partial_max[threadIdx.x] = (a >= b) ? a : b;
		}
		__syncthreads();
	}
  if (threadIdx.x == 0) {
		temp[blockIdx.x] = partial_max[0];
	}
}

void wrapper(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int* nov, int* nnz, int siter){

  printf("Wrapper here! \n");

  //TO DO: DRIVER CODE
  int rows = *nov;

  cudaEvent_t start,stop;
  float elapsedTime;
  size_t double_size = rows * sizeof(double);
  int* d_adj, *d_xadj, *d_tadj, *d_txadj;
  double *d_rv, *d_cv;

  int NO_BLOCKS = (int)ceil(rows / NO_THREADS + 1);
  size_t double_block_size = NO_BLOCKS * sizeof(double);
  double* temp = new double[NO_BLOCKS];
  double* d_temp;

  //Preparing the memory
  gpuErrchk(cudaMalloc(  (void **) &d_adj, (*nnz) * sizeof(int)));
  gpuErrchk(cudaMalloc(  (void **) &d_xadj, (rows + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(  (void **) &d_tadj, (*nnz) * sizeof(int)));
  gpuErrchk(cudaMalloc(  (void **) &d_txadj, (rows + 1) * sizeof(int) ));
  gpuErrchk(cudaMalloc(  (void **) &d_rv, double_size ));
  gpuErrchk(cudaMalloc(  (void **) &d_cv, double_size ));
  gpuErrchk(cudaMalloc(  (void **) &d_temp, double_block_size ));

  gpuErrchk(cudaPeekAtLastError());
  for(int i = 0; i < rows; i++ ) {
    rv[i] = cv[i] = 1.0;
  }

  gpuErrchk(cudaMemcpy( d_adj, adj, (*nnz) * sizeof(int), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy( d_xadj, xadj, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy( d_tadj, tadj, (*nnz) * sizeof(int), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy( d_txadj, txadj, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy( d_rv, rv, double_size, cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy( d_cv, cv, double_size, cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy( d_temp, temp, double_block_size, cudaMemcpyHostToDevice ));

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  //gpuErrchk(cudaMemset( d_rv, 1.0, double_size));
  //gpuErrchk(cudaMemset( d_cv, 1.0, double_size));
  // int NO_BLOCKS = (rows + NO_THREADS -1) / NO_THREADS;
  //cout << "NO_BLOCKS "<< NO_BLOCKS << " NO_THREADS " << NO_THREADS << endl;

  less_row_scalesk<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_tadj, d_txadj, d_rv, d_cv, rows);
  for(int i = 0; i<siter; i++){
    //less_row_scalesk<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_tadj, d_txadj, d_rv, d_cv, rows);
    less_col_scalesk<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_tadj, d_txadj, d_rv, d_cv, rows);
    no_conflict_error_scalesk<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_tadj, d_txadj, d_rv, d_cv, rows, d_temp);
    gpuErrchk(cudaMemcpy( temp, d_temp, double_block_size, cudaMemcpyDeviceToHost ));
    cout << "iter " << i <<" - error " << *max_element(temp, temp + NO_BLOCKS) << endl;
  }

  gpuErrchk( cudaDeviceSynchronize() );

  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU scale took: %f s\n", elapsedTime/1000);

  // free memory in device and host
  cudaFree(d_adj);
  cudaFree(d_xadj);
  cudaFree(d_tadj);
  cudaFree(d_tadj);
  cudaFree(d_txadj);
  cudaFree(d_rv);
  cudaFree(d_cv);
  cudaFree(d_temp);

  free(adj);
  free(xadj);
  free(tadj);
  free(txadj);
  free(rv);
  free(cv);
  free(temp);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}
