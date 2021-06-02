#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <algorithm>
//#include <iostream>
#include <math.h>
#include <cmath>
using namespace std;
//this will be your implemantation of MPI_Alltoall
//it is simpler than the original one: each processor has msg_size integers to send
//                                   : the number of processors is a power of 2 (IMPORTANT)
//sb: is the send buffer
//rb: is the receive buffer
//msg_size: number of integers to send
//comm: MPI_Comm communicator
int MPI_Alltoall_int(int* sb, int* rb, int msg_size, MPI_Comm comm, int proc_size, int rank) { // all to all broadcast hypercube implementation
  //you are only allowed to use pairwise communications: i.e., MPI_Send and MPI_Receive
  MPI_Status status;
  int dimension = log2(proc_size);
  int partner,temp;
  int current_msg_size = msg_size;
  int * result;
  int * msg;
  for(int i = 0; i<msg_size; i++){
    rb[(rank * msg_size) + i]  = rank;
  }
  for(int i = 0; i < dimension; i++){
      result = (int *) malloc(sizeof(int)*current_msg_size);
      msg = (int *) malloc(sizeof(int)*current_msg_size);
      temp = pow(2,i);
      partner = rank ^ temp; // determine the neighbour
      for(int i = 0; i<current_msg_size; i++){
        result[i] = rb[(sb[0] * msg_size) + i]; // prepare the message to be sent to the neighbour
      }
      if(rank < partner){
        MPI_Send(result, current_msg_size, MPI_INT, partner, rank, comm);  // current rank sends mesage to partner first
        MPI_Recv(msg, current_msg_size, MPI_INT, partner,MPI_ANY_TAG , comm, &status); // current rank receives message from partner secondly
      }else{
        MPI_Recv(msg, current_msg_size, MPI_INT, partner, MPI_ANY_TAG , comm, &status); // current rank receives message from partner first
        MPI_Send(result, current_msg_size, MPI_INT, partner, rank, comm); // current rank sends message to partner secondly
      }
      for(int i = 0; i < current_msg_size ; i++){ // update the rb buffer
        if(msg[i] < sb[0]){
          sb[0] = msg[i]; // find the rank of the minimum element that is contained in the msg
        }
        rb[(msg[0] * msg_size) + i] = msg[i];
      }
      current_msg_size *= 2; // every iteration message size will be doubled
  }
  return 0; //0 is for success
}

int main( int argc, char *argv[]) {
  int rank, size;
  int msg_size = 128; //no ints
  int i;
  int *sb;
  int *rb;
  int status, gstatus;
  int debug;
  double t1, t2;
  /*******************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  msg_size = atoi(argv[1]);
  debug = atoi(argv[2]);

  for ( i=1 ; i < argc ; ++i ) {
    if (argv[i][0] != '-' )
      continue;
    switch(argv[i][1]) {
    case 'm':
      msg_size = atoi(argv[++i]);
      break;
    default:
      fprintf(stderr, "Unrecognized argument %s\n", argv[i]);fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  sb = (int *)malloc(size * msg_size * sizeof(int));
  rb = (int *)malloc(size * msg_size * sizeof(int));

  /*************** MPI_Alltoall ***************************/
  for (i = 0; i < size * msg_size; ++i) {
    sb[i] = rank;
    rb[i] = 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  status = MPI_Alltoall(sb, msg_size, MPI_INT, rb, msg_size, MPI_INT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
	double sum = 0;
   	for(i = 0; i < size * msg_size; i++) {
    	sum += rb[i];
    }
    printf("MPI: elapsed time is %f - %f = %f\n", t2 - t1, sum, msg_size * (size * (size - 1.0) / 2));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(debug && rank == size - 1) {
    for(i = 0; i < size * msg_size; i++) {
    	printf("%d ", rb[i]);
    }
    printf("\n");
  }

  /*************** Custom MPI_Alltoall ********************/
  for (i = 0; i < size * msg_size; ++i) {
    sb[i] = rank;
    rb[i] = 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  status = MPI_Alltoall_int(sb, rb, msg_size, MPI_COMM_WORLD,size, rank); // added two additional parameters (processor size, rank of the processor)
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
	double sum = 0;
   	for(i = 0; i < size * msg_size; i++) {
    	sum += rb[i];
    }
    printf("MPI: elapsed time is %f - %f = %f\n", t2 - t1, sum, msg_size * (size * (size - 1.0) / 2));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if(debug && rank == size -1) { // size  -1
    for(i = 0; i < size * msg_size; i++) {
    	printf("%d ", rb[i]);
    }
    printf("\n");
  }
  //*******************************************************/
  free(sb);
  free(rb);
  MPI_Finalize();
  return 0;
}
