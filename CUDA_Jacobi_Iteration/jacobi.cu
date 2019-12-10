#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 4
#define M 4
#define EPSILON 0.001
#define ERROR_MAX 999


  //----------Kernal Global Function----------//
__global__ void errorSum(float *a, float *b ,float *x_old, float *x_new, float *sum_x){
  int tId = threadIdx.x;

  for(int i =0;i<N;i++){
    if(tId != i){
      sum_x[tId] +=  a[tId * N + i] * x_old[i];
    }
  }
  x_new[tId] = (b[tId] - sum_x[tId])/a[tId*N + tId];
}


  //----------Read File Function----------//
void readFile(FILE** ptr, const char filename[50], const char mode[10]){
   *ptr = fopen(filename,mode);
   if(filename == NULL){
     printf("Error reading file");
   }
}


int main(){

  //----------Variable Declaration On Host----------//
  int count = 0;
  float *h_a,*h_b,*h_x_new,*h_x_old,*h_error_x,*h_sum_x;
  float error_max = 999;
  float sum_error = 0.0;

  //----------Variable Declaration On Device----------//
  float *d_a;
  float *d_b;
  float *d_x_old;
  float *d_x_new;
  float *d_sum_x;
  float temp_vec;

  //----------Variable Declaration For File Pointers----------//

  FILE *A_ptr, *B_ptr;

  //----------Memory Allocation On Host----------//
  h_a = (float *)malloc(N*M*sizeof(float));
  h_b = (float *)malloc(N*sizeof(float));
  h_x_new = (float *)malloc(N*sizeof(float));
  h_x_old = (float *)malloc(N*sizeof(float));
  h_error_x = (float *)malloc(N*sizeof(float));
  h_sum_x = (float *)malloc(N*sizeof(float));


  //----------Reading From Files----------//
  readFile(&A_ptr,"A_vector.dat","r");
  readFile(&B_ptr,"B_vector.dat","r");


  //----------Copy Data From Files Into A Matrix----------//
  for(int i =0;i<N;i++){
    for(int j =0;j<N;j++){
      fscanf(A_ptr,"%f",&temp_vec);
      h_a[(i*M) + j] = temp_vec;
    }
  }


  //----------Copy Data From Files Into B Vector----------//
  for(int i =0;i<N;i++){
      fscanf(B_ptr,"%f",&temp_vec);
      h_b[i] = temp_vec;
      h_error_x[i] = 999;
  }


  //----------Print A Matrix and B Vector----------//
  for(int i =0;i<N;i++){
    for(int j =0;j<N;j++){
      printf("%0.2f ", h_a[i]);
    }
    printf("\n");
  }

  printf("\n");

  for(int i =0;i<N;i++){
      printf("%0.2f ", h_b[i]);
  }
  printf("\n");


  //----------Memory Allocation On Device----------//
  cudaMalloc((void**) &d_a,N*N*sizeof(float));
  cudaMalloc((void**) &d_b,N*sizeof(float));
  cudaMalloc((void **) &d_x_old,N*sizeof(float));
  cudaMalloc((void  **) &d_x_new,N*sizeof(float));
  cudaMalloc((void  **) &d_sum_x,N*sizeof(float));


  //----------Copying A Matrix and B Vector Memory From Host To Device----------//
  cudaMemcpy(d_a, h_a, N*N*sizeof(float) , cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
 

  //----------EPSILON Calculations----------//
  while(error_max > EPSILON){
    cudaMemcpy(d_x_old, h_x_old, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_new, h_x_new, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_x, h_sum_x, N*sizeof(float), cudaMemcpyHostToDevice);
    //----------Kernal Call----------//
    errorSum<<<1,N>>>(d_a,d_b,d_x_old,d_x_new,d_sum_x);
    cudaMemcpy(h_x_new, d_x_new,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_old, d_x_old,N*sizeof(float),cudaMemcpyDeviceToHost);
    sum_error = 0.0;
    for(int k = 0; k < N;k++){
      h_error_x[k] = h_x_new[k] - h_x_old[k];
      sum_error = sum_error + h_error_x[k] * h_error_x[k];
    }
    error_max = sqrt(sum_error);
    count = count + 1;
    //----------Print Error And Number Of Iterations----------//
    printf("Count: %d | Errror: %0.2f\n",count,error_max);

    for(int l = 0;l < N; l++){
      h_x_old[l] = h_x_new[l];
      }
}

for(int i =0; i<N;i++){
  printf("%s\n"h_x_new[i]);
}

  //----------Free Memory On Device----------//
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_x_old);
  cudaFree(d_x_new);
  cudaFree(d_sum_x);

  //----------Free Memory On Host----------//
  free(h_a);
  free(h_b);
  free(h_x_new);
  free(h_x_old);
  free(h_error_x);
  free(h_sum_x);

  //----------Closed Opened Files----------//
  fclose(A_ptr);
  fclose(B_ptr);

  return 0;
}
