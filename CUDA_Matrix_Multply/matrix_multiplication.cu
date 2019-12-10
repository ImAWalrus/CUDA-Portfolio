//Got Help from Henry
#include <stdio.h> //Standard Input/Output Lib
#include <stdlib.h> //Standard Lib
#define N 3 //Dimensions for row matirx
#define M 3 //Dimensions for column matrix


/* Call Kernal and pass in flat A matrix and B vector
   Matrix Multiply A and B and store output in C array */
__global__ void matrix_vector_mult(float *a, float *b, float *c){
  int tId = threadIdx.x; //Thread ID(Row Index) for flat A Matrix
  float sum = 0.0;
  for(int i = 0; i < M; i++){ //Iterate through Columns
    //Will allow us to iterate through columns
    sum += a[tId * N + i] * b[i]; //(Row Index * 3) + column
  }
  c[tId] = sum; //Add Values to tID(Current row index)
}

/*ALLOCATE MEMORY ARRAY */
void allocate_mem(float** arr1d, int n,int m){
    *arr1d = (float*)malloc(n*m*sizeof(float));
}

int main(){
  float *h_a; //HOST | CPU
  float *h_b; //HOST | CPU
  float *h_c; //HOST | CPU

  float *d_a; //DEVICE | GPU
  float *d_b; //DEVICE | GPU
  float *d_c; //DEVICE | GPU

  /*ALLOCATE 1D IN CPU MEM */
  allocate_mem(&h_b,M,1);
  allocate_mem(&h_c,M,1);
  //Fill 1D Vectors
  for(int i = 0; i < M; i++){
    h_b[i] = i; //0, 1 ,2
    h_c[i] = 0.0;// 0.0, 0.0, 0.0
    printf("h_b:%0.1f h_c%0.1f\n", h_b[i], h_c[i]);
  }

  /* ALLOCATE 2D IN CPU MEM */
  allocate_mem(&h_a,N,M);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M;j++){
        h_a[i * N + j] = j; //0,1,2 0,1,2 0,1,2
        printf("h_a:%0.1f\n",h_a[i * N + j]);
    }
  }
  /* ALLOCATE CUDA MEMORY */
  cudaMalloc((void**) &d_a, N*M*sizeof(float));
  cudaMalloc((void**) &d_b, N*sizeof(float));
  cudaMalloc((void**) &d_c, N*sizeof(float));

  /* Copy Memory From Host(CPU) to Device(GPU)*/
  cudaMemcpy(d_a, h_a, N*M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, M*sizeof(float), cudaMemcpyHostToDevice);

  /* Invoke Kernal (GPU)*/
  matrix_vector_mult<<<1,N>>>(d_a,d_b,d_c);
  /* Copy Memory Back From Device(GPU) to Host(CPU)*/
  cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
  /* Print Output from Multiplied Matrices*/
  for(int i = 0; i < N; i++){
    printf("%0.1f\n", h_c[i]);
  }

  /* Free DEVICE(GPU) Memory First(ALWAYS)*/
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  /*Free HOST(CPU) Memory Last */
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
