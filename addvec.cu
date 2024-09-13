#include <stdio.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
  for (int i=0; i<n; ++i) {
      C_h[i] = A_h[i] + B_h[i];
    }
}

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) { //init
    C[i] = A[i] + B[i];
  }
}

void cuVecAdd(float* A, float* B, float* C, int n) {
  int size = n*sizeof(float);
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **) &A_d, size);
  cudaMalloc((void **) &B_d, size);
  cudaMalloc((void **) &C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  // call kernel
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
  
  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  int n = 256;
  float A[n], B[n], C[n];

  for (int i=0; i<n; ++i) {
    A[i] = 1.1;
    B[i] = 2.3;
    C[i] = 0.0;
  }
  
  cuVecAdd(A, B, C, n);

  for (int i=0; i<n; ++i) {
    printf("C[%d] = %f\n", i, C[i]);
  }
 
  return 0;
} 
