#include <stdio.h>
#include <stdint.h>

#define ARRAY_SIZE 1024
#define BLOCK_SIZE 256

__global__
void copyKernel(const uint8_t* input, uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

int main() {
    uint8_t* h_input;
    uint8_t* h_output;
    uint8_t* d_input;
    uint8_t* d_output;
    int size = ARRAY_SIZE * sizeof(uint8_t);

    // Allocate host memory
    h_input = (uint8_t*)malloc(size);
    h_output = (uint8_t*)malloc(size);

    // Initialize input array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (uint8_t)(i % 256);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input array from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copyKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, ARRAY_SIZE);

    // Copy output array from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (h_input[i] != h_output[i]) {
            printf("Mismatch at index %d: input = %d, output = %d\n", i, h_input[i], h_output[i]);
        }
    }

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
