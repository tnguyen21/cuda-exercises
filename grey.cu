#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 1024

// Helper function to read a line and skip comments
char* read_line(FILE* file, char* buffer, int max_length) {
    while (fgets(buffer, max_length, file) != NULL) {
        if (buffer[0] != '#') {
            return buffer;
        }
    }
    return NULL;
}

uint8_t* read_ppm_to_array(const char* filename, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file\n");
        return NULL;
    }

    char buffer[MAX_LINE_LENGTH];
    
    if (!read_line(file, buffer, MAX_LINE_LENGTH) || strcmp(buffer, "P6\n") != 0) {
        fprintf(stderr, "Unsupported file format. Only P6 PPM is supported.\n");
        fclose(file);
        return NULL;
    }

    while (!read_line(file, buffer, MAX_LINE_LENGTH));
    if (sscanf(buffer, "%d %d", width, height) != 2) {
        fprintf(stderr, "Invalid PPM file format\n");
        fclose(file);
        return NULL;
    }

    int max_color;
    while (!read_line(file, buffer, MAX_LINE_LENGTH));
    if (sscanf(buffer, "%d", &max_color) != 1) {
        fprintf(stderr, "Invalid PPM file format\n");
        fclose(file);
        return NULL;
    }

    if (max_color != 255) {
        fprintf(stderr, "Unsupported max color value. Only 8-bit color depth is supported.\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data
    uint8_t* array = (uint8_t*)malloc(*width * *height * 3 * sizeof(uint8_t));
    if (!array) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    if (fread(array, sizeof(uint8_t), *width * *height * 3, file) != *width * *height * 3) {
      fprintf(stderr, "Error reading pixel data\n");
      free(array);
      fclose(file);
      return NULL;
    }

    fclose(file);
    return array;
}

__global__
void greyscaleKernel(uint8_t* arr_in, uint8_t* arr_out, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) {
        uint8_t grey = (uint8_t)(0.299*arr_in[i*3] + 0.587*arr_in[i*3+1] + 0.114*arr_in[i*3+2]);

        arr_out[i*3] = grey;
        arr_out[i*3+1] = grey;
        arr_out[i*3+2] = grey;
    }
}

void convert_to_greyscale(uint8_t* arr_in, uint8_t* arr_out, int height, int width) {
    // allocate memory on device
    int size = height * width * 3 * sizeof(uint8_t);
    uint8_t *arr_in_d, *arr_out_d;

    cudaMalloc((void **) &arr_in_d, size);
    cudaMalloc((void **) &arr_out_d, size); 
    
    // copy host data -> device
    cudaMemcpy(arr_in_d, arr_in, size, cudaMemcpyHostToDevice);

    // call kernel
    int nThreads = 256;
    int nBlocks = (size + nThreads - 1) / nThreads;
    greyscaleKernel<<<nBlocks, nThreads>>>(arr_in_d, arr_out_d, size);

    // copy device result -> host
    cudaMemcpy(arr_out, arr_out_d, size, cudaMemcpyDeviceToHost);

    // cudaFree any data
    cudaFree(arr_in_d);
    cudaFree(arr_out_d);
}

void write_tensor_to_ppm(const char* filename, uint8_t* array, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }

    // Write PPM header
    fprintf(file, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    fwrite(array, sizeof(uint8_t), width*height*3, file);

    fclose(file);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file.ppm> <output_file.ppm>\n", argv[0]);
        printf("Converts a PPM image to greyscale.\n");
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    // Check if input file has .ppm extension
    const char* input_ext = strrchr(input_filename, '.');
    if (!input_ext || strcmp(input_ext, ".ppm") != 0) {
        fprintf(stderr, "Error: Input file must have .ppm extension\n");
        return 1;
    }

    // Check if output file has .ppm extension
    const char* output_ext = strrchr(output_filename, '.');
    if (!output_ext || strcmp(output_ext, ".ppm") != 0) {
        fprintf(stderr, "Error: Output file must have .ppm extension\n");
        return 1;
    }

    int width, height;
    uint8_t* array = read_ppm_to_array(input_filename, &width, &height);
    uint8_t array_out[width*height*3];

    if (array) {
        printf("PPM file read successfully!\n");
        printf("Width: %d, Height: %d\n", width, height);

        // Convert the image to greyscale
        convert_to_greyscale(array, array_out, height, width);

        // Write the modified tensor back to a new PPM file
        write_tensor_to_ppm(output_filename, array_out, width, height);
        printf("Greyscale image saved as %s\n", output_filename);

        // Free array 
    } else {
        fprintf(stderr, "Failed to read input file: %s\n", input_filename);
        return 1;
    }

    return 0;
}

