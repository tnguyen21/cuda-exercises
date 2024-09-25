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
    
    // Read PPM format
    if (!read_line(file, buffer, MAX_LINE_LENGTH) || strcmp(buffer, "P6\n") != 0) {
        fprintf(stderr, "Unsupported file format. Only P6 PPM is supported.\n");
        fclose(file);
        return NULL;
    }

    // Read width and height
    while (!read_line(file, buffer, MAX_LINE_LENGTH));
    if (sscanf(buffer, "%d %d", width, height) != 2) {
        fprintf(stderr, "Invalid PPM file format\n");
        fclose(file);
        return NULL;
    }

    // Read max color value
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

    // Allocate the 1D array
    uint8_t* array = (uint8_t*)malloc(*width * *height * 3 * sizeof(uint8_t));
    if (!array) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data
    if (fread(array, sizeof(uint8_t), *width * *height * 3, file) != *width * *height * 3) {
        fprintf(stderr, "Error reading pixel data\n");
        free(array);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return array;
}

void blur(uint8_t* array, int width, int height) {
    uint8_t* tmp = (uint8_t*) malloc(height * width * 3 * sizeof(uint8_t));

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int nPixels = 0;
            int avgR = 0;
            int avgG = 0;
            int avgB = 0;

            for (int i = -1; i < 2; ++i) {
                for (int j = -1; j < 2; ++j) {
                    if ((x+i) >= 0 && (x+i) < height && (y+j) >= 0 && (y+j) < width) {
                        int idx = ((x+i) * width + (y+j)) * 3;
                        avgR += array[idx];
                        avgG += array[idx+1];
                        avgB += array[idx+2];
                        nPixels++;
                    }
                }
            }
            
            int idx = (x * width + y) * 3;
            uint8_t blurR = (uint8_t)(avgR / nPixels);
            uint8_t blurG = (uint8_t)(avgG / nPixels);
            uint8_t blurB = (uint8_t)(avgB / nPixels);
            tmp[idx]   = blurR;
            tmp[idx+1] = blurG;
            tmp[idx+2] = blurB;
        }
    }

    memcpy(array, tmp, height * width * 3 * sizeof(uint8_t));
    free(tmp);
}
void write_array_to_ppm(const char* filename, uint8_t* array, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);
    fwrite(array, sizeof(uint8_t), width * height * 3, file);
    fclose(file);
}

void print_usage(const char* program_name) {
    printf("Usage: %s <input_file.ppm> <output_file.ppm>\n", program_name);
    printf("Blurs a PPM image.\n");
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    const char* input_ext = strrchr(input_filename, '.');
    if (!input_ext || strcmp(input_ext, ".ppm") != 0) {
        fprintf(stderr, "Error: Input file must have .ppm extension\n");
        return 1;
    }

    const char* output_ext = strrchr(output_filename, '.');
    if (!output_ext || strcmp(output_ext, ".ppm") != 0) {
        fprintf(stderr, "Error: Output file must have .ppm extension\n");
        return 1;
    }

    int width, height;
    uint8_t* array = read_ppm_to_array(input_filename, &width, &height);

    if (array) {
        printf("PPM file read successfully!\n");
        printf("Width: %d, Height: %d\n", width, height);

        blur(array, width, height);

        write_array_to_ppm(output_filename, array, width, height);
        printf("Blurred image saved as %s\n", output_filename);

        free(array);
    } else {
        fprintf(stderr, "Failed to read input file: %s\n", input_filename);
        return 1;
    }

    return 0;
}
