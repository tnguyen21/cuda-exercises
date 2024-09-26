#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <float.h>

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

void kuwahara(uint8_t* array, int height, int width, int ksize) {
    int pad = ksize / 2;
    int stride = width * 3;

    uint8_t* tmp = (uint8_t*) malloc(height * width * 3 * sizeof(uint8_t));

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            double min_var = DBL_MAX;
            int best_mean[3] = {0, 0, 0};

            int regions[4][4] = {
                {-pad, -pad, 0, 0},
                {-pad, 0, 0, pad},
                {0, -pad, pad, 0},
                {0, 0, pad, pad}
            };

            for (int r = 0; r < 4; r++) {
                int sum[3] = {0, 0, 0};
                int sum_sq[3] = {0, 0, 0};
                int count = 0;
                
                for (int i = regions[r][0]; i <= regions[r][2]; i++) {
                    for (int j = regions[r][1]; j <= regions[r][3]; j++) {
                        int yi = y + i;
                        int xj = x + j;
                        if (yi >= 0 && yi < height && xj >= 0 && xj < width) {
                            int idx = (yi * width + xj) * 3;
                            for (int c = 0; c < 3; c++) {
                                int val = array[idx + c];
                                sum[c] += val;
                                sum_sq[c] += val * val;
                            }
                            count++;
                        }
                    }
                }

                if (count > 0) {
                    double var = 0;
                    for (int c = 0; c < 3; c++) {
                        double mean = (double)sum[c] / count;
                        var += (sum_sq[c] - 2*mean*sum[c] + count*mean*mean) / count;
                    }
                    var /= 3;
                
                    if (var < min_var) {
                        min_var = var;
                        for (int c =0; c < 3; c++) {
                            best_mean[c] = sum[c] / count;
                        }
                    }
                }
            }

            int idx = (x * width + y) * 3;
            for (int c = 0; c < 3; c++) {
                tmp[idx + c] = (uint8_t)best_mean[c];
            }  
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

        kuwahara(array, width, height, 3);

        write_array_to_ppm(output_filename, array, width, height);
        printf("image saved as %s\n", output_filename);

        free(array);
    } else {
        fprintf(stderr, "Failed to read input file: %s\n", input_filename);
        return 1;
    }

    return 0;
}
