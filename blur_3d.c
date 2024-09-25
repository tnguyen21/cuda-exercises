#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

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

void free_tensor(uint8_t*** tensor, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            free(tensor[i][j]);
        }
        free(tensor[i]);
    }
    free(tensor);
}

uint8_t*** read_ppm_to_tensor(const char* filename, int* width, int* height) {
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

    // Allocate the 3D tensor
    uint8_t*** tensor = (uint8_t***)malloc(*height * sizeof(uint8_t**));
    for (int i = 0; i < *height; i++) {
        tensor[i] = (uint8_t**)malloc(*width * sizeof(uint8_t*));
        for (int j = 0; j < *width; j++) {
            tensor[i][j] = (uint8_t*)malloc(3 * sizeof(uint8_t));
        }
    }

    // Read pixel data
    for (int i = 0; i < *height; i++) {
        for (int j = 0; j < *width; j++) {
            if (fread(tensor[i][j], sizeof(uint8_t), 3, file) != 3) {
                fprintf(stderr, "Error reading pixel data\n");
                free_tensor(tensor, *height, *width);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    return tensor;
}

void blur(uint8_t*** tensor, int height, int width) {
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            int nPixels = 0;
            int avgR = 0;
            int avgG = 0;
            int avgB = 0;
            

            for (int i = -1; i < 2; ++i) {
                for (int j = -1; j < 2; ++j) {
                    if ((x+i) >= 0 && (x+i) < height && (y+j) >= 0 && (y+j) < width) {
                        avgR += tensor[x+i][y+j][0];
                        avgG += tensor[x+i][y+j][1];
                        avgB += tensor[x+i][y+j][2];
                        nPixels++;
                    }
                }
            }
            uint8_t blurR = (uint8_t)(avgR / nPixels);
            uint8_t blurG = (uint8_t)(avgG / nPixels);
            uint8_t blurB = (uint8_t)(avgB / nPixels);
            tensor[x][y][0] = blurR;
            tensor[x][y][1] = blurG;
            tensor[x][y][2] = blurB;
        }
    }
}

void write_tensor_to_ppm(const char* filename, uint8_t*** tensor, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fwrite(tensor[i][j], sizeof(uint8_t), 3, file);
        }
    }

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
    uint8_t*** tensor = read_ppm_to_tensor(input_filename, &width, &height);

    if (tensor) {
        printf("PPM file read successfully!\n");
        printf("Width: %d, Height: %d\n", width, height);

        blur(tensor, height, width);

        write_tensor_to_ppm(output_filename, tensor, width, height);
        printf("Greyscale image saved as %s\n", output_filename);

        free_tensor(tensor, height, width);
    } else {
        fprintf(stderr, "Failed to read input file: %s\n", input_filename);
        return 1;
    }

    return 0;
}
