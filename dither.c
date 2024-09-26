#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 1024
#define MIN(a,b) (((a)<(b))?(a):(b))

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

void dither(uint8_t* array, int width, int height) {
    uint8_t* tmp = (uint8_t*) malloc(height * width * 3 * sizeof(uint8_t));

    double M[4][4] = {
        {0.0/16, 8.0/16, 2.0/16, 10.0/16},
        {12.0/16, 4.0/16, 14.0/16, 6.0/16},
        {3.0/16, 11.0/16, 1.0/16, 9.0/16},
        {15.0/16, 7.0/16, 13.0/16, 5.0/16}
    };

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < 3; c++) {
                int idx = (y*width + x) * 3 + c;
                int old_pixel = array[idx];
                int new_pixel = MIN(255, old_pixel + (int)(M[y%4][x%4] * 255));
                tmp[idx] = (uint8_t)new_pixel;
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

        dither(array, width, height);

        write_array_to_ppm(output_filename, array, width, height);
        printf("image saved as %s\n", output_filename);

        free(array);
    } else {
        fprintf(stderr, "Failed to read input file: %s\n", input_filename);
        return 1;
    }

    return 0;
}
