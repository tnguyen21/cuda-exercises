#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

void kuwahara(uint8_t* img, int width, int height, int ksize) {
    uint8_t* tmp = (uint8_t*)malloc(height * width * 3 * sizeof(uint8_t)); 

    int pad = ksize / 2;
    int stride = width * 3;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double min_var = DBL_MAX;
            int best_mean[3] = {0, 0, 0};

            int regions[4][4] = {
                {-pad, -pad, 0, 0},           // top left
                {-pad, 0, 0, pad},            // top right
                {0, -pad, pad, 0},            // bottom left
                {0, 0, pad, pad}              // bottom right
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
                                int val = img[idx + c];
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
                        var += (sum_sq[c] - 2 * mean * sum[c] + count * mean * mean) / count;
                    }
                    var /= 3;

                    if (var < min_var) {
                        min_var = var;
                        for (int c = 0; c < 3; c++) {
                            best_mean[c] = sum[c] / count;
                        }
                    }
                }
            }

            int out_idx = (y * width + x) * 3;
            for (int c = 0; c < 3; c++) {
                tmp[out_idx + c] = (uint8_t)best_mean[c];
            }
        }
    }

    memcpy(img, tmp, height * width * 3 * sizeof(uint8_t));
    free(tmp);
}

// Helper function to read PPM image
uint8_t* read_ppm(const char* filename, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        return NULL;
    }

    char header[3];
    fscanf(file, "%2s", header);
    if (header[0] != 'P' || header[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        fclose(file);
        return NULL;
    }

    fscanf(file, "%d %d", width, height);
    int max_value;
    fscanf(file, "%d", &max_value);
    fgetc(file);  // Skip newline

    uint8_t* img = (uint8_t*)malloc((*width) * (*height) * 3);
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        fclose(file);
        return NULL;
    }

    fread(img, 1, (*width) * (*height) * 3, file);
    fclose(file);
    return img;
}

// Helper function to write PPM image
void write_ppm(const char* filename, uint8_t* img, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);
    fwrite(img, 1, width * height * 3, file);
    fclose(file);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.ppm> <output.ppm> <kernel_size>\n", argv[0]);
        return 1;
    }

    int width, height;
    uint8_t* img = read_ppm(argv[1], &width, &height);
    if (!img) {
        return 1;
    }

    int ksize = atoi(argv[3]);
    if (ksize % 2 == 0 || ksize < 3) {
        fprintf(stderr, "Kernel size must be odd and greater than 1\n");
        free(img);
        return 1;
    }

    kuwahara(img, width, height, ksize);
    write_ppm(argv[2], img, width, height);

    free(img);
    return 0;
}
