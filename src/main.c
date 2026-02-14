#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <SDL.h>
#include <string.h>
#include <omp.h>

#define INPUT_SIZE 784      // nb of pixels in 28x28 image
#define HIDDEN_SIZE 128     // nb of neurons in hidden layer
#define OUTPUT_SIZE 10      // nb of classes (digits 0-9)
#define TRAIN_SIZE 60000    // nb of training samples in MNIST
#define BATCH_SIZE 32

#define WIN 280
#define GRID 28
#define SCALE 10

uint8_t canvas[GRID][GRID];
int brush_size = 1;

typedef struct {
    float w1[HIDDEN_SIZE][INPUT_SIZE];
    float b1[HIDDEN_SIZE];
    float w2[OUTPUT_SIZE][HIDDEN_SIZE];
    float b2[OUTPUT_SIZE];
} Model;

void softmax(float *input, int size);
void print_confidence_graph(float *scores);
void predict_from_buffer(uint8_t *pixels, Model *m);
void predict_external_image(char *filename, Model *m);
void sdl_draw_and_predict(Model *m);
void init_model(Model *m);

void draw_brush(int gx, int gy, int value) {
    for (int dy = -brush_size; dy <= brush_size; dy++) {
        for (int dx = -brush_size; dx <= brush_size; dx++) {
            if (dx*dx + dy*dy <= brush_size*brush_size) {
                int x = gx + dx;
                int y = gy + dy;
                if (x >= 0 && x < 28 && y >= 0 && y < 28)
                    canvas[y][x] = value;
            }
        }
    }
}

void canvas_to_buffer(uint8_t out[784]) {
    for (int y = 0; y < 28; y++)
        for (int x = 0; x < 28; x++)
            out[y*28 + x] = canvas[y][x];
}

#pragma GCC optimize ("no-tree-vectorize")
void predict_from_buffer(uint8_t *pixels, Model *m) {
    float h_layer[HIDDEN_SIZE], scores[OUTPUT_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float sum = m->b1[j];
        for (int k = 0; k < 784; k++)
            sum += (pixels[k] / 255.0f) * m->w1[j][k];
        h_layer[j] = (sum > 0) ? sum : 0;
    }

    for (int j = 0; j < OUTPUT_SIZE; j++) {
        scores[j] = m->b2[j];
        for (int k = 0; k < HIDDEN_SIZE; k++)
            scores[j] += h_layer[k] * m->w2[j][k];
    }

    softmax(scores, OUTPUT_SIZE);
    print_confidence_graph(scores);

    int pred = 0;
    for (int i = 1; i < 10; i++)
        if (scores[i] > scores[pred]) pred = i;

    printf("\n\033[1;34mDRAW RESULT: %d\033[0m\n", pred);
}

void sdl_draw_and_predict(Model *m) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win = SDL_CreateWindow("Draw Digit (- ENTER to predict -)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN, WIN, 0);

    SDL_Renderer *r = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    memset(canvas, 0, sizeof(canvas));

    int running = 1;
    int do_predict = 0;

    printf("\n--- DRAW MODE ---\n");
    printf("Left click = draw\n");
    printf("Right click = erase\n");
    printf("+ / - = brush size\n");
    printf("C = clear\n");
    printf("ENTER = predict\n");
    printf("ESC = cancel\n\n");

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {

            if (e.type == SDL_QUIT) running = 0;

            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    running = 0;
                }

                if (e.key.keysym.sym == SDLK_RETURN) {
                    do_predict = 1;
                    running = 0;
                }

                if (e.key.keysym.sym == SDLK_c) {
                    memset(canvas, 0, sizeof(canvas));
                }

                if (e.key.keysym.sym == SDLK_EQUALS || e.key.keysym.sym == SDLK_PLUS) {
                    if (brush_size < 4) brush_size++;
                }

                if (e.key.keysym.sym == SDLK_MINUS) {
                    if (brush_size > 1) brush_size--;
                }
            }

            if (e.type == SDL_MOUSEMOTION || e.type == SDL_MOUSEBUTTONDOWN) {
                int mx, my;
                SDL_GetMouseState(&mx, &my);
                int gx = mx / SCALE;
                int gy = my / SCALE;

                if (gx>=0 && gx<28 && gy>=0 && gy<28) {
                    if (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_LEFT))
                        draw_brush(gx, gy, 255);

                    if (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_RIGHT))
                        draw_brush(gx, gy, 0);
                }
            }
        }

        SDL_SetRenderDrawColor(r, 0,0,0,255);
        SDL_RenderClear(r);

        for (int y=0;y<28;y++){
            for(int x=0;x<28;x++){
                uint8_t v = canvas[y][x];
                SDL_SetRenderDrawColor(r, v,v,v,255);
                SDL_Rect rect = {x*SCALE, y*SCALE, SCALE, SCALE};
                SDL_RenderFillRect(r, &rect);
            }
        }

        SDL_RenderPresent(r);
        SDL_Delay(16);
    }

    if (do_predict) {
        uint8_t buffer[784];
        canvas_to_buffer(buffer);
        predict_from_buffer(buffer, m);
    }

    SDL_DestroyRenderer(r);
    SDL_DestroyWindow(win);
    SDL_Quit();
}

void softmax(float *input, int size) {
    float max = input[0], sum = 0.0;
    for (int i = 1; i < size; i++) if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) input[i] /= sum;
}

// print graph : [4] : ■■■■■■■■■■■■■■■■■■■  99.99%
void print_confidence_graph(float *scores) {
    printf("\n\033[1mIA trust:\033[0m\n");
    for (int i = 0; i < 10; i++) {
        printf("[%d] : ", i);

        int max_bars = 40;
        int bars = (int)(scores[i] * max_bars);

        for (int b = 0; b < bars; b++) printf("■");
        for (int b = bars; b < max_bars; b++) printf(" ");

        printf(" %.2f%%\n", scores[i] * 100.0f);
    }
}

void predict_external_image(char *filename, Model *m) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR : Can't open %s\n", filename);
        return;
    }

    char magic[3]; int w, h, max;
    fscanf(f, "%s %d %d %d", magic, &w, &h, &max); fgetc(f);
    if (w != 28 || h != 28) {
        printf("ERROR : Image must be 28x28\n");
        fclose(f);
        return;
    }

    unsigned char pixels[784];
    fread(pixels, 1, 784, f);
    fclose(f);

    printf("\nImage loaded (%s) :\n", filename);
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            unsigned char p = pixels[r * 28 + c];
            if (p > 220)      printf("██"); // Full density
            else if (p > 150) printf("▓▓"); // Dark shade
            else if (p > 80)  printf("▒▒"); // Medium shade
            else if (p > 20)  printf("░░"); // Light shade
            else              printf("  "); // Empty
        }
        printf("\n");
    }

    float h_layer[HIDDEN_SIZE], scores[OUTPUT_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float sum = m->b1[j];
        for (int k = 0; k < 784; k++) {
            sum += (pixels[k] / 255.0f) * m->w1[j][k];
        }
        h_layer[j] = (sum > 0) ? sum : 0;
    }
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        scores[j] = m->b2[j];
        for (int k = 0; k < HIDDEN_SIZE; k++) scores[j] += h_layer[k] * m->w2[j][k];
    }
    softmax(scores, OUTPUT_SIZE);

    print_confidence_graph(scores);
    int pred = 0;
    for (int i = 1; i < 10; i++) if (scores[i] > scores[pred]) pred = i;
    printf("\n\033[1;34mFINAL RESULT: %d\033[0m\n", pred);
}

void init_model(Model *m) {
    float scale1 = sqrtf(2.0f / INPUT_SIZE);
    float scale2 = sqrtf(2.0f / HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        m->b1[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++)
            m->w1[i][j] = ((float)random()/RAND_MAX * 2.0f - 1.0f) * scale1;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        m->b2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            m->w2[i][j] = ((float)random()/RAND_MAX * 2.0f - 1.0f) * scale2;
    }
}

int main() {
    srandom(time(0));

    FILE *f_img = fopen("data/train-images-idx3-ubyte", "rb");
    FILE *f_lbl = fopen("data/train-labels-idx1-ubyte", "rb");
    if (!f_img || !f_lbl) { printf("MNIST missing\n"); return 1; }
    fseek(f_img, 16, SEEK_SET); fseek(f_lbl, 8, SEEK_SET);
    unsigned char *raw_imgs = malloc(TRAIN_SIZE * INPUT_SIZE);
    unsigned char *raw_lbls = malloc(TRAIN_SIZE);
    fread(raw_imgs, 1, TRAIN_SIZE * INPUT_SIZE, f_img);
    fread(raw_lbls, 1, TRAIN_SIZE, f_lbl);
    fclose(f_img); fclose(f_lbl);

    float *normalized_imgs = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    for (int i = 0; i < TRAIN_SIZE * INPUT_SIZE; i++) {
        normalized_imgs[i] = raw_imgs[i] / 255.0f;
    }

    Model *m = malloc(sizeof(Model));
    FILE *f_mod = fopen("data/mnist_mlp_best.bin", "rb");
    if (f_mod) {
        fread(m, sizeof(Model), 1, f_mod);
        fclose(f_mod);
        printf("\033[0;32mMLP model loaded!\033[0m\n");
    } else {
        printf("No existing model found, starting training...\n");
        init_model(m);
        float lr = 0.05f;
        for (int epoch = 0; epoch < 100; epoch++) {
            int correct = 0;
            double start = omp_get_wtime();
            #pragma omp parallel for reduction(+:correct) schedule(dynamic)
            for (int b = 0; b < TRAIN_SIZE; b += BATCH_SIZE) {
                Model grad = {0};
                for (int i = b; i < b + BATCH_SIZE && i < TRAIN_SIZE; i++) {
                    float *img = &normalized_imgs[i * INPUT_SIZE];
                    float h_layer[HIDDEN_SIZE], scores[OUTPUT_SIZE];
                    for (int j = 0; j < HIDDEN_SIZE; j++) {
                        float sum = m->b1[j];
                        for (int k = 0; k < INPUT_SIZE; k++) sum += img[k] * m->w1[j][k];
                        h_layer[j] = (sum > 0) ? sum : 0;
                    }
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        scores[j] = m->b2[j];
                        for (int k = 0; k < HIDDEN_SIZE; k++) scores[j] += h_layer[k] * m->w2[j][k];
                    }
                    softmax(scores, OUTPUT_SIZE);
                    int pred = 0;
                    for (int j = 1; j < OUTPUT_SIZE; j++) if (scores[j] > scores[pred]) pred = j;
                    if (pred == raw_lbls[i]) correct++;

                    float d_out[OUTPUT_SIZE];
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        d_out[j] = scores[j] - (j == raw_lbls[i] ? 1.0f : 0.0f);
                        grad.b2[j] += d_out[j];
                        for (int k = 0; k < HIDDEN_SIZE; k++) grad.w2[j][k] += d_out[j] * h_layer[k];
                    }
                    for (int j = 0; j < HIDDEN_SIZE; j++) {
                        if (h_layer[j] <= 0) continue;
                        float d_h = 0;
                        for (int k = 0; k < OUTPUT_SIZE; k++) d_h += d_out[k] * m->w2[k][j];
                        grad.b1[j] += d_h;
                        for (int k = 0; k < INPUT_SIZE; k++) grad.w1[j][k] += d_h * img[k];
                    }
                }
                #pragma omp critical
                {
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        m->b2[j] -= lr * grad.b2[j] / BATCH_SIZE;
                        for (int k = 0; k < HIDDEN_SIZE; k++) m->w2[j][k] -= lr * grad.w2[j][k] / BATCH_SIZE;
                    }
                    for (int j = 0; j < HIDDEN_SIZE; j++) {
                        m->b1[j] -= lr * grad.b1[j] / BATCH_SIZE;
                        for (int k = 0; k < INPUT_SIZE; k++) m->w1[j][k] -= lr * grad.w1[j][k] / BATCH_SIZE;
                    }
                }
            }
            printf("Epoch %d : Precision %.2f%%\n", epoch, (float)correct/600);
            if (epoch % 20 == 0) lr *= 0.8f;
        }
        FILE *f_save = fopen("data/mnist_mlp_best.bin", "wb");
        fwrite(m, sizeof(Model), 1, f_save);
        fclose(f_save);
    }

    int idx;
    while(1) {
        printf("\n[0-59999]=MNIST | -3=Draw | -2=PGM | -1=Quit : ");
        if (scanf("%d", &idx) != 1 || idx == -1) break;

        if (idx == -2) {
            predict_external_image("data/MNIST_TEST.pgm", m);
            continue;
        }
        if (idx == -3) {
            sdl_draw_and_predict(m);
            continue;
        }
        if (idx < 0 || idx > 59999) continue;

        printf("\nImage :\n");
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                unsigned char p = raw_imgs[idx * INPUT_SIZE + (r * 28 + c)];
                if (p > 220)      printf("██");
                else if (p > 150) printf("▓▓");
                else if (p > 80)  printf("▒▒");
                else if (p > 20)  printf("░░");
                else              printf("  ");
            }
            printf("\n");
        }

        float hl[HIDDEN_SIZE], sc[OUTPUT_SIZE];
        float *img = &normalized_imgs[idx * INPUT_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float s = m->b1[j];
            for (int k = 0; k < INPUT_SIZE; k++) s += img[k] * m->w1[j][k];
            hl[j] = (s > 0) ? s : 0;
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sc[j] = m->b2[j];
            for (int k = 0; k < HIDDEN_SIZE; k++) sc[j] += hl[k] * m->w2[j][k];
        }
        softmax(sc, 10);
        print_confidence_graph(sc);

        int best = 0;
        for (int i = 1; i < 10; i++) if (sc[i] > sc[best]) best = i;
        if (best == raw_lbls[idx]) {
            printf("\n\033[0;32mSUCCESS (Real: %d)\033[0m\n", raw_lbls[idx]);
        } else {
            printf("\n\033[0;31mFAILURE (Real: %d)\033[0m\n", raw_lbls[idx]);
        }
    }

    free(raw_imgs); free(raw_lbls); free(normalized_imgs); free(m);
    return 0;
}