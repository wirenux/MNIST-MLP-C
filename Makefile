# M1 Pro Optimization
CC = clang
CFLAGS = -O3 -ffast-math -mcpu=apple-m1 -Xpreprocessor -fopenmp
LDFLAGS = -lomp -L$(shell brew --prefix libomp)/lib
INCLUDES = -I$(shell brew --prefix libomp)/include

all:
	$(CC) $(CFLAGS) $(INCLUDES) main.c -o mnist $(LDFLAGS) -lm

run: all
	@./mnist