UNAME_S := $(shell uname -s)

CC = gcc
CFLAGS = -O3 -ffast-math -march=native -fopenmp
LDFLAGS = -fopenmp -lm
INCLUDES = 

ifeq ($(UNAME_S), Darwin)
    CC = clang
    BREW_PREFIX = $(shell brew --prefix libomp 2>/dev/null)
    
    CFLAGS = -O3 -ffast-math -mcpu=apple-m1 -Xpreprocessor -fopenmp
    INCLUDES = -I$(BREW_PREFIX)/include
    LDFLAGS = -L$(BREW_PREFIX)/lib -lomp -lm
endif

TARGET = build/mnist
SRC = src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

run: all
	@./$(TARGET)

clean:
	rm -rf build/
