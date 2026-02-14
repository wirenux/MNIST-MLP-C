UNAME_S := $(shell uname -s)

CC = gcc
TARGET = build/mnist
SRC = src/main.c

# SDL2 flags
SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LIBS   := $(shell sdl2-config --libs)

# Default flags (Linux)
CFLAGS = -O3 -ffast-math -march=native -fopenmp $(SDL_CFLAGS)
LDFLAGS = -fopenmp -lm $(SDL_LIBS)
INCLUDES =

# macOS (Apple Silicon)
ifeq ($(UNAME_S), Darwin)
    CC = clang
    BREW_PREFIX = $(shell brew --prefix libomp 2>/dev/null)

    CFLAGS = -O3 -ffast-math -mcpu=apple-m1 \
             -Xpreprocessor -fopenmp \
             $(SDL_CFLAGS)

    INCLUDES = -I$(BREW_PREFIX)/include
    LDFLAGS = -L$(BREW_PREFIX)/lib -lomp -lm $(SDL_LIBS)
endif

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

run: all
	@./$(TARGET)

clean:
	rm -rf build
