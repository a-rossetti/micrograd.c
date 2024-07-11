CC = gcc
CFLAGS = -Wall -g
LDFLAGS = -lm

# Default target
all: test_engine test_nn

# Build test_engine
test_engine: test_engine.o engine.o
	$(CC) $(CFLAGS) -o test_engine test_engine.o engine.o $(LDFLAGS)

# Build test_nn
test_nn: test_nn.o nn.o engine.o
	$(CC) $(CFLAGS) -o test_nn test_nn.o nn.o engine.o $(LDFLAGS)

# To obtain object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f *.o test_engine test_nn

# Dependencies for the objects
test_engine.o: test_engine.c engine.h
nn.o: nn.c nn.h engine.h
test_nn.o: test_nn.c nn.h engine.h
engine.o: engine.c engine.h

