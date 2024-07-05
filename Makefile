CC = gcc
CFLAGS = -std=c11 -Wall -Wextra
LDFLAGS = -lm

SRCS = main.c engine.c
OBJS = $(SRCS:.c=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)

