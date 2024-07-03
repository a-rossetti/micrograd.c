#ifndef VALUE_H
#define VALUE_H

#include <stdbool.h>

typedef struct Value {
    double data;                    // scalar value
    double grad;                    // gradient of the value
    struct Value* prev[2];          // pointers to previous values (binary operations only)
    char op[10];                    // operation that produced this value
    void (*backward)(struct Value*) // Function pointer for backpropagation
} Value;

Value* create_value(double data);
Value* add(Value* a, Value* b);
Value* multiply(Value* a, Value* b);
Value* pow(Value* a, Value* b);
Value* relu(Value* a, Value* b);

#endif

