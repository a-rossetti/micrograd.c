#ifndef ENGINE_H
#define ENGINE_H

#include <stdbool.h>

typedef struct Value {
    double data;                        // scalar value
    double grad;                        // gradient of the value
    struct Value* prev[2];              // pointers to previous values (binary operations only)
    char op[10];                        // operation that produced this value
    void (*backward)(struct Value*);    // Function pointer for backpropagation
} Value;

Value* create_value(double data);
Value* add(Value* a, Value* b);
Value* mul(Value* a, Value* b);
Value* power(Value* a, double b);
Value* relu(Value* a);
Value* neg(Value* a);
Value* sub(Value* a, Value* b);
Value* truediv(Value* a, Value* b);
void backward(Value* v);

#endif

