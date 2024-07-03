// value.c
#include <stdlib.h>
#include <string.h>
#include "value.h"

// backward functions
void add_backward(Value* out) {
    Value *a = out->prev[0];
    Value *b = out->prev[1];
    a->grad += out->grad;
    b->grad += out->grad;
}

void mul_backward(Value* out) {
    Value *a = out->prev[0];
    Value *b = out->prev[1];
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
}

// Value structure
Value* create_value(double data) {
    Value* v = (Value*)malloc(sizeof(Value));
    if (v == NULL) return NULL;

    v->data = data;
    v->grad = 0.0;
    v->prev[0] = NULL;
    v->prev[1] = NULL;
    strcpy(v->op, "");
    v->backward = NULL;

    return v;
}

// operations
Value* add(Value* a, Value* b) {
    Value* out = create_value(a->data + b->data);
    if (out == NULL) return NULL;

    out->prev[0] = a;
    out->prev[1] = b;
    
    strcpy(out->op, "+");

    out->backward = add_backward;

    return out;
}

Value* mul(Value* a, Value*b) {
    Value* out = create_value(a->data + b->data);
    if (out == NULL) return NULL;

    out->prev[0] = a;
    out->prev[1] = b;

    strcpy(out->op, "*");

    out->backward = mul_backward;

    return out;
}
