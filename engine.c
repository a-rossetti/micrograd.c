// engine.c
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"

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

void power_backward(Value* out) {
    Value *a = out->prev[0];
    double b = atof(out->op + 1);  // Extract exponent from the op string
    a->grad += (b * pow(a->data, b - 1)) * out->grad;
}

void relu_backward(Value* out) {
    Value *a = out->prev[0];
    if (out->data > 0) 
        a->grad += out->grad;
}

// Value structure
Value* create_value(double data) {
    Value* v = (Value*)malloc(sizeof(Value));
    if (v == NULL) return NULL;

    v->data = data;
    v->grad = 0.0;
    v->prev[0] = NULL;
    v->prev[1] = NULL;
    v->op[0] = '\0';
    v->backward = NULL;

    return v;
}

char* repr(Value* v) {
    static char vrepr[100];
    snprintf(vrepr, sizeof(vrepr), "Value(data=%f, grad=%f)", v->data, v->grad);
    return vrepr;
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
    Value* out = create_value(a->data * b->data);
    if (out == NULL) return NULL;

    out->prev[0] = a;
    out->prev[1] = b;

    strcpy(out->op, "*");

    out->backward = mul_backward;

    return out;
}

Value* power(Value* a, double b) {
    Value* out = create_value(pow(a->data, b));
    if (out == NULL) return NULL;

    out->prev[0] = a;
    out->prev[1] = NULL;
    
    snprintf(out->op, sizeof(out->op), "^%f", b);

    out->backward = power_backward;

    return out;
}

Value* relu(Value* a) {
    Value* out = create_value(a->data < 0 ? 0 : a->data);
    if (out == NULL) return NULL;

    out->prev[0] = a;
    out->prev[1] = NULL;

    strcpy(out->op, "ReLU");

    out->backward = relu_backward;

    return out;
}

Value* neg(Value* a) {
    return mul(a, create_value(-1));
}

Value* sub(Value* a, Value* b) { 
    return add(a, neg(b));
}

Value* truediv(Value* a, Value* b) {
    return mul(a, power(b, -1));
}

// Backward function
void build_topo(Value* v, Value** topo, int* topo_size, Value** visited, int* visited_size) {
    for (int i = 0; i < *visited_size; i++) {
        if (visited[i] == v) return;
    }
    visited[(*visited_size)++] = v;

    for (int i = 0; i < 2; i++) {
        if (v->prev[i] != NULL) {
            build_topo(v->prev[i], topo, topo_size, visited, visited_size);
        }
    }
    topo[(*topo_size)++] = v;
}

void backward(Value* v) {
    Value* topo[1000];
    int topo_size = 0;
    Value* visited[1000];
    int visited_size = 0;

    build_topo(v, topo, &topo_size, visited, &visited_size);

    v->grad = 1.0;
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward != NULL) {
            topo[i]->backward(topo[i]);
        }
    }
}

