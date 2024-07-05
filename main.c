#include <stdio.h>
#include <math.h>
#include "engine.h"

void test_add() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);
    Value* c = add(a, b);
    printf("add: %f (expected 5.0)\n", c->data);
    backward(c);
    printf("a.grad: %f (expected 1.0)\n", a->grad);
    printf("b.grad: %f (expected 1.0)\n", b->grad);
}

void test_mul() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);
    Value* c = mul(a, b);
    printf("multiply: %f (expected 6.0)\n", c->data);
    backward(c);
    printf("a.grad: %f (expected 3.0)\n", a->grad);
    printf("b.grad: %f (expected 2.0)\n", b->grad);
}

void test_pow() {
    Value* a = create_value(2.0);
    Value* c = power(a, 3.0);
    printf("power: %f (expected 8.0)\n", c->data);
    backward(c);
    printf("a.grad: %f (expected 12.0)\n", a->grad); // 3 * 2^2
}

void test_relu() {
    Value* a = create_value(-1.0);
    Value* b = create_value(2.0);
    Value* c = relu(a);
    Value* d = relu(b);
    printf("relu(-1.0): %f (expected 0.0)\n", c->data);
    printf("relu(2.0): %f (expected 2.0)\n", d->data);
    backward(d);
    printf("b.grad: %f (expected 1.0)\n", b->grad);
}

void test_combined() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);
    Value* c = create_value(4.0);
    Value* d = add(a, b); // d = a + b
    Value* e = mul(d, c); // e = d * c
    Value* f = relu(e); // f = relu(e)
    printf("combined: %f (expected 20.0)\n", f->data); // (2 + 3) * 4 = 20
    backward(f);
    printf("a.grad: %f\n", a->grad);
    printf("b.grad: %f\n", b->grad);
    printf("c.grad: %f\n", c->grad);
}

int main() {
    printf("Testing add function:\n");
    test_add();

    printf("\nTesting multiply function:\n");
    test_mul();

    printf("\nTesting power function:\n");
    test_pow();

    printf("\nTesting relu function:\n");
    test_relu();

    printf("\nTesting combined operations:\n");
    test_combined();

    return 0;
}

