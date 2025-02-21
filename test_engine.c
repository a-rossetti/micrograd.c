#include <stdio.h>
#include "engine.h"

void test_repr() {
    Value *a = create_value(2.5);
    a->grad = 1.0;
    char* repr_str = repr(a);
    printf("repr: %s (expected 'Value(data=2.500000, grad=1.000000)')\n", repr_str);
}

void test_add() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);
    Value* c = add(a, b);
    printf("add: %.1f (expected 5.0)\n", c->data);
    backward(c);
    printf("a.grad: %.1f (expected 1.0)\n", a->grad);
    printf("b.grad: %.1f (expected 1.0)\n", b->grad);
}

void test_mul() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);
    Value* c = mul(a, b);
    printf("multiply: %.1f (expected 6.0)\n", c->data);
    backward(c);
    printf("a.grad: %.1f (expected 3.0)\n", a->grad);
    printf("b.grad: %.1f (expected 2.0)\n", b->grad);
}

void test_pow() {
    Value* a = create_value(2.0);
    Value* c = power(a, 3.0);
    printf("power: %.1f (expected 8.0)\n", c->data);
    backward(c);
    printf("a.grad: %.1f (expected 12.0)\n", a->grad); // 3 * 2^2
}

void test_relu() {
    Value* a = create_value(-1.0);
    Value* b = create_value(2.0);
    Value* c = relu(a);
    Value* d = relu(b);
    printf("relu(-1.0): %.1f (expected 0.0)\n", c->data);
    printf("relu(2.0): %.1f (expected 2.0)\n", d->data);
    backward(d);
    printf("b.grad: %.1f (expected 1.0)\n", b->grad);
}

void test_combined() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);
    Value* c = create_value(4.0);
    Value* d = add(a, b); // d = a + b
    Value* e = mul(d, c); // e = d * c
    Value* f = relu(e); // f = relu(e)
    printf("combined: %.1f (expected 20.0)\n", f->data); // (2 + 3) * 4 = 20
    backward(f);
    printf("a.grad: %.1f (expected 4.0)\n", a->grad);
    printf("b.grad: %.1f (expected 4.0)\n", b->grad);
    printf("c.grad: %.1f (expected 5.0)\n", c->grad);
}

void test_loss() {
    Value* inputs[2] = {create_value(0.5), create_value(1.0)};
    Value* targets[2] = {create_value(2.0), create_value(3.0)};
    Value* total_loss = create_value(0.0);
    Value* diffs[2];
    Value* squares[2];
    Value* temp_losses[2];
    printf("expected grads: -3.00, -4.00.\ngrads:\n");
    for (int i = 0; i < 2; i++) {
        diffs[i] = sub(targets[i], inputs[i]);
        squares[i] = power(diffs[i], 2.0);
        temp_losses[i] = create_value(total_loss->data);
        total_loss = add(temp_losses[i], squares[i]);
        backward(total_loss);
        printf("  inputs[%d].grad: %.2f\n", i, inputs[i]->grad);
    }
    printf("combined: %.2f (expected 6.25)\n", total_loss->data);
}

int main() {
    printf("Testing repr function:\n");
    test_repr();

    printf("\nTesting add function:\n");
    test_add();

    printf("\nTesting multiply function:\n");
    test_mul();

    printf("\nTesting power function:\n");
    test_pow();

    printf("\nTesting relu function:\n");
    test_relu();

    printf("\nTesting combined operations:\n");
    test_combined();

    printf("\nTesting loss:\n");
    test_loss();

    return 0;
}

