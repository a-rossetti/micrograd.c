// nn.c
#include <stdlib.h>
#include <time.h>
#include "nn.h"
#include "value.h"

// Neuron functions
void neuron_zero_grad(Neuron *neuron) {
    for (int i = 0; i < neuron->n_inputs; i++) {
        neuron->w[i].grad = 0;
    }
    neuron->b.grad = 0;
}

void neuron_init(Neuron *neuron, int n_inputs, NeuronConfig config) {
    neuron->w = (Value *)malloc(n_inputs * sizeof(Value));
    neuron->n_inputs = n_inputs;
    neuron->config = config;

    for (int i = 0; i < n_inputs; i++) {
        neuron->w[i].data = ((double)rand() / RAND_MAX) * 2 - 1;
        neuron->w[i].grad = 0;
    }
    neuron->b.data = 0;
    neuron->b.grad = 0;
}

Value* neuron_call(Neuron *neuron, int n_inputs, Value *x) {
    Value *act = create_value(neuron->b.data);
    for (int i = 0; i < neuron->n_inputs; i++) {
        act = add(act, mul(&(neuron->w[i]), &(x[i])));
    }
    if (neuron->config.nonlin == true) {
        act = relu(act);
    }
    return act;
}

// Layer functions
void layer_zero_grad(Layer *layer) {
    for (int i = 0; i < layer->n_neurons; i++) {
        neuron_zero_grad(&layer->neurons[i]);
    }
}

void layer_init(Layer *layer, int n_inputs, int n_neurons, NeuronConfig config) {
    layer->neurons = (Neuron *)malloc(n_neurons * sizeof(Neuron));
    layer->n_neurons = n_neurons;
    for (int i = 0; i < n_neurons; i++) {
        neuron_init(&layer->neurons[i], n_inputs, config);
    }
}

// MLP functions
void mlp_zero_grad(MLP *mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
        layer_zero_grad(&mlp->layers[i]);
    }
}



