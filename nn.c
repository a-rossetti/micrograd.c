// nn.c
#include <stdlib.h>
#include <time.h>
#include "nn.h"
#include "engine.h"

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

Value* neuron_call(Neuron *neuron, Value *x) {
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

Value** layer_call(Layer *layer, Value **x) {
    Value **out = malloc(layer->n_neurons * sizeof(Value*));
    if (out == NULL) return NULL;

    for (int i = 0; i < layer->n_neurons; i++) {
        out[i] = neuron_call(&layer->neurons[i], x[i]);
    }
    return out;
}

// MLP functions
void mlp_zero_grad(MLP *mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
        layer_zero_grad(&mlp->layers[i]);
    }
}

void mlp_init(MLP *mlp, int nin, int *nouts, int nouts_len) {
    int *sizes = malloc((nouts_len + 1) * sizeof(int));
    sizes[0] = nin;

    for (int i = 0; i < nouts_len; i++) {
        sizes[i+1] = nouts[i];
    }
    mlp->layers = malloc(nouts_len * sizeof(Layer));
    mlp->n_layers = nouts_len;
    
    for (int i = 0; i < nouts_len; i++) {
        NeuronConfig config = {.nonlin = (i != nouts_len - 1)};
        layer_init(&mlp->layers[i], sizes[i], sizes[i+1], config);
    }

    free(sizes);
}

Value** mlp_call(MLP *mlp, Value **x, int n_inputs) {
    Value **current_x = x;
    int current_n = n_inputs;

    for (int i = 0; i < mlp->n_layers; i++) {
        Value **layer_out = layer_call(&mlp->layers[i], current_x);

        if (i > 0) {
            for (int j = 0; j < current_n; j++) {
                free(x[j]);
            }
            free(x);
        }

        x = layer_out;
        current_n = mlp->layers[i].n_neurons;
    }
    return current_x;
}

