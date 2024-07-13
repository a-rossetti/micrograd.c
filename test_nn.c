#include "engine.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void update_parameters(MLP *mlp, Value *learning_rate) {
    Value** params = mlp_parameters(mlp);
    int n_params = mlp_n_params(mlp);
    for (int i = 0; i < n_params; i++) {
        params[i]->data -= learning_rate->data * params[i]->grad;
    }
    free(params);
}

int main() {
    srand(time(NULL));

    int nin = 3;
    int nouts[] = {4, 4, 1};
    int nouts_len = sizeof(nouts) / sizeof(nouts[0]);

    MLP mlp;
    mlp_init(&mlp, nin, nouts, nouts_len);

    double raw_inputs[][3] = {{2.0, 3.0, -1.0},
                          {3.0, -1.0, 0.5},
                          {0.5, 1.0, 1.0},
                          {1.0, 1.0, -1.0}};
    double raw_targets[] = {1.0, -1.0, -1.0, 1.0};
    int num_samples = sizeof(raw_targets) / sizeof(raw_targets[0]);

    Value* inputs[num_samples][nin];
    Value* targets[num_samples];

    for (int i = 0; i < 4; i++) {
        targets[i] = create_value(raw_targets[i]);
        for (int j = 0; j < 3; j++) {
            inputs[i][j] = create_value(raw_inputs[i][j]);
        }
    }

    Value* learning_rate = create_value(0.1);
    const int epochs = 20;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        mlp_zero_grad(&mlp);

        // Forward pass
        Value* total_loss = create_value(0.0);
        MLPOutput** outputs = malloc(num_samples * sizeof(MLPOutput*));

        for (int i = 0; i < num_samples; i++) {
            outputs[i] = mlp_call(&mlp, inputs[i]);
            Value* ypred = outputs[i]->layer_outputs[mlp.n_layers - 1][0];
            Value* diff = sub(ypred, targets[i]);
            Value* squared = power(diff, 2.0);
            Value* sample_loss = truediv(squared, create_value((double)num_samples));
            Value* new_total_loss = add(total_loss, sample_loss);
            free(total_loss);
            total_loss = new_total_loss;

            free(diff);
            free(squared);
            free(sample_loss);
        }

        // Backward pass
        backward(total_loss);

        printf("Epoch %d: Loss: %f\n", epoch, total_loss->data);

        // Update parameters
        update_parameters(&mlp, learning_rate);

        // Free memory
        for (int i = 0; i < num_samples; i++) {
            for (int j = 0; j < outputs[i]->n_layers; j++) {
                for (int k = 0; k < mlp.layers[j].n_neurons; k++) {
                    free(outputs[i]->layer_outputs[j][k]);
                }
                free(outputs[i]->layer_outputs[j]);
            }
            free(outputs[i]->layer_outputs);
            free(outputs[i]);
        }
        free(outputs);
        free(total_loss);
    }

    printf("Final predictions:\n");
    for (int i = 0; i < num_samples; i++) {
        MLPOutput* output = mlp_call(&mlp, inputs[i]);
        Value* final_pred = output->layer_outputs[mlp.n_layers - 1][0];
        printf("Prediction for sample %d: %s\n", i + 1, repr(final_pred));

        // Free the output structure
        for (int j = 0; j < output->n_layers; j++) {
            for (int k = 0; k < mlp.layers[j].n_neurons; k++) {
                free(output->layer_outputs[j][k]);
            }
            free(output->layer_outputs[j]);
        }
        free(output->layer_outputs);
        free(output);
    }

    return 0;
}
