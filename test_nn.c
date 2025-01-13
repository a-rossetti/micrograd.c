#include "engine.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void update_parameters(MLP *mlp, Value *learning_rate) {
    Value** params = mlp_parameters(mlp);
    int n_params = mlp_n_params(mlp);
    
    for (int i = 0; i < n_params; i++) {
        // Create a Value for the gradient update
        Value* grad_update = mul(create_value(params[i]->grad), learning_rate);
        // Subtract the update from the parameter
        Value* new_param = sub(params[i], grad_update);
        // Update the parameter while maintaining graph connections
        params[i]->data = new_param->data;
        params[i]->grad = 0.0;  // Zero out the gradient for next iteration
        
        // Clean up
        free(grad_update);
        free(new_param);
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

    double raw_inputs[][3] = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    double raw_targets[] = {1.0, -1.0, -1.0, 1.0};
    int num_samples = sizeof(raw_targets) / sizeof(raw_targets[0]);

    Value* inputs[num_samples][nin];
    Value* targets[num_samples];

    // Initialize inputs and targets
    for (int i = 0; i < num_samples; i++) {
        targets[i] = create_value(raw_targets[i]);
        for (int j = 0; j < 3; j++) {
            inputs[i][j] = create_value(raw_inputs[i][j]);
        }
    }

    Value* learning_rate = create_value(0.1);
    const int epochs = 20;

    // Training loop
    for (int epoch = 1; epoch <= epochs; epoch++) {
        // Zero gradients before forward pass
        mlp_zero_grad(&mlp);

        // Forward pass
        Value* total_loss = create_value(0.0);
        for (int i = 0; i < num_samples; i++) {
            Value **ypred = mlp_call(&mlp, inputs[i]);
            Value* diff = sub(ypred[0], targets[i]);
            Value* square = power(diff, 2.0);
            Value* new_total = add(total_loss, square);

            // Free intermediate values
            free(diff);
            free(square);
            free(total_loss);
            free(ypred[0]);
            free(ypred);

            total_loss = new_total;
        }

        // Backward pass
        backward(total_loss);

        // Debug print parameters and gradients
        Value** params = mlp_parameters(&mlp);
        int n_params = mlp_n_params(&mlp);
        printf("Epoch %d Loss: %f\n", epoch, total_loss->data);
        for (int i = 0; i < n_params; i++) {
            printf("Param %d: data=%f, grad=%f\n", i, params[i]->data, params[i]->grad);
        }
        free(params);

        // Update parameters
        update_parameters(&mlp, learning_rate);
        free(total_loss);
    }

    // Final predictions
    printf("Final predictions:\n");
    for (int i = 0; i < num_samples; i++) {
        Value **final_pred = mlp_call(&mlp, inputs[i]);
        printf("Prediction for sample %d: %s\n", i + 1, repr(final_pred[0]));

        for (int j = 0; j < mlp.layers[mlp.n_layers - 1].n_neurons; j++) {
            free(final_pred[j]);
        }
        free(final_pred);
    }

    // Clean up everything
    for (int i = 0; i < num_samples; i++) {
        free(targets[i]);
        for (int j = 0; j < nin; j++) {
            free(inputs[i][j]);
        }
    }

    free(learning_rate);
    mlp_free(&mlp);

    return 0;
}
