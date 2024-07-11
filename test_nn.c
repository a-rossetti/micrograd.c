#include "engine.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Mean Squared Error
Value* mse(Value** outputs, Value** targets, int count) {
    Value* mse = create_value(0.0);
    Value* v_count = create_value(count);
    for (int i = 0; i < count; i++) {
        Value* diff = sub(outputs[i], targets[i]);
        Value* squared = power(diff, 2.0);
        mse = add(mse, squared);
    }
    mse = truediv(mse, v_count);
    return mse;
}

int main() {
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

    Value** ypred = malloc(num_samples * sizeof(Value*));

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        Value* loss = create_value(0.0);
        for (int i = 0; i < num_samples; i++) {
            ypred = mlp_call(&mlp, inputs[i], nin);
            for (int j = 0; j < mlp.layers[mlp.n_layers - 1].n_neurons; j++) {
                loss = add(loss, mse(&ypred[j], &targets[i], 1)); 
            }
            free(ypred);
        }

        // Backward pass
        mlp_zero_grad(&mlp);
        backward(loss);

        // Update
        Value** params = mlp_parameters(&mlp);
        int n_params = mlp_n_params(&mlp);
        for (int i = 0; i < n_params; i++) {
            params[i]->data = sub(params[i], mul(learning_rate, create_value(params[i]->grad)))->data;
        }
        
        printf("Epoch %d: Loss = %f\n", epoch, loss->data);
    }

    printf("Final predictions:\n");
    for (int i = 0; i < num_samples; i++) {
        printf("Prediction for sample %d: %s\n", i + 1, repr(ypred[i]));
    }

    return 0;
}
