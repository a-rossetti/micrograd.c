#include "engine.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void update_parameters(MLP *mlp, Value *learning_rate) {
    Value** params = mlp_parameters(mlp);
    int n_params = mlp_n_params(mlp);
    //printf("%d\n", n_params);
    for (int i = 0; i < n_params; i++) {
        //char* repri = repr(params[i]);
        //printf("%s\n", repri);
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
        mlp_zero_grad(&mlp);

        Value* total_loss = create_value(0.0);
        Value* diffs[num_samples];
        Value* squares[num_samples];
        Value* temp_losses[num_samples];
        for (int i = 0; i < num_samples; i++) {
            Value **ypred = mlp_call(&mlp, inputs[i]);
            diffs[i] = sub(ypred[0], targets[i]);
            squares[i] = power(diffs[i], 2.0);
            temp_losses[i] = create_value(total_loss->data);
            total_loss = add(temp_losses[i], squares[i]);
            backward(total_loss);
        }

        printf("Epoch %d: Loss: %f\n", epoch, total_loss->data);

        update_parameters(&mlp, learning_rate);

        free(total_loss);
    }

    printf("Final predictions:\n");
    for (int i = 0; i < num_samples; i++) {
        Value **final_pred = mlp_call(&mlp, inputs[i]);
        printf("Prediction for sample %d: %s\n", i + 1, repr(final_pred[0]));

        for (int j = 0; j < mlp.layers[mlp.n_layers - 1].n_neurons; j++) {
            free(final_pred[j]);
        }
        free(final_pred);
    }

    return 0;
}
