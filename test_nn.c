#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

// Test MLP initialization and parameter count
void test_mlp_init() {
    MLP mlp;
    int nouts[] = {4, 4, 1};  // 3-layer network: 4 hidden, 4 hidden, 1 output
    mlp_init(&mlp, 3, nouts, 3);
    
    int expected_params = (3*4 + 4) + (4*4 + 4) + (4*1 + 1);  // (inputs*hidden + hidden) + (hidden*hidden + hidden) + (hidden*output + output)
    int actual_params = mlp_n_params(&mlp);
    printf("MLP Init Test:\n");
    printf("  Expected params: %d\n", expected_params);
    printf("  Actual params:   %d (%s)\n\n", actual_params, 
          actual_params == expected_params ? "PASS" : "FAIL");
    
    mlp_free(&mlp);
}

// Test forward pass dimensions
void test_forward_pass() {
    MLP mlp;
    int nouts[] = {4, 4, 1};  // 3-layer network: 4 hidden, 4 hidden, 1 output
    mlp_init(&mlp, 3, nouts, 3);
    
    // Create input Values
    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}
    };
    
    Value** outputs[4];
    for (int i = 0; i < 4; i++) {
        outputs[i] = mlp_call(&mlp, inputs[i]);
        printf("Forward Pass Test %d:\n", i+1);
        printf("  Expected output size: 1\n");
        printf("  Actual output size:   %d (%s)\n\n", mlp.layers[2].n_neurons, 
              mlp.layers[2].n_neurons == 1 ? "PASS" : "FAIL");
    }
    
    // Cleanup
    for (int i = 0; i < 4; i++) {
        free(outputs[i]);
        for (int j = 0; j < 3; j++) {
            free(inputs[i][j]);
        }
    }
    mlp_free(&mlp);
}

// Test gradient computation
void test_backward() {
    MLP mlp;
    int nouts[] = {4, 4, 1};
    mlp_init(&mlp, 3, nouts, 3);
    
    // Create input and target
    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}
    };
    Value* targets[4] = {create_value(1.0), create_value(-1.0), create_value(-1.0), create_value(1.0)};
    
    // Forward pass
    Value** outputs[4];
    Value* losses[4];
    for (int i = 0; i < 4; i++) {
        outputs[i] = mlp_call(&mlp, inputs[i]);
        losses[i] = power(sub(outputs[i][0], targets[i]), 2.0);
    }
    
    // Backward pass
    mlp_zero_grad(&mlp);
    for (int i = 0; i < 4; i++) {
        backward(losses[i]);
    }
    
    // Check gradients
    Value** params = mlp_parameters(&mlp);
    int all_zero = 1;
    for(int i=0; i<mlp_n_params(&mlp); i++) {
        if(params[i]->grad != 0.0) {
            all_zero = 0;
            break;
        }
    }
    
    printf("Backward Pass Test:\n");
    printf("  All params have non-zero grad: %s (%s)\n\n", 
          all_zero ? "NO" : "YES", 
          all_zero ? "FAIL" : "PASS");
    
    // Cleanup
    free(params);
    for (int i = 0; i < 4; i++) {
        free(outputs[i]);
        free(losses[i]);
        for (int j = 0; j < 3; j++) {
            free(inputs[i][j]);
        }
        free(targets[i]);
    }
    mlp_free(&mlp);
}

// Simple training test
void test_training() {
    MLP mlp;
    int nouts[] = {4, 4, 1};  // 3-layer network
    mlp_init(&mlp, 3, nouts, 3);
    
    // Inputs and targets
    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}
    };
    Value* targets[4] = {create_value(1.0), create_value(-1.0), create_value(-1.0), create_value(1.0)};

    // Get all parameters once (outside the loop)
    Value** params = mlp_parameters(&mlp);
    int n_params = mlp_n_params(&mlp);

    // Adam state variables
    double* m = calloc(n_params, sizeof(double));  // 1st moment
    double* v = calloc(n_params, sizeof(double));  // 2nd moment
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double lr = 0.01;  // Adam typically uses smaller learning rates

    // Training loop
    float total_losses[200];
    for(int epoch=0; epoch<200; epoch++) {
        mlp_zero_grad(&mlp);
        
        // Forward pass and accumulate loss
        Value* total_loss = create_value(0.0);
        Value* losses[4];
        for (int i = 0; i < 4; i++) {
            Value** output = mlp_call(&mlp, inputs[i]);
            losses[i] = power(sub(output[0], targets[i]), 2.0);  // L2 loss
            total_loss = add(total_loss, losses[i]);
            free(output);  // Free output array (not Values)
        }
        
        // Compute mean loss
        Value* divisor = create_value(4.0);  // Batch size = 4
        Value* avg_loss = truediv(total_loss, divisor);
        
        // Backward pass
        backward(avg_loss);
        
        // Update weights (SGD)
        for(int i=0; i<mlp_n_params(&mlp); i++) {
            // Update moments
            m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
            v[i] = beta2 * v[i] + (1 - beta2) * pow(params[i]->grad, 2);

            // Bias correction
            double m_hat = m[i] / (1 - pow(beta1, epoch + 1));
            double v_hat = v[i] / (1 - pow(beta2, epoch + 1));

            // Update parameter
            params[i]->data -= lr * m_hat / (sqrt(v_hat) + eps);
        }

        // Store and print loss
        total_losses[epoch] = avg_loss->data;
        printf("Training Step %d - Loss: %.8f\n", epoch+1, avg_loss->data);
        
        // Cleanup
        free(avg_loss);
        free(divisor);
        free(total_loss);
        for (int i = 0; i < 4; i++) free(losses[i]);
    }

    // Cleanup
    free(m);
    free(v);
    free(params);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) free(inputs[i][j]);
        free(targets[i]);
    }
    mlp_free(&mlp);
}

int main() {
    srand(time(NULL));
    
    test_mlp_init();
    test_forward_pass();
    test_backward();
    test_training();
    
    return 0;
}

