#include "functions.h"
#include <iostream>
#include <cblas-openblas.h>
#include <vector>
#include <cmath>
#include "matrixops.h"
#include <cstdlib> 
#include <ctime>   

// extern "C" {
//     void openblas_set_num_threads(int);
// }
 
int main() {
    // openblas_set_num_threads(2);
    int n = 1000;  // number of samples

    std::vector<std::vector<double>> x(n, std::vector<double>(4));  // features
    std::vector<double> y(n);

    // Seed the random number generator
    srand(time(0));

    // Generate data
    for (int i = 0; i < n; ++i) { // Random features
        x[i][0] = rand() % 100 + 1;  
        x[i][1] = rand() % 100 + 1;  
        x[i][2] = rand() % 100 + 1;  
        x[i][3] = rand() % 100 + 1;  
        y[i] = 5.0 + x[i][0] + x[i][1] + x[i][2] + x[i][3] + (rand() % 10);  // y = 5 + feature1 + feature2 + ... + noise

        // Check for invalid (NaN) values in the generated data
        if (std::isnan(x[i][0]) || std::isnan(x[i][1]) || std::isnan(y[i])) {
            std::cerr << "Error: NaN detected in generated data at index " << i << std::endl;
            return 1; 
        }
    }

    std::vector<double> initial_w(x[0].size(), 0.0);  
    double initial_b = 0.0;  

    // Hyperparameters
    double alpha = 0.0001; 
    int iterations = 1000;  

    // Measure time before gradient descent
    clock_t start_time = clock();

    gradient_descent(x, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations);

    // Measure time after gradient descent
    clock_t end_time = clock();
    double time_taken = double(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Final weights: ";
    for (const double& w : initial_w) {
        std::cout << w << " ";
    }
    std::cout << "\nFinal bias: " << initial_b << std::endl;

    std::cout << "Execution Time: " << time_taken << " seconds" << std::endl;

    // Make predictions
    std::vector<double> predictions;
    compute_predictions(x, initial_w, initial_b, predictions);

    // Predictions vs actual target values
    std::cout << "\nPredictions vs Actual values (first 5):\n";
    for (size_t i = 0; i < std::min(x.size(), 5UL); ++i) {
        std::cout << "Prediction: " << predictions[i] << ", Target value: " << y[i] << std::endl;
    }

    return 0;
}