#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "logistic_regression_functions.h"
#include "matrixops.h"

int main() {
    int n = 1000;  // number of samples
    int features = 4;

    std::vector<std::vector<double>> x(n, std::vector<double>(features));  //features
    std::vector<double> y(n);

    srand(time(0));

    // Generate data
    for (int i = 0; i < n; ++i) {
        // Generate random features between 1 and 100
        for (int j = 0; j < features; ++j) {
            x[i][j] = rand() % 100 + 1;
        }

        double z = 5.0 + x[i][0] + x[i][1] + x[i][2] + x[i][3] + (rand() % 10);

        // threshold at 0.5 
        y[i] = (1.0 / (1.0 + std::exp(-z))) > 0.5 ? 1.0 : 0.0;

        if (std::isnan(x[i][0]) || std::isnan(y[i])) {
            std::cerr << "Error: NaN detected in data at index " << i << std::endl;
            return 1;
        }
    }

    std::vector<double> initial_w(features, 0.0);
    double initial_b = 0.0;

    // Hyperparameters
    double alpha = 0.0001;
    int iterations = 1000;

    // Timing
    timeval start, end;
    gettimeofday(&start, nullptr);

    // Train
    gradient_descent(x, y, initial_w, initial_b, alpha, iterations);

    gettimeofday(&end, nullptr);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_usec - start.tv_usec) / 1e6;

    std::cout << "\nFinal weights: ";
    for (const double& w : initial_w) std::cout << w << " ";
    std::cout << "\nFinal bias: " << initial_b << std::endl;
    std::cout << "Execution Time: " << time_taken << " seconds\n";

    // Make predictions
    std::vector<double> predictions;
    compute_predictions(x, initial_w, initial_b, predictions);

    // Predictions vs actual target values
    std::cout << "\nPredictions vs Actual (first 5 samples):\n";
    for (size_t i = 0; i < std::min(n, 5); ++i) {
        std::cout << "Prediction (prob): " << predictions[i]
                  << ", Label: " << y[i]
                  << ", P Class: " << (predictions[i] >= 0.5 ? 1 : 0)
                  << std::endl;
    }

    return 0;
}