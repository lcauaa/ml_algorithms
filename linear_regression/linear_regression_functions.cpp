#include "linear_regression_functions.h"
#include "matrixops.h"
#include <iostream>
#include <cblas-openblas.h>
#include <omp.h>
#include <cmath>

double compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, double b) {
    size_t m = x.size();  
    double cost = 0.0;

    // Parallelize the cost calculation
    #pragma omp parallel for reduction(+:cost)
    for (size_t i = 0; i < m; ++i) {
        double f_wb_i = dot_product(x[i].data(), w.data(), w.size()) + b;  
        double error = f_wb_i - y[i];
        cost += error * error;
    }

    cost /= (2 * m);
    
    return cost;
}

// Do per thread reduction later
void compute_gradient(std::vector<std::vector<double>>& x, std::vector<double>& y,
    std::vector<double>& w, double b,
    std::vector<double>& dj_dw, double& dj_db) {
    size_t m = x.size();  
    size_t n = x[0].size();  

    std::fill(dj_dw.begin(), dj_dw.end(), 0.0);
    dj_db = 0.0;

    // Parallelize the gradient calculation
    #pragma omp parallel for reduction(+:dj_db)
    for (size_t i = 0; i < m; ++i) {
        double error = dot_product(x[i].data(), w.data(), w.size()) + b - y[i];  // error = f(w, b) - y

        for (size_t j = 0; j < n; ++j) {
            #pragma omp atomic
            dj_dw[j] += error * x[i][j];
        }

        dj_db += error;
    }

    for (size_t j = 0; j < n; ++j) {
        dj_dw[j] /= m;
    }

    dj_db /= m;
}

void gradient_descent(std::vector<std::vector<double>>& x, std::vector<double>& y, 
    std::vector<double>& initial_w, double& initial_b, 
    double& alpha, 
    int& iterations) {
    size_t m = x.size();
    // std::vector<double> cost_hist;

    for (int i = 0; i < iterations; ++i) {
        std::vector<double> dj_dw(initial_w.size(), 0.0);
        double dj_db = 0.0;

        compute_gradient(x, y, initial_w, initial_b, dj_dw, dj_db);

        // Update weights
        #pragma omp parallel for
        for (size_t j = 0; j < initial_w.size(); ++j) {
            initial_w[j] -= alpha * dj_dw[j];
        }

        initial_b -= alpha * dj_db;

        // Store cost for visualization 
        // if (i % 100 == 0) {
        // double cost = compute_cost(x, y, initial_w, initial_b);
        // cost_hist.push_back(cost);
        // std::cout << "Iteration " << i << ", Cost: " << cost << std::endl;
        // }
    }
}

void compute_predictions(std::vector<std::vector<double>>& x, 
    std::vector<double>& w, 
    double b, std::vector<double>& predictions) {

    size_t m = x.size();
    predictions.resize(m);

    // Parallelize prediction computation
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        predictions[i] = dot_product(x[i].data(), w.data(), w.size()) + b;
    }
}