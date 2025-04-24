#include "functions.h"
#include "matrixops.h"
#include <iostream>
#include <cblas-openblas.h>
#include <omp.h>
#include <cmath>

double compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y,
    std::vector<double>& w, double b) {
    int m = x.size(); 
    double cost {0.0}; 

    // Parallelize the cost calculation over all examples
    #pragma omp parallel for reduction(+:cost)
    for (size_t i = 0; i < m; ++i) {
        double f_wb_i = dot_product(x[i].data(), w.data(), w.size()) + b; 

        double err = f_wb_i - y[i];
            cost += err * err;
    }

    cost /= (2 * m);

    return cost;
}

// Do per thread reduction later maybe
void compute_gradient(std::vector<std::vector<double>>& x, std::vector<double>& y,
    std::vector<double>& w, double b,
    std::vector<double>& dj_dw, double& dj_db) {

    size_t m = x.size(); 
    size_t n = x[0].size();  

    std::fill(dj_dw.begin(), dj_dw.end(), 0.0);
    dj_db = 0.0;

    // Parallelize the gradient calculation over the m data points
    #pragma omp parallel for reduction(+:dj_db)
    for (size_t i = 0; i < m; ++i) {
        double err = dot_product(x[i].data(), w.data(), n) + b - y[i];

        for (size_t j = 0; j < n; ++j) {
            #pragma omp atomic // atomic to prevent race conditions
            dj_dw[j] += err * x[i][j];  
        }

        dj_db += err;
    }

    for (size_t j = 0; j < n; ++j) {
        dj_dw[j] /= m;
    }
    
    dj_db /= m;
}

void gradient_descent(std::vector<std::vector<double>>& x, std::vector<double>& y, 
    std::vector<double>& initial_w, double& initial_b, 
    ComputeCostPtr cost_fn, ComputeGradientPtr gradient_fn, double& alpha, // (test) remove compute_cost later to improve performance
    int& iterations) {
    
    std::vector<double> cost_hist;
    std::fill(initial_w.begin(), initial_w.end(), 0.0);  
    initial_b = 0.0;  

    std::vector<double> dj_dw(initial_w.size(), 0.0);  
    double dj_db = 0.0; 

    for (size_t i = 0; i < iterations; ++i) {
        // Reset gradients to zero
        std::fill(dj_dw.begin(), dj_dw.end(), 0.0);
        dj_db = 0.0;

        gradient_fn(x, y, initial_w, initial_b, dj_dw, dj_db);

        // Compute cost and store, remove later maybe
        if (i % 100 == 0) { 
            double cost = cost_fn(x, y, initial_w, initial_b);
            cost_hist.push_back(cost);
            std::cout << "Iteration " << i << ", Cost: " << cost << std::endl;
        }

        // Parallelized update of weights
        #pragma omp parallel for
        for (size_t j = 0; j < initial_w.size(); ++j) {
            initial_w[j] -= alpha * dj_dw[j];
        }
        
        initial_b -= alpha * dj_db;
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