#include "logistic_regression_functions.h"
#include "matrixops.h"
#include <iostream>
#include <math.h>
#include <omp.h>

double sigmoid(double z) {
    if (z >= 0) {
        double exp_neg = std::exp(-z);
        return 1.0 / (1.0 + exp_neg);
    } else {
        double exp_pos = std::exp(z);
        return exp_pos / (1.0 + exp_pos);
    }
}

double compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, double& b) {
    size_t m {x.size()};
    double cost {0.0};
    
    for (size_t i {0}; i < m; ++i) {
        double ho_xi = sigmoid(dot_product(x[i].data(), w.data(), w.size()) + b);
        ho_xi = std::min(std::max(ho_xi, 1e-15), 1.0 - 1e-15);
        cost += y[i] * log(ho_xi) + (1 - y[i]) * log(1 - ho_xi);
    }

    return -cost / m;
}

void gradient_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, double& b, 
std::vector<double>& dj_dw, double& dj_db){
    size_t m {x.size()};
    size_t n {x[0].size()};

    std::fill(dj_dw.begin(), dj_dw.end(), 0.0);
    dj_db = 0;

    #pragma omp parallel for reduction(+:dj_db)
    for (size_t i = 0; i < m; ++i) {
        double error = sigmoid(dot_product(x[i].data(), w.data(), w.size()) + b) - y[i];
        
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

void gradient_descent(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& initial_w, double& initial_b, double& alpha, int& iterations) {
    size_t m {x.size()};
    // std::vector<double> cost_hist;

    for (size_t i {0}; i < iterations; ++i){
        std::vector<double>dj_dw(initial_w.size(), 0.0);
        double dj_db {0.0};

        gradient_cost(x, y, initial_w, initial_b, dj_dw, dj_db);

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

void compute_predictions(std::vector<std::vector<double>>& x, std::vector<double>& w, 
double& b, std::vector<double>& predictions) {
    size_t m {x.size()};
    predictions.resize(m);

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i){
        predictions[i] = sigmoid(dot_product(x[i].data(), w.data(), w.size()) + b);
    }

}