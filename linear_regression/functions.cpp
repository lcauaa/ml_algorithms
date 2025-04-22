#include "functions.h"
#include <cblas-openblas.h>
#include <omp.h>
#include <cmath>

void compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, 
    std::vector<double>& w, int b, double& cost) {
    size_t m = x.size();    // training examples
    size_t n = w.size();    // number of features
    cost = 0.0;

    // Flatten x into a 1D array for better BLAS compatibility
    std::vector<double> x_flat(m * n);
    for (size_t i = 0; i < m; ++i) {
        if (x[i].size() != n) {
            // Handle error: inconsistent feature dimensions
            return; // Or throw an exception
        }
        std::copy(x[i].begin(), x[i].end(), x_flat.begin() + i * n);
    }

    // result vector for predictions
    std::vector<double> y_hat(m, 0.0);

    // matrix multiplication y_hat = alpha * x * w + beta * y_hat
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
    static_cast<int>(m), static_cast<int>(n),
    1.0,
    x_flat.data(), static_cast<int>(n),
    w.data(), 1,
    0.0,
    y_hat.data(), 1);

    // add bias to each prediction and compute parallelized mean squared error
    #pragma omp parallel for reduction(+:cost)
    for (size_t i = 0; i < m; ++i) {
        double error = y_hat[i] + b - y[i];
        cost += error * error;
    }

    cost /= (2.0 * m); 
}

void compute_gradient(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, int b, std::vector<double>& dj_dw, double& dj_db) {
    size_t m {x.size()};
    size_t n {x[0].size()};

    dj_dw = std::vector<double>(n, 0.0);
    dj_db = 0.0;

    // Flatten x into a 1D array for better BLAS compatibility
    std::vector<double> x_flat(m * n);
    for (size_t i = 0; i < m; ++i) {
        if (x[i].size() != n) {
            // Handle error: inconsistent feature dimensions
            return; // Or throw an exception
        }
        std::copy(x[i].begin(), x[i].end(), x_flat.begin() + i * n);
    }

    std::vector<double> y_hat(m);

    // matrix multiplication y_hat = alpha * x * w + beta * y_hat
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
        static_cast<int>(m), static_cast<int>(n),
        1.0,
        x_flat.data(), static_cast<int>(n),
        w.data(), 1,
        0.0,
        y_hat.data(), 1);

        
    for (size_t i = 0; i < m; ++i) {
        double error = y_hat[i] + b - y[i];
        for (size_t j = 0; j < n; ++j) {
            dj_dw[j] += error * x[i][j];
        }
        dj_db = dj_db + error;
    }

    for (size_t j {0}; j < n; ++j){
        dj_dw[j] /= m;
    }
    dj_db /= m;
}
