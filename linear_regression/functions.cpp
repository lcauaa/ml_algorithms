#include "functions.h"
#include <cblas.h>
#include <omp.h>
#include <cmath>

double compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, int b) {
    size_t m {x.size()}; // training examples
    size_t n {w.size()}; // number of features
    double cost {0.0};

    // result vector xw + b
    std::vector<double> y_hat(m);
    
    /*
    // Flatten x into a 1D array for better memory layout and BLAS compatibility
    std::vector<double> x_flat(m * n);
    for (size_t i = 0; i < m; ++i) {
        std::copy(x[i].begin(), x[i].end(), x_flat.begin() + i * n);
    }
    */

    // matrix multiplication y_hat = alpha * a * x + beta * y
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
    m, n,
    1.0,
    reinterpret_cast<double*>(x.data()), n,
    w.data(), 1,
    0.0,
    y_hat.data() ,1
    );

    // add bias to each prediction and compute parallelized mean sqared error 
    #pragma omp parallel for reduction(+:cost)
    for (size_t i {0}; i < m; ++i){
        double error = y_hat[i] + b - y[i];
        cost += error * error;
    }
    
    return cost / (2.0 * m);
}