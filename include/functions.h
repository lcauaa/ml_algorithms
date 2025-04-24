#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <cblas-openblas.h>
#include <vector>

// functions pointer type
typedef double (*ComputeCostPtr) (
    std::vector<std::vector<double>>&,
    std::vector<double>&,
    std::vector<double>&,
    double
);

typedef void (*ComputeGradientPtr) (
    std::vector<std::vector<double>>&, std::vector<double>&,
    std::vector<double>&, 
    double,
    std::vector<double>&, 
    double&
);

double compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, double b);

void compute_gradient(std::vector<std::vector<double>>& x, std::vector<double>& y,
    std::vector<double>& w, double b,
    std::vector<double>& dj_dw, double& dj_db);


void gradient_descent(std::vector<std::vector<double>>& x, std::vector<double>& y, 
    std::vector<double>& initial_w, double& initial_b, 
    ComputeCostPtr cost_fn, ComputeGradientPtr gradient_fn, double& alpha, 
    int& iterations);

void compute_predictions(std::vector<std::vector<double>>& x, 
    std::vector<double>& w, 
    double b, std::vector<double>& predictions);

#endif