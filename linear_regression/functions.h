#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <cblas-openblas.h>
#include <vector>

void compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, int b, double& cost);

void compute_gradient(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, int b, std::vector<double>& dj_dw, double& dj_db);

#endif