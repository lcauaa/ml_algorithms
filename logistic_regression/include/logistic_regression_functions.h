#ifndef LOGISTIC_REGRESSION_FUNCTIONS_H
#define LOGISTIC_REGRESSION_FUNCTIONS_H
#include <vector>

double sigmoid(double z);

double compute_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, double& b);

void gradient_cost(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& w, double& b, 
std::vector<double>& dj_dw, double& dj_db);

void gradient_descent(std::vector<std::vector<double>>& x, std::vector<double>& y, std::vector<double>& initial_w, double& initial_b, double& alpha, int& iterations);

void compute_predictions(std::vector<std::vector<double>>& x, std::vector<double>& w, double& b, std::vector<double>& predictions);

#endif