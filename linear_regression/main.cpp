#include "functions.h"
#include <iostream>
#include <cblas-openblas.h>
#include <vector>
#include <cmath>

extern "C" {
    void openblas_set_num_threads(int);
}

int main() {
    openblas_set_num_threads(2);
    std::vector<std::vector<double>> x {
        {3.0, 2.0},
        {2.0, 3.0},
        {5.0, 4.0}
    };

    std::vector<double> y = {5.0, 8.0, 11.0};
    std::vector<double> w = {1.0, 2.0};
    int b {1};
    std::vector<double> dj_dw;
    double dj_db;

    double cost {0};
    compute_cost(x, y, w, b, cost);
    compute_gradient(x, y, w, b, dj_dw, dj_db);

    std::cout << "Cost: " << cost << std::endl;
    for (double dw : dj_dw) {
        std::cout << "dj_dw: " << dw << " ";
    }
    std::cout << "dj_db: " << dj_db << std::endl;

    return 0;
}