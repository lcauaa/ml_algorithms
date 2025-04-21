#include <iostream>
#include <cblas.h>
#include <vector>
#include <cmath>

#include "linear_regression.h"

int main() {
    const int N {5};
    double A[N] {1.0, 2.0, 3.0, 4.0, 5.0};
    double B[N] {5.0, 4.0, 3.0, 2.0, 1.0};

    double result = cblas_ddot(N, A, 1, B, 1);

    std::cout << result << std::endl;
    return 0;
}