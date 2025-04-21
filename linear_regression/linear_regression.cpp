#include "linear_regression.h"


Lr::Lr(std::vector<double> x, std::vector<double> w) : x(x), w(w) {}

void Lr::test(double x, double w) {
    std::cout << x << w << std::endl;
}
