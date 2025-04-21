#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <iostream>
#include <vector>

class Lr {
    public:
        Lr() = default;
        Lr(std::vector<double> x, std::vector<double> w);
        void test(double x, double w);
        ~Lr() = default;
    protected:
        std::vector<double> x;
        std::vector<double> w;
};

#endif