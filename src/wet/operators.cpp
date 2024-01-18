#include "wet.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

extern calculation_type CAL_TYPE;

inline double plus(VEC_DOUBLE x)
{
    return x[0] + x[1];
}
inline VEC_DOUBLE plus_gradient(VEC_DOUBLE x)
{
    VEC_DOUBLE gradient;
    gradient.assign(2, 1.0);
    return gradient;
}

inline double minus(VEC_DOUBLE x)
{
    return x[0] - x[1];
}
inline VEC_DOUBLE minus_gradient(VEC_DOUBLE x)
{
    VEC_DOUBLE gradient;
    gradient.assign(2, 1.0);
    gradient[1] = -1.0;
    return gradient;
}

inline double multi(VEC_DOUBLE x)
{
    return x[0] * x[1];
}

inline VEC_DOUBLE multi_gradient(VEC_DOUBLE x)
{
    VEC_DOUBLE gradient;
    gradient.assign(2, 0.0);
    gradient[0] = x[1];
    gradient[1] = x[0];
    return gradient;
}

inline double protect_divide(VEC_DOUBLE x)
{
    if (fabs(x[1]) < 1e-8)
        return 1.0;
    else
        return x[0] / x[1];
}

inline VEC_DOUBLE protect_divide_gradient(VEC_DOUBLE x)
{
    VEC_DOUBLE gradient;
    gradient.assign(2, 0.0);
    if (fabs(x[1]) < 1e-8)
    {
    }
    else
    {
        gradient[0] = 1.0 / x[1];
        gradient[1] = -x[0] / (x[1] * x[1]);
    }
    return gradient;
}

inline double operator_example(VEC_DOUBLE x)
{
    return 1.0;
}

inline VEC_DOUBLE example_gradient(VEC_DOUBLE x)
{
    return VEC_DOUBLE({0.0, 0.0});
}

inline double hill2(VEC_DOUBLE x)
{
    return (x[0] * x[0]) / (x[0] * x[0] + x[1] * x[1] + 1e-8);
}

inline VEC_DOUBLE hill2_gradient(VEC_DOUBLE x)
{
    VEC_DOUBLE t = {0.0, 0.0};
    t[0] = 2.0 * x[0] / ((x[0] * x[0] + x[1] * x[1] + 1e-8) * (x[0] * x[0] + x[1] * x[1] + 1e-8));
    t[1] = -2.0 * x[0] * x[0] * x[1] / ((x[0] * x[0] + x[1] * x[1] + 1e-8) * (x[0] * x[0] + x[1] * x[1] + 1e-8));
    return t;
}

TreeOp Tree_plus = {2, "+", plus, plus_gradient};
TreeOp Tree_minus = {2, "-", minus, minus_gradient};
TreeOp Tree_multi = {2, "*", multi, multi_gradient};
TreeOp Tree_divide = {2, "/", protect_divide, protect_divide_gradient};
TreeOp Tree_hill2 = {2, "h/2", hill2, hill2_gradient};

TreeOpSet TreeOps = {{Tree_plus, Tree_minus, Tree_multi, Tree_divide}, 2};
// TreeOpSet TreeOps = {};

void setTreeOpsFromFile(std::string filename)
{
    std::ifstream fin;
    fin.open(filename, std::ios::in);
    if (!fin.is_open())
    {
        return;
    }
    // 逐行读取
    TreeOps.funcs.clear();
    std::string str;
    std::vector<std::string> ops;

    std::getline(fin, str);
    if (str == "[operator]")
    {
        std::cout << "This is a right file!\n";
    }
    while (std::getline(fin, str))
    {
        str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
        std::vector<std::string>::iterator iter = std::find(ops.begin(), ops.end(), str);
        if (iter != ops.end())
        {
            continue;
        }
        ops.push_back(str);
        if (str == "+")
        {
            TreeOps.funcs.push_back(Tree_plus);
            continue;
        }
        if (str == "-")
        {
            TreeOps.funcs.push_back(Tree_minus);
            continue;
        }
        if (str == "*")
        {
            TreeOps.funcs.push_back(Tree_multi);
            continue;
        }
        if (str == "/")
        {
            TreeOps.funcs.push_back(Tree_divide);
            continue;
        }
        if (str == "h/2")
        {
            TreeOps.funcs.push_back(Tree_hill2);
            continue;
        }
    }
    std::cout << TreeOps.funcs.size() << std::endl;
}