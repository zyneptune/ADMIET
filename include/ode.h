#include "wet.h"
#include "forest.h"
#include "interpolation.h"
struct odeGradient
{
    std::map<unsigned, std::map<unsigned, VEC_DOUBLE>> dxdv;
};

struct odeSolution
{
    VEC_DOUBLE t;
    std::vector<VEC_DOUBLE> u;
    std::vector<odeGradient> grad;
};

odeSolution Solve(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt = -1.0, VEC_DOUBLE saveat = VEC_DOUBLE(), std::string method = "RK45");
odeSolution Solve_withgrad(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt = -1.0, VEC_DOUBLE saveat = VEC_DOUBLE(), std::string method = "RK45");
std::map<unsigned, std::map<unsigned, VEC_DOUBLE>> generate_dxdv_fromForest(Forest &f);

double trees_loss_mse(Forest &f, VEC_DOUBLE time, std::vector<VEC_DOUBLE> x, double dt, std::string method = "RK45");

int trees_train_fixstep(Forest &f, MAP_GRADIENT g, double step_length, double epsilon, bool IsPenalty, std::pair<double, double> constant_range);
std::map<unsigned, VEC_DOUBLE> trees_loss_mse_gradient(Forest &f, VEC_DOUBLE time, std::vector<VEC_DOUBLE> x, double dt, std::string method);
std::map<unsigned, VEC_DOUBLE> trees_loss_search_mse_gradient(Forest &f, VEC_DOUBLE time, std::vector<VEC_DOUBLE> x, double dt,double gamma,std::vector<Spline> &interps, std::string method);