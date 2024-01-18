#include "wet.h"
#include "tool.h"

void Momentum(MAP_GRADIENT &v, MAP_GRADIENT &delta, double eta = 0.9, double alpha = 0.1);
void Adam(MAP_GRADIENT &v, MAP_GRADIENT &s, MAP_GRADIENT &delta, MAP_GRADIENT &g, int t, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

void normalize_grad(MAP_GRADIENT &g, std::map<unsigned, int> node_type);
void add_grad(MAP_GRADIENT &g, MAP_GRADIENT src);
void make_feasible_grad(MAP_GRADIENT &g, std::map<unsigned, int> node_type, std::map<unsigned, VEC_DOUBLE> weights, std::pair<double, double> cr);
MAP_GRADIENT generate_penalty_grad(std::map<unsigned, VEC_DOUBLE> weights, std::map<unsigned, int> node_type, VEC_DOUBLE(p_func_g)(VEC_DOUBLE x), double gamma);
void adaptive_penalty_grad(MAP_GRADIENT &g, std::map<unsigned, int> node_type, std::map<unsigned, VEC_DOUBLE> weights, VEC_DOUBLE(p_func_g)(VEC_DOUBLE x), double eps);
