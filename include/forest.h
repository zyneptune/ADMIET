#ifndef FOREST_H
#define FOREST_H
#include "wet.h"
struct Forest
{
    std::vector<Tree> trees;
    std::vector<unsigned> root_id;
    unsigned fdim;
    unsigned xdim;
    Weights trees_weights;
    Progress trees_progress;
};

VEC_DOUBLE run_forest(Forest &forest, VEC_DOUBLE input);
void gradient_forest(Forest &forest, VEC_DOUBLE input);
Forest build_forest_standard(unsigned f_dim, unsigned x_dim, unsigned head_length, std::pair<double, double> cr, int randseed);
std::map<unsigned, VEC_DOUBLE> generate_gradient_map(Forest f);
std::map<unsigned, int> generate_nodetype_map(Forest f);
std::map<unsigned, VEC_DOUBLE> forest_loss_mse_gradient(Forest &f, std::vector<VEC_DOUBLE> datax, std::vector<VEC_DOUBLE> datay);
double forest_loss_mse_batch(Forest &f, std::vector<VEC_DOUBLE> datax, std::vector<VEC_DOUBLE> datay);
void TRAIN_FOREST(Forest &f, MAP_GRADIENT &g);
void PRINT_FOREST(Forest &f);
#endif