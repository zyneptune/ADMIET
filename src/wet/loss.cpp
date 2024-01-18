#include "wet.h"
#include "tool.h"
#include <cmath>

extern calculation_type CAL_TYPE;
extern calculation_node_map CAL_NODE_MAP;

double tree_loss_mse(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input, double y)
{
    double t = std::fabs(run_tree(tree, ws, ps, input) - y);
    return t * t;
}

double tree_loss_mse_batch(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> input, VEC_DOUBLE y)
{
    double sum_ = 0.0;
    for (unsigned i = 0; i < input.size(); i++)
    {
        double t = std::fabs(run_tree(tree, ws, ps, input[i]) - y[i]);
        sum_ += t * t;
    }
    return sum_ / double(input.size());
}

std::map<unsigned, VEC_DOUBLE> tree_loss_mse_gradient(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input, double y)
{
    std::map<unsigned, VEC_DOUBLE> loss_gradient;
    gradient_tree(tree, ws, ps, input);
    // 2 * (x-y) * dx/dw
    double c = 2.0 * (ps.node_values[0][0] - y);
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        loss_gradient.insert(std::pair<unsigned, VEC_DOUBLE>(id, vec_mul(ws.dnode_dvariable[0][id], c)));
    }
    return loss_gradient;
}

std::map<unsigned, VEC_DOUBLE> tree_loss_mse_gradient_batch(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> input, VEC_DOUBLE y)
{
    std::map<unsigned, VEC_DOUBLE> loss_gradient;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        loss_gradient.insert(std::pair<unsigned, VEC_DOUBLE>(id, VEC_DOUBLE()));
        loss_gradient[id].assign(ws.weights[id].size(), 0.0);
    }

    for (unsigned j = 0; j < input.size(); j++)
    {
        run_tree(tree, ws, ps, input[j]);
        gradient_tree(tree, ws, ps, input[j]);
        // 2 * (x-y) * dx/dw
        double c = 2.0 * (ps.node_values[0][0] - y[j]);
        for (unsigned i = 0; i < tree.node_names.size(); i++)
        {
            auto id = tree.node_names[i];
            vec_add(loss_gradient[id], vec_mul(ws.dnode_dvariable[0][id], c));
        }
    }
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        vec_mul_inplace(loss_gradient[id], 1.0 / double(input.size()));
    }
    return loss_gradient;
}

VEC_DOUBLE tree_loss_mse_gradient_batch_2node(Tree tree, Weights &ws, Progress &ps, unsigned node_id, std::vector<VEC_DOUBLE> input, VEC_DOUBLE y)
{
    VEC_DOUBLE loss_gradient;
    loss_gradient.assign(ws.weights[node_id].size(), 0.0);

    for (unsigned j = 0; j < input.size(); j++)
    {
        run_tree(tree, ws, ps, input[j]);
        gradient_tree_top2node(tree, ws, ps, node_id, input[j]);
        // 2 * (x-y) * dx/dw
        double c = 2.0 * (ps.node_values[0][0] - y[j]);
        vec_add(loss_gradient, vec_mul(ws.dnode_dvariable[0][node_id], c));
    }
    vec_mul_inplace(loss_gradient, 1.0 / double(input.size()));
    return loss_gradient;
}

std::map<unsigned, VEC_DOUBLE> tree_loss_mse_gradient_node(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input, double y)
{
    std::map<unsigned, VEC_DOUBLE> loss_gradient;
    gradient_tree(tree, ws, ps, input);
    // 2 * (x-y) * dx/dw
    double c = 2.0 * (ps.node_values[0][0] - y);
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        loss_gradient.insert(std::pair<unsigned, VEC_DOUBLE>(id, vec_mul(ws.dnode_dvariable[0][id], c)));
    }
    return loss_gradient;
}

double penalty_func_1(Tree tree, Weights &ws)
{
    double sum_ = 0.0;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR)
        {
            double t = std::pow(norm(ws.weights[id], '2'), 2);
            sum_ += (1.0 - t);
        }
    }
    return sum_;
}

// ## new method
std::map<unsigned, VEC_DOUBLE> tree_loss_gradient(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> x, VEC_DOUBLE y, double(error_func_g)(double x, double y))
{
    std::map<unsigned, VEC_DOUBLE> loss_gradient = generate_gradient_map(ws);
    double c;
    for (int i=0;i<x.size();i++){
        run_tree(tree,ws,ps,x[i]);
        gradient_tree(tree,ws,ps,x[i]);
        c = error_func_g(ps.node_values[0][0],y[i]) / double(x.size());
        for (auto it : loss_gradient){
            vec_add(loss_gradient[it.first], vec_mul(ws.dnode_dvariable[0][it.first],c));
        }
    }
    return loss_gradient;
}

double tree_loss(Tree tree,Weights &ws,Progress &ps, std::vector<VEC_DOUBLE> x, VEC_DOUBLE y, double(error_func)(double x, double y)){
    double result = 0.0;
    for (int i=0;i<x.size();i++) {
        run_tree(tree,ws,ps,x[i]);
        result += error_func(ps.node_values[0][0],y[i]) / double(x.size());
    }
    return result;
}

double tree_penalty(Tree tree,Weights &ws, double(pen_func)(VEC_DOUBLE z)) {
    double sum_ = 0.0;
    for (auto it : ws.weights){
        if (CAL_NODE_MAP.node_isonehot[tree.node_type[it.first]]){
            sum_ += pen_func(it.second);
        }
    }
    return sum_;
}