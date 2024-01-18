#include <iostream>
#include "wet.h"
#include "tool.h"
#include "read.h"
#include "cmdlines.h"
#include "optimizer.h"
#include <random>

double L2(double x,double y){
    return (x-y)*(x-y);
}
double L1(double x,double y){
    return std::fabs(x-y);
}
double related_L1(double x,double y){
    return std::fabs(x-y) / (std::fabs(y)+1.0);
}
double L2_grad(double x, double y){
    return 2.0*(x-y);
}

int main(int argc, char *argv[])
{
    cmdline::parser a;
    a.add<std::string>("DataX", 'x', "Data of independent variable.", true);
    a.add<std::string>("DataY", 'y', "Data of dependent variable.", true);
    a.add<double>("ConstantUpperBound", 'u', "Upper bound of constant.", false, 1.0);
    a.add<double>("ConstantLowerBound", 'l', "Lower bound of constant.", false, -1.0);
    a.add<int>("RandSeed", 'r', "The seed of random number generator.", false, 1);
    a.add<int>("NumNodes", 'n', "The number of nodes.", true);
    a.add<double>("StepLength", 's', "Step length.", false, 0.01);
    a.add<double>("Epsilon", 'e', "Epsilon to control the penalty function.", false, 1.0);
    a.add<int>("MaxEpoch", '\0', "Maximum number of iterations", false, 10000);
    a.add<int>("WhenPenalty", '\0', "The time begin to decrease penalty function.", false, 500);
    a.add<bool>("ShowError", '\0', "Show the progress.", false, false);

    a.parse_check(argc, argv);

    std::vector<std::vector<double>> datax = readmatrix(a.get<std::string>("DataX"));
    std::vector<double> datay = readvector(a.get<std::string>("DataY"));

    int m = datax.size();
    int n = datax[0].size();
    // printf("Data size: %d,%d", m, n);
    register_node(n);

    auto constant_lower = a.get<double>("ConstantLowerBound");
    auto constant_upper = a.get<double>("ConstantUpperBound");
    auto num_nodes = a.get<int>("NumNodes");
    auto randseed = a.get<int>("RandSeed");
    auto tree = build_tree_standard(num_nodes);
    auto ws = generate_weights_from_tree(tree, n, 2, std::pair<double, double>(constant_lower, constant_upper), randseed);
    auto ps = generate_progress_from_tree(tree);
    auto showError = a.get<bool>("ShowError");
    double epsilon = a.get<double>("Epsilon");
    auto constant_range = std::pair<double, double>(constant_lower, constant_upper);
    int j = 1;
    int MaxEpoch = a.get<int>("MaxEpoch");
    int WhenPenalty = a.get<int>("WhenPenalty");
    MAP_GRADIENT v, s, delta, g;
    v = generate_gradient_map(ws);
    s = generate_gradient_map(ws);
    //g = generate_gradient_map(ws);
    double error = 100, penalty = 100.0;
    while (j <= MaxEpoch)
    {
        delta = tree_loss_gradient(tree,ws,ps,datax,datay,L2_grad);
        make_feasible_grad(delta,tree.node_type,ws.weights,constant_range);
        Adam(v, s, delta, g, j);
        if (j>=WhenPenalty)
            adaptive_penalty_grad(g,tree.node_type,ws.weights,penalty_func_gradient_3,epsilon);
        make_feasible_grad(g, tree.node_type,ws.weights,constant_range);
        TRAIN_TREE(tree,ws,g);
        error = tree_loss(tree,ws,ps,datax,datay,L2);
        penalty = penalty_func_1(tree, ws);
        if (showError){
            printf("Iteration:%d Error: %f, Penalty function: %f\n",j,error,penalty);
        }
        if (error < 1e-12 && penalty == 0.0){
            break;
        }
        j++;
    }
    if (showError)
    {
        print_tree(tree, ws);
        printf("\n");
    }
    print_tree(tree, ws);
    printf("\n");
    printf("%f %f %f\n", tree_loss(tree,ws,ps,datax,datay,L1), penalty_func_1(tree, ws),tree_loss(tree,ws,ps,datax,datay,related_L1));
    return 0;
}
