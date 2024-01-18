#include <iostream>
#include "wet.h"
#include "tool.h"
#include "read.h"
#include "forest.h"
#include "cmdlines.h"
#include "ode.h"
#include "optimizer.h"
#include <random>

void test1()
{
    auto f = build_forest_standard(2, 2, 1, std::pair<double, double>(-1, 1), 1);
    auto data = std::vector<double>({1.0, 2.0});
    auto result = run_forest(f, data);
    gradient_forest(f, data);
    std::cout << result[0] << std::endl;
    std::cout << f.trees_weights.dnode_dvariable[0][ID_N][0] << ", " << f.trees_weights.dnode_dvariable[0][ID_N][1] << std::endl;

    auto sol = Solve(f, data, std::pair<double, double>(0.0, 100.0));
    for (auto i = 0; i < sol.t.size(); i++)
    {
        std::cout << sol.t[i] << " " << sol.u[i][0] << " " << sol.u[i][1] << std::endl;
    }
    auto sol2 = Solve_withgrad(f, data, std::pair<double, double>(0.0, 100.0));
}

int main(int argc, char *argv[])
{

    cmdline::parser a;
    a.add<std::string>("DataT", 't', "Time points", true);
    a.add<std::string>("DataX", 'x', "Data", true);
    a.add<double>("ConstantUpperBound", 'u', "Upper bound of constant.", false, 1.0);
    a.add<double>("ConstantLowerBound", 'l', "Lower bound of constant.", false, -1.0);
    a.add<int>("RandSeed", 'r', "The seed of random number generator.", false, 1);
    a.add<int>("NumNodes", 'n', "The number of nodes.", true);
    a.add<double>("StepLength", 's', "Step length.", false, 0.01);
    a.add<double>("Epsilon", 'e', "Epsilon to control the penalty function.", false, 1.0);
    a.add<int>("MaxEpoch", '\0', "Maximum number of iterations", false, 10000);
    a.add<int>("WhenPenalty", '\0', "The time begin to decrease penalty function.", false, 500);
    a.add<int>("PreTraining", '\0', "Pre-training. Default is false", false, 0);
    a.add<std::string>("PreX", '\0', "Pretraining data x.", false, " ");
    a.add<std::string>("PreY", '\0', "Pretraining data y.", false, " ");
    a.parse_check(argc, argv);

    std::vector<std::vector<double>> datax = readmatrix(a.get<std::string>("DataX"));
    std::vector<double> datat = readvector(a.get<std::string>("DataT"));

    int n = datax[0].size();
    int m = datat.size();
    register_node(n);

    auto constant_lower = a.get<double>("ConstantLowerBound");
    auto constant_upper = a.get<double>("ConstantUpperBound");
    auto num_nodes = a.get<int>("NumNodes");
    auto randseed = a.get<int>("RandSeed");

    auto trees = build_forest_standard(n, n, num_nodes, std::pair<double, double>(constant_lower, constant_upper), randseed);
    double step_length = a.get<double>("StepLength");
    double epsilon = a.get<double>("Epsilon");
    int MaxEpoch = a.get<int>("MaxEpoch");
    int WhenPenalty = a.get<int>("WhenPenalty");
    printf("Starting time: %f\n", trees_loss_mse(trees, datat, datax, 10.0));
    int j = 1;
    bool IfPenalty = false;
    // 预训练
    auto IfPre = a.get<int>("PreTraining");
    MAP_GRADIENT v, s, delta, g;
    v = generate_gradient_map(trees);
    s = generate_gradient_map(trees);
    auto nodetype = generate_nodetype_map(trees);
    auto constant_range = std::pair<double, double>(constant_lower, constant_upper);
    if (IfPre == 1)
    {
        std::vector<std::vector<double>> pre_x = readmatrix(a.get<std::string>("PreX"));
        std::vector<std::vector<double>> pre_y = readmatrix(a.get<std::string>("PreY"));
        while (j < 100)
        {
            delta = forest_loss_mse_gradient(trees, pre_x, pre_y);
            // Momentum(v, delta, 0.8, 0.2);
            make_feasible_grad(delta, nodetype, trees.trees_weights.weights,constant_range);
            Adam(v, s, delta, g, j);
            // trees_train_fixstep(trees, g, step_length, epsilon, IfPenalty, std::pair<double, double>(constant_lower, constant_upper));
            make_feasible_grad(g, nodetype, trees.trees_weights.weights,constant_range);
            TRAIN_FOREST(trees, g);
            //printf("Pretraining Iter time: %d       %f\n", j, forest_loss_mse_batch(trees, pre_x, pre_y));
            j++;
            step_length *= 0.95;
        }
        j = 1;
        step_length = 1;
    }



    printf("Iter time: %d       %f\n", j, trees_loss_mse(trees, datat, datax, 10.0));
    while (j < MaxEpoch)
    {
         VEC_DOUBLE train_t;
         std::vector<VEC_DOUBLE> train_x;
         for (int q = 0; q < std::min(int(double(j) / 50.0), m) + 2 && q < datat.size(); q++)
         {
             train_t.push_back(datat[q]);
             train_x.push_back(datax[q]);
         }
        delta = trees_loss_mse_gradient(trees, train_t, train_x, 5.0, "Euler");
        make_feasible_grad(delta, nodetype, trees.trees_weights.weights,constant_range);
        Adam(v, s, delta, g, j);
        if (j>=WhenPenalty)
            adaptive_penalty_grad(g,nodetype,trees.trees_weights.weights,penalty_func_gradient_3,epsilon);
        make_feasible_grad(g, nodetype, trees.trees_weights.weights,constant_range);
        TRAIN_FOREST(trees, g);
        double pp = 0.0;
        for (auto t : trees.trees){
            pp +=  penalty_func_1(t, trees.trees_weights);
        }
        if (j % 100 == 0)
            printf("Iter time: %d  Error: %f  Penalty: %f\n", j, trees_loss_mse(trees, datat, datax, 10.0,"Euler"),pp);
        j++;
    }
    PRINT_FOREST(trees);
    return 0;
}

// ../../build/ ./odefitting --DataX=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_x.txt --DataT=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_t.txt --ConstantUpperBound=10.0 --ConstantLowerBound=-10.0 --NumNodes=3 --StepLength=0.05 --Epsilon=10.0 --MaxEpoch=5000 --WhenPenalty=3000 --PreTraining=1 --PreX=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_x.txt --PreY=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_pre_y.txt

//./odefitting --DataX=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_x2.txt --DataT=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_t2.txt --ConstantUpperBound=10.0 --ConstantLowerBound=-10.0 --NumNodes=3 --StepLength=1.0 --Epsilon=10.0 --MaxEpoch=5000 --WhenPenalty=3000 --PreTraining=1 --PreX=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_x.txt --PreY=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_pre_y.txt