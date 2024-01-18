#include <iostream>
#include "wet.h"
#include "tool.h"
#include "read.h"
#include "cmdlines.h"

#include <random>

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
    double step_length = a.get<double>("StepLength");
    double epsilon = a.get<double>("Epsilon");

    int j = 0;
    double temp_error = tree_loss_mse_batch(tree, ws, ps, datax, datay);
    double current_error = 0.0;
    bool IfPenalty = false;
    auto pv = penalty_func_1(tree, ws);

    int MaxEpoch = a.get<int>("MaxEpoch");
    int WhenPenalty = a.get<int>("WhenPenalty");

    while (j < MaxEpoch)
    {
        // pv = penalty_func_1(tree, ws);
        if (pv > 0)
        {
            auto g = tree_loss_mse_gradient_batch(tree, ws, ps, datax, datay);
            tree_train_fixstep(tree, ws, g, step_length, epsilon, IfPenalty, std::pair<double, double>(constant_lower, constant_upper));
        }
        else
        {
            tree_train_linearsearch(tree, ws, ps, datax, datay, tree_loss_mse_batch, epsilon, true, std::pair<double, double>(constant_lower, constant_upper));
        }
        pv = penalty_func_1(tree, ws);
        current_error = tree_loss_mse_batch(tree, ws, ps, datax, datay);
        if (j % 100 == 0 && showError)
        {
            printf("%f\n", step_length);
            printf("#####################################\n");
            print_tree(tree, ws);
            printf("\n");
            std::cout << "Run " << j << " Error: " << current_error << "  Penalty : " << pv << std::endl;
            for (int k1 = 0; k1 < m; k1++)
            {
                printf("%f %f %f\n", datax[k1][0], datay[k1], run_tree(tree, ws, ps, datax[k1]));
            }
            printf("#####################################\n\n\n");
        }
        if (j == WhenPenalty)
        {
            IfPenalty = true;
        }
        if (std::fabs(current_error - temp_error) < 1e-6 && IfPenalty)
        {
            break;
        }
        // if (current_error > temp_error && IfPenalty == false)
        //{
        //  step_length *= 0.9;
        //}
        if (current_error > temp_error && pv == 0.0)
        {
            step_length *= 0.9;
        }
        // if (step_length < 1e-6)
        //{
        //      break;
        // }
        temp_error = current_error;
        j++;
    }
    if (showError)
    {
        print_tree(tree, ws);
        printf("\n");
    }
    printf("%f %f\n", tree_loss_mse_batch(tree, ws, ps, datax, datay), penalty_func_1(tree, ws));
    return 0;
}
