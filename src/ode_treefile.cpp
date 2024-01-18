#include <iostream>
#include "wet.h"
#include "tool.h"
#include "read.h"
#include "forest.h"
#include "cmdlines.h"
#include "ode.h"
#include "optimizer.h"
#include "interpolation.h"

// test
#include <fstream>

int main(int argc, char *argv[])
{
    cmdline::parser a;
    a.add<std::string>("DataT", 't', "Time points", true);
    a.add<std::string>("DataX", 'x', "Data", true);
    a.add<double>("ConstantUpperBound", 'u', "Upper bound of constant.", false, 1.0);
    a.add<double>("ConstantLowerBound", 'l', "Lower bound of constant.", false, -1.0);
    a.add<std::string>("TreeFile", 'n', "JSON file describing tree structure.", true);
    a.add<double>("Epsilon", 'e', "Epsilon to control the penalty function.", false, 1.0);
    a.add<int>("MaxEpoch", '\0', "Maximum number of iterations", false, 10000);
    a.add<int>("WhenPenalty", '\0', "The time begin to decrease penalty function.", false, 500);
    a.add<int>("PreTraining", '\0', "Pre-training. Default is false", false, 0);
    a.add<std::string>("OperatorSetting", 'o', "The setting file of operator.", false, "op.ini");
    a.parse_check(argc, argv);

    std::vector<std::vector<double>> datax = readmatrix(a.get<std::string>("DataX"));
    std::vector<double> datat = readvector(a.get<std::string>("DataT"));

    int n = datax[0].size();
    int m = datat.size();
    int G = 100; // 这个变量表示将全时间长度分成G份
    double DeltaT = (datat[datat.size() - 1] - datat[0]) / double(G+1);
    // register_node(n);

    auto constant_lower = a.get<double>("ConstantLowerBound");
    auto constant_upper = a.get<double>("ConstantUpperBound");
    auto treefile = a.get<std::string>("TreeFile");
    auto operatorfile = a.get<std::string>("OperatorSetting");
    setTreeOpsFromFile(operatorfile);

    auto trees = parse_forest(treefile);
    double epsilon = a.get<double>("Epsilon");
    int MaxEpoch = a.get<int>("MaxEpoch");
    int WhenPenalty = a.get<int>("WhenPenalty");
    // printf("Starting time: %f\n", trees_loss_mse(trees, datat, datax, DeltaT));
    int j = 1;
    // 预训练
    auto IfPre = a.get<int>("PreTraining");
    MAP_GRADIENT v, s, delta, g;
    v = generate_gradient_map(trees);
    s = generate_gradient_map(trees);
    auto nodetype = generate_nodetype_map(trees);
    auto constant_range = std::pair<double, double>(constant_lower, constant_upper);

    // 对输入数据插值处理
    std::vector<Spline> interps;
    for (int i = 0; i < n; i++)
    {
        std::vector<double> y;
        for (auto it : datax)
        {
            y.push_back(it[i]);
        }
        interps.push_back(Spline(datat, y, 1));
    }
    // 生成向场插值数据
    std::vector<std::vector<double>> pre_x;
    std::vector<std::vector<double>> pre_y;
    auto lin_time = linspace(datat[0], datat[datat.size() - 1], DeltaT);
    for (int i = 0; i < lin_time.size(); i++)
    {
        pre_x.emplace_back();
        pre_y.emplace_back();
        for (int j = 0; j < n; j++)
        {
            pre_x[i].push_back(interps[j].f(lin_time[i]));
            pre_y[i].push_back(interps[j].f_d(lin_time[i]));
        }
    }

    // test
    //    std::ofstream fout;
    //    fout.open("/home/zhouyu/work/wetcpp/scripts/temp_savedata.txt",std::ios::out);
    //    for (int i=0;i < lin_time.size();i++){
    //        fout << lin_time[i];
    //        for (int j=0;j<n;j++){
    //            fout << " " << pre_x[i][j];
    //        }
    //        for (int j=0;j<n;j++){
    //            fout << " " << pre_y[i][j];
    //        }
    //        fout << "\n";
    //    }
    //    fout.close();

    if (IfPre >= 1)
    {
        // std::vector<std::vector<double>> pre_x = readmatrix(a.get<std::string>("PreX"));
        // std::vector<std::vector<double>> pre_y = readmatrix(a.get<std::string>("PreY"));
        auto dataloader = DataLoader(pre_x, pre_y, 1, true);
        while (j <= IfPre)
        {
            for (auto data : dataloader)
            {
                delta = forest_loss_mse_gradient(trees, data.first, data.second);
                // Momentum(v, delta, 0.8, 0.2);
                // make_feasible_grad(delta, nodetype, trees.trees_weights.weights, constant_range);
                Adam(v, s, delta, g, j);
                // if (j>=WhenPenalty)
                // adaptive_penalty_grad(g,nodetype,trees.trees_weights.weights,penalty_func_gradient_3,epsilon);
                make_feasible_grad(g, nodetype, trees.trees_weights.weights, constant_range);
                TRAIN_FOREST(trees, g);
            }
            printf("Pretraining Iter time: %d       %f\n", j, forest_loss_mse_batch(trees, pre_x, pre_y));
            j++;
        }
        j = 1;
    }
    // find dt

    // 找dt 标准: 不能太小(计算量大) 不能太大(误差大) 先人为给定

    // generate data
    double gamma;
    int count_converge = 0;
    double loss_record_1 = 100, loss_record_2 = 100;
    while (j < MaxEpoch)
    {
        //        VEC_DOUBLE train_t;
        //        std::vector<VEC_DOUBLE> train_x;
        //        for (int q = 0; q < std::min(int(double(j) / 50.0), m) + 2 && q < datat.size(); q++)
        //        {
        //            train_t.push_back(datat[q]);
        //            train_x.push_back(datax[q]);
        //        }
        //        delta = trees_loss_mse_gradient(trees, train_t, train_x, 0.1, "Euler");
        gamma = 1.0 / (1 + 9.0 * std::exp(-double(0.001 * j)));
        delta = trees_loss_search_mse_gradient(trees, datat, datax, DeltaT, gamma, interps, "RK45");
        // make_feasible_grad(delta, nodetype, trees.trees_weights.weights,constant_range);
        Adam(v, s, delta, g, j);
        if (j >= WhenPenalty)
            adaptive_penalty_grad(g, nodetype, trees.trees_weights.weights, penalty_func_gradient_3, epsilon);
        make_feasible_grad(g, nodetype, trees.trees_weights.weights, constant_range);
        TRAIN_FOREST(trees, g);
        loss_record_1 = trees_loss_mse(trees, datat, datax, DeltaT, "RK45");
        double pp = 0.0;
        for (auto t : trees.trees)
        {
            pp += penalty_func_1(t, trees.trees_weights);
        }
        printf("Iter time: %d   Error:%f    Penalty:%f\n", j, loss_record_1, pp);
        if (pp == 0.0)
        {
            if (std::abs(loss_record_1 - loss_record_2) < 1e-6)
            {
                count_converge++;
                if (count_converge >= 20)
                {
                    break;
                }
            }
            else
            {
                count_converge = 0;
            }
        }
        loss_record_2 = loss_record_1;
        j++;
    }
    PRINT_FOREST(trees);
    return 0;
}

//--DataX=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_x2.txt --DataT=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_t2.txt --ConstantUpperBound=10.0 --ConstantLowerBound=-10.0 --TreeFile=/home/zhouyu/work/wetcpp/scripts/example.json --StepLength=1.0 --Epsilon=10.0 --MaxEpoch=5000 --WhenPenalty=3000 --PreTraining=1 --PreX=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_x.txt --PreY=/home/zhouyu/work/wetcpp/scripts/julia/data/ode_1_pre_y.txt
