#include "wet.h"
#include "forest.h"
#include "tool.h"
#include <iostream>
// multidimension function

// 计算梯度的过程中需要
// f对x x对theta, f对theta

void run_tree_(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input)
{
    for (int i = tree.layer_nodes.size() - 1; i >= 0; i--)
    {
        for (unsigned j = 0; j < tree.layer_nodes[i].size(); j++)
        {
            run_node(ps, ws, tree.layer_nodes[i][j], tree.connection_down[tree.layer_nodes[i][j]], input, tree.node_type[tree.layer_nodes[i][j]]);
        }
    }
}

VEC_DOUBLE run_forest(Forest &forest, VEC_DOUBLE input)
{
    VEC_DOUBLE a;
    a.assign(forest.fdim, 0.0);
    for (unsigned i = 0; i < forest.trees.size(); i++)
    {
        run_tree_(forest.trees[i], forest.trees_weights, forest.trees_progress, input);
    }
    for (unsigned i = 0; i < forest.fdim; i++)
    {
        a[i] = forest.trees_progress.node_values[forest.root_id[i]][0];
    }
    // if (norm(a, '2') > 1e5)
    // {
    //     vec_mul_inplace(a, 1.0 / norm(a, '2'));
    // }
    return a;
}

void gradient_forest(Forest &forest, VEC_DOUBLE input)
{
    for (unsigned i = 0; i < forest.trees.size(); i++)
    {
        gradient_tree(forest.trees[i], forest.trees_weights, forest.trees_progress, input);
    }
}

Tree build_tree_standard_IDBIAS(unsigned head_length, unsigned ID_BIAS)
{
    Tree tree;
    tree.node_names.clear();

    // connection
    for (unsigned int i = 0; i < head_length * 2 + 1; i++)
    {
        if (i < head_length)
        {
            tree.node_names.push_back(i + ID_BIAS);
            tree.operator_nodes.push_back(i + ID_BIAS);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_BIAS, {2 * i + 1 + ID_BIAS, 2 * i + 2 + ID_BIAS}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_BIAS, CT_OPS));
        }
        else
        {
            tree.node_names.push_back(i + ID_BIAS);
            tree.operator_nodes.push_back(i + ID_BIAS);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_BIAS, {i + ID_N * 2 + ID_BIAS, ID_N + i + ID_BIAS}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_BIAS, CT_WEIGHTED_PLUS_2));

            tree.node_names.push_back(i + ID_N * 2 + ID_BIAS);
            tree.terminal_nodes.push_back(i + ID_N * 2 + ID_BIAS);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N * 2 + ID_BIAS, {}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N * 2 + ID_BIAS, CT_CON));

            tree.node_names.push_back(i + ID_N + ID_BIAS);
            tree.terminal_nodes.push_back(i + ID_N + ID_BIAS);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N + ID_BIAS, {i + ID_N * 3 + ID_BIAS, i + ID_N * 4 + ID_BIAS}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N + ID_BIAS, CT_MUL2));

            tree.node_names.push_back(i + ID_N * 3 + ID_BIAS);
            tree.terminal_nodes.push_back(i + ID_N * 3 + ID_BIAS);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N * 3 + ID_BIAS, {}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N * 3 + ID_BIAS, CT_CON));

            tree.node_names.push_back(i + ID_N * 4 + ID_BIAS);
            tree.terminal_nodes.push_back(i + ID_N * 4 + ID_BIAS);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N * 4 + ID_BIAS, {}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N * 4 + ID_BIAS, CT_VAR));
        }
    }
    // gradient chain
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        std::vector<unsigned> l;
        l.push_back(tree.node_names[i]);
        tree.connection_gradient.insert(std::pair<unsigned, VEC_ID>(tree.node_names[i], {ID_N}));
        while (!l.empty())
        {
            auto c = l.back();
            l.pop_back();
            tree.connection_gradient[tree.node_names[i]].push_back(c);
            std::for_each(tree.connection_down[c].begin(), tree.connection_down[c].end(), [&](unsigned n)
                          { l.push_back(n); });
        }
    }
    // layer
    tree.layer_nodes.push_back({ID_BIAS});
    std::vector<unsigned> l = tree.layer_nodes[0];

    while (!l.empty())
    {
        std::vector<unsigned> l2;
        for (unsigned i = 0; i < l.size(); i++)
        {
            for (unsigned j = 0; j < tree.connection_down[l[i]].size(); j++)
            {
                l2.push_back(tree.connection_down[l[i]][j]);
            }
        }
        if (l2.size() != 0)
        {
            tree.layer_nodes.push_back({});
            tree.layer_nodes.back() = l2;
        }
        l = l2;
    }

    return tree;
}

Forest build_forest_standard(unsigned f_dim, unsigned x_dim, unsigned head_length, std::pair<double, double> cr, int randseed)
{
    Forest f;
    f.fdim = f_dim;
    std::vector<Weights> trees_weights;
    std::vector<Progress> trees_progress;
    for (unsigned i = 0; i < f_dim; i++)
    {
        int ID_BIAS = i * 10000;
        f.trees.push_back(build_tree_standard_IDBIAS(head_length, ID_BIAS));
        f.root_id.push_back(ID_BIAS);
        trees_weights.push_back(generate_weights_from_tree(f.trees[i], x_dim, 1, cr, randseed));
        trees_progress.push_back(generate_progress_from_tree(f.trees[i]));
    }
    f.trees_weights = trees_weights[0];
    f.trees_progress = trees_progress[0];
    for (unsigned i = 1; i < f_dim; i++)
    {
        f.trees_weights = merge_weights(f.trees_weights, trees_weights[i]);
        f.trees_progress = merge_progress(f.trees_progress, trees_progress[i]);
    }
    f.trees_weights.untrainable.clear();
    return f;
}

std::map<unsigned, VEC_DOUBLE> generate_gradient_map(Forest f)
{
    std::map<unsigned, VEC_DOUBLE> gradient_map;
    for (auto it : f.trees_weights.weights)
    {
        gradient_map.insert({it.first, std::vector<double>()});
        gradient_map[it.first].assign(it.second.size(), 0.0);
    }
    return gradient_map;
}

std::map<unsigned, int> generate_nodetype_map(Forest f)
{
    std::map<unsigned, int> node_type;
    for (auto it : f.trees)
    {
        node_type.insert(it.node_type.begin(), it.node_type.end());
    }
    return node_type;
}
// 向量值函数拟合

double forest_loss_mse(Forest &f, VEC_DOUBLE datax, VEC_DOUBLE datay)
{
    auto y_bar = run_forest(f, datax);
    return dist_2(y_bar, datay);
}

double forest_loss_mse_batch(Forest &f, std::vector<VEC_DOUBLE> datax, std::vector<VEC_DOUBLE> datay)
{
    double res = 0.0;
    for (int i = 0; i < datax.size(); i++)
    {
        res += dist_2(run_forest(f, datax[i]), datay[i]);
    }
    return res / double(datax.size());
}

std::map<unsigned, VEC_DOUBLE> forest_loss_mse_gradient(Forest &f, std::vector<VEC_DOUBLE> datax, std::vector<VEC_DOUBLE> datay)
{
    std::map<unsigned, VEC_DOUBLE> loss_gradient = generate_gradient_map(f);
    double m = double(datax.size());
    for (int i = 0; i < datax.size(); i++)
    {
        std::map<unsigned, double> x, x_bar;
        auto x_ = run_forest(f, datax[i]);
        for (int j = 0; j < f.fdim; j++)
        {
            x.insert({f.root_id[j], datax[i][j]});
            x_bar.insert({f.root_id[j], x_[j]});
        }
        gradient_forest(f, datax[i]);
        for (auto root_id : f.root_id)
        {
            for (auto it : loss_gradient)
            {
                vec_add(loss_gradient[it.first], vec_mul(f.trees_weights.dnode_dvariable[root_id][it.first], 2.0 * (x_bar[root_id] - x[root_id]) / m));
            }
        }
    }
    return loss_gradient;
}

void TRAIN_FOREST(Forest &f, MAP_GRADIENT &g)
{
    auto nodetype = generate_nodetype_map(f);
    for (auto it : g)
    {
        if (in_vector(f.trees_weights.untrainable, it.first))
        {
            continue;
        }
        else
        {
            vec_add(f.trees_weights.weights[it.first], vec_mul(g[it.first], -1.0));
            if (nodetype[it.first] == CT_OPS || nodetype[it.first] == CT_VAR || nodetype[it.first] == CT_WEIGHTED_PLUS || nodetype[it.first] == CT_WEIGHTED_PLUS_2)
            {
                onehot(f.trees_weights.weights[it.first]);
                if (norm(f.trees_weights.weights[it.first], '2') == 1.0)
                {
                    f.trees_weights.untrainable.push_back(it.first);
                    continue;
                }
            }
        }
    }
}

void PRINT_FOREST(Forest &f)
{
    for (int i = 0; i < f.fdim; i++)
    {
        std::cout << "dx" << i + 1 << "/dt = ";
        print_tree(f.trees[i], f.trees_weights);
        std::cout << std::endl;
    }
}