#include "wet.h"
#include "forest.h"
#include "tool.h"

Tree build_tree_bio1_IDBIAS(unsigned head_length, unsigned ID_BIAS)
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

Forest build_forest_bio1(unsigned f_dim, unsigned x_dim, unsigned head_length, std::pair<double, double> cr, int randseed)
{
    Forest f;
    f.fdim = f_dim;
    std::vector<Weights> trees_weights;
    std::vector<Progress> trees_progress;
    for (unsigned i = 0; i < f_dim; i++)
    {
        int ID_BIAS = i * 10000;
        f.trees.push_back(build_tree_bio1_IDBIAS(head_length, ID_BIAS));
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