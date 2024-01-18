#include "wet.h"
#include "tool.h"

/*
    build tree
*/

Tree build_tree_standard(unsigned head_length)
{
    Tree tree;
    tree.node_names.clear();

    // connection
    for (unsigned int i = 0; i < head_length * 2 + 1; i++)
    {
        if (i < head_length)
        {
            tree.node_names.push_back(i);
            tree.operator_nodes.push_back(i);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i, {2 * i + 1, 2 * i + 2}));
            tree.node_type.insert(std::pair<unsigned, int>(i, CT_OPS));
        }
        else
        {
            tree.node_names.push_back(i);
            tree.operator_nodes.push_back(i);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i, {i + ID_N * 2, ID_N + i}));
            tree.node_type.insert(std::pair<unsigned, int>(i, CT_WEIGHTED_PLUS_2));

            tree.node_names.push_back(i + ID_N * 2);
            tree.terminal_nodes.push_back(i + ID_N * 2);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N * 2, {}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N * 2, CT_CON));

            tree.node_names.push_back(i + ID_N);
            tree.terminal_nodes.push_back(i + ID_N);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N, {i + ID_N * 3, i + ID_N * 4}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N, CT_MUL2));

            tree.node_names.push_back(i + ID_N * 3);
            tree.terminal_nodes.push_back(i + ID_N * 3);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N * 3, {}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N * 3, CT_CON));

            tree.node_names.push_back(i + ID_N * 4);
            tree.terminal_nodes.push_back(i + ID_N * 4);
            tree.connection_down.insert(std::pair<unsigned, VEC_ID>(i + ID_N * 4, {}));
            tree.node_type.insert(std::pair<unsigned, int>(i + ID_N * 4, CT_VAR));
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
    tree.layer_nodes.push_back({0});
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
