#include "wet.h"
#include "tool.h"
#include <random>
/*
    extern operators
*/
extern TreeOpSet TreeOps;
extern calculation_type CAL_TYPE;
extern calculation_node_map CAL_NODE_MAP;

double run_tree(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input)
{
    for (int i = tree.layer_nodes.size() - 1; i >= 0; i--)
    {
        for (unsigned j = 0; j < tree.layer_nodes[i].size(); j++)
        {
            run_node(ps, ws, tree.layer_nodes[i][j], tree.connection_down[tree.layer_nodes[i][j]], input, tree.node_type[tree.layer_nodes[i][j]]);
        }
    }
    return ps.node_values[0][0];
}

void run_node(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input, int type)
{
    CAL_NODE_MAP(type)
    (ps, ws, node_id, conn, input);
}

void gradient_tree(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input)
{
    // After run_tree
    // From bottom to top
    for (int i = tree.layer_nodes.size() - 1; i >= 0; i--)
    {
        for (unsigned j = 0; j < tree.layer_nodes[i].size(); j++)
        {
            unsigned id = tree.layer_nodes[i][j];
            for (unsigned k = 0; k < tree.connection_gradient[id].size(); k++)
            {
                gradient_node(ps, ws, id, tree.connection_gradient[id][k], tree.connection_down[id], input, tree.node_type[id]);
            }
        }
    }
}

void gradient_node(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input, int type)
{
    CAL_NODE_MAP[type](ps, ws, node_id, weight_id, conn, input);
}

void gradient_tree_top2node(Tree tree, Weights &ws, Progress &ps, unsigned node_id, VEC_DOUBLE input)
{
    // After run_tree
    // From bottom to top
    for (int i = tree.layer_nodes.size() - 1; i >= 0; i--)
    {
        for (unsigned j = 0; j < tree.layer_nodes[i].size(); j++)
        {
            unsigned id = tree.layer_nodes[i][j];
            if (in_vector(tree.connection_gradient[id], node_id))
            {
                gradient_node(ps, ws, id, node_id, tree.connection_down[id], input, tree.node_type[id]);
            }
        }
    }
}

Weights generate_weights_from_tree(Tree tree, int input_dim, int init_method, std::pair<double, double> constant_range, int randseed)
{
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(constant_range.first, constant_range.second); // 左闭右闭区间
    e.seed(randseed);
    Weights ws;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        ws.weights.insert(std::pair<unsigned, VEC_DOUBLE>(id, std::vector<double>()));
        ws.weights[id].assign(CAL_NODE_MAP.node_weight_dim[tree.node_type[id]], u(e));
        if (tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            if (init_method == 1)
            {
                vec_average_init(ws.weights[id]);
            }
            if (init_method == 2)
            {
                vec_rand_init(ws.weights[id], randseed);
            }
        }
    }
    ws.weights.insert(std::pair<unsigned, VEC_DOUBLE>(ID_N, std::vector<double>()));
    ws.weights[ID_N].assign(input_dim, 0.0);
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        ws.dnode_dvariable.insert(std::pair<unsigned, std::map<unsigned, VEC_DOUBLE>>(id, std::map<unsigned, VEC_DOUBLE>()));
        for (unsigned j = 0; j < tree.connection_gradient[id].size(); j++)
        {
            ws.dnode_dvariable[id].insert(std::pair<unsigned, VEC_DOUBLE>(tree.connection_gradient[id][j], std::vector<double>()));
            ws.dnode_dvariable[id][tree.connection_gradient[id][j]].assign(ws.weights[tree.connection_gradient[id][j]].size(), 0.0);
        }
        ws.dnode_dvariable[id].insert(std::pair<unsigned, VEC_DOUBLE>(ID_N, std::vector<double>()));
        ws.dnode_dvariable[id][ID_N].assign(input_dim, 0.0);
    }

    return ws;
}

Weights *generate_weights_pointer_from_tree(Tree tree, int input_dim, int init_method, std::pair<double, double> constant_range, int randseed)
{
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(constant_range.first, constant_range.second); // 左闭右闭区间
    e.seed(randseed);
    Weights *ws = new Weights;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        ws->weights.insert(std::pair<unsigned, VEC_DOUBLE>(id, std::vector<double>()));
        ws->weights[id].assign(CAL_NODE_MAP.node_weight_dim[tree.node_type[id]], u(e));
        if (tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            if (init_method == 1)
            {
                vec_average_init(ws->weights[id]);
            }
            if (init_method == 2)
            {
                vec_rand_init(ws->weights[id], randseed);
            }
        }
    }
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        ws->dnode_dvariable.insert(std::pair<unsigned, std::map<unsigned, VEC_DOUBLE>>(id, std::map<unsigned, VEC_DOUBLE>()));
        for (unsigned j = 0; j < tree.connection_gradient[id].size(); j++)
        {
            ws->dnode_dvariable[id].insert(std::pair<unsigned, VEC_DOUBLE>(tree.connection_gradient[id][j], std::vector<double>()));
            ws->dnode_dvariable[id][tree.connection_gradient[id][j]].assign(ws->weights[tree.connection_gradient[id][j]].size(), 0.0);
        }
        ws->dnode_dvariable[id].insert(std::pair<unsigned, VEC_DOUBLE>(ID_N, std::vector<double>()));
        ws->dnode_dvariable[id][ID_N].assign(input_dim, 0.0);
    }

    return ws;
}

Progress generate_progress_from_tree(Tree tree)
{
    Progress ps;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        ps.node_values.insert(std::pair<unsigned, VEC_DOUBLE>(tree.node_names[i], {0.0}));
    }
    return ps;
}

Progress *generate_progress_pointer_from_tree(Tree tree)
{
    Progress *ps = new Progress;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        ps->node_values.insert(std::pair<unsigned, VEC_DOUBLE>(tree.node_names[i], {0.0}));
    }
    return ps;
}

Weights merge_weights(Weights w1, Weights w2)
{
    Weights new_w;
    new_w = w1;
    for (auto it : w2.weights)
    {
        if (new_w.weights.contains(it.first))
        {
            new_w.dnode_dvariable[it.first].insert(w2.dnode_dvariable[it.first].begin(), w2.dnode_dvariable[it.first].end());
        }
        else
        {
            new_w.weights.insert(it);
            new_w.dnode_dvariable.insert(std::pair<unsigned, std::map<unsigned, VEC_DOUBLE>>(it.first, w2.dnode_dvariable[it.first]));
        }
    }
    new_w.untrainable.resize(new_w.weights.size());
    std::merge(w1.untrainable.begin(), w1.untrainable.end(), w2.untrainable.begin(), w2.untrainable.end(), new_w.untrainable.begin());
    return new_w;
}

Progress merge_progress(Progress g1, Progress g2)
{
    Progress new_p;
    new_p.node_values = g1.node_values;
    new_p.node_values.insert(g2.node_values.begin(), g2.node_values.end());
    return new_p;
}

std::map<unsigned, VEC_DOUBLE> generate_gradient_map(Weights weights)
{
    std::map<unsigned, VEC_DOUBLE> gradient_map;
    for (auto it : weights.weights)
    {
        gradient_map.insert({it.first, std::vector<double>()});
        gradient_map[it.first].assign(it.second.size(), 0.0);
    }
    return gradient_map;
}