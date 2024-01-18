#include "wet.h"
#include "tool.h"

extern calculation_type CAL_TYPE;
extern calculation_node_map CAL_NODE_MAP;

double penalty_func_2(VEC_DOUBLE x)
{
    double sum_ = 0.0;
    sum_ = 1.0 / std::pow(norm(x, '2'), 2) - 1.0;
    return sum_;
}

VEC_DOUBLE penalty_func_gradient_2(VEC_DOUBLE x)
{
    VEC_DOUBLE g = x;
    double sum_ = norm(x, '2');
    sum_ = sum_ * sum_;
    for (int i = 0; i < x.size(); i++)
    {
        g[i] = -2.0 * x[i] / sum_;
    }
    return g;
}

double penalty_func_3(VEC_DOUBLE x)
{
    auto x1 = x;
    VEC_DOUBLE p;
    p.assign(x1.size(), -1.0 / double(x1.size()));
    vec_add(x1, p);
    double sum_ = std::pow(norm(x1, '2'), 2);
    x1.assign(x1.size(), 0.0);
    x1[0] = 1.0;
    vec_add(x1, p);
    double sum_2 = std::pow(norm(x1, '2'), 2);
    return sum_2 - sum_;
}

VEC_DOUBLE penalty_func_gradient_3(VEC_DOUBLE x)
{
    VEC_DOUBLE g = x;
    VEC_DOUBLE p;
    p.assign(x.size(), -1.0 / double(x.size()));
    vec_add(g, p);
    vec_mul_inplace(g, -2.0);
    return g;
}

// 固定步长的更新
int tree_train_fixstep(Tree tree, Weights &ws, MAP_GRADIENT g, double step_length, double epsilon, bool IsPenalty, std::pair<double, double> constant_range)
{
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        if (in_vector(ws.untrainable, id))
        {
            continue;
        }
        VEC_DOUBLE orth_basis, m, g_, penalty_gradient;
        double wg_wg, wg_lg;
        double r;
        // feasiable direction
        if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            if (g[id].size() == 1)
            {
                continue;
            }
            if (norm(g[id], '2') == 1.0)
            {
                ws.untrainable.push_back(id);
                continue;
            }
            orth_basis.assign(g[id].size(), 1.0);
            normalize(g[id]);
            normalize(orth_basis);
            m = vec_mul(orth_basis, -inner_mul(orth_basis, g[id]));
            // grad_err = grad - (grad*orthB)[1] * orthB'
            g_ = g[id];
            vec_add(g_, m);
            penalty_gradient = penalty_func_gradient_3(ws.weights[id]);
            normalize(penalty_gradient);
            for (unsigned j = 0; j < g_.size(); j++)
            {
                if (g_[j] > 0.0 && ws.weights[id][j] <= 0.0)
                {
                    m.assign(g_.size(), 0.0);
                    m[j] = 1.0;
                    auto vt = vec_mul(orth_basis, -inner_mul(m, orth_basis));
                    vec_add(m, vt);
                    normalize(m);
                    // m 是与边界垂直的向量
                    auto m1 = vec_mul(m, -inner_mul(g_, m));
                    vec_add(g_, m1);
                    g_[j] = 0.0;
                    if (IsPenalty && CAL_NODE_MAP.node_isonehot[tree.node_type[id]])
                    {
                        auto m2 = vec_mul(m, -inner_mul(penalty_gradient, m));
                        vec_add(penalty_gradient, m2);
                        normalize(penalty_gradient);
                        if (inner_mul(penalty_gradient, g_) < 0.0)
                        {
                            vec_add(g_, vec_mul(penalty_gradient, -inner_mul(penalty_gradient, g_) + epsilon));
                        }
                    }
                }
            }
            if (IsPenalty && CAL_NODE_MAP.node_isonehot[tree.node_type[id]])
            {
                // penalty function p(w) = 1-Sigma(w_i^2)
                // dp/dw_i = -2 * w_i
                penalty_gradient = penalty_func_gradient_3(ws.weights[id]);
                normalize(penalty_gradient);
                // penalty_gradient = vec_mul(ws.weights[id], -2.0);
                // penalty_gradient = penalty_func_gradient_2(ws.weights[id]);
                m = vec_mul(orth_basis, -inner_mul(orth_basis, penalty_gradient));
                vec_add(penalty_gradient, m);
                wg_wg = inner_mul(penalty_gradient, penalty_gradient);
                wg_lg = inner_mul(penalty_gradient, g_);
                if (std::fabs(wg_wg) <= 1e-8)
                {
                    r = epsilon;
                }
                else
                {
                    r = std::fabs(std::min(wg_lg / wg_wg, 0.0)) + epsilon;
                }
                m = vec_mul(penalty_gradient, r);
                vec_add(g_, m);
            }
            normalize(g_);
        }
        else
        {
            g_ = g[id];
            normalize(g_);
            vec_mul_inplace(g_, constant_range.second - constant_range.first);
        }
        auto L = vec_mul(g_, -step_length);
        vec_add(ws.weights[id], L);
        if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_WEIGHTED_PLUS || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            onehot(ws.weights[id]);
        }
        if (tree.node_type[id] == CT_CON)
        {
            if (ws.weights[id][0] > constant_range.second)
            {
                ws.weights[id][0] = constant_range.second;
            }
            if (ws.weights[id][0] < constant_range.first)
            {
                ws.weights[id][0] = constant_range.first;
            }
            if (std::isnan(ws.weights[id][0]))
            {
                ws.weights[id][0] = constant_range.second;
            }
        }
    }
    return 0;
}

// 一维线性搜索
void tree_train_linearsearch(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> datax, VEC_DOUBLE datay, batch_loss_func loss_func, double epsilon, bool IsPenalty, std::pair<double, double> constant_range)
{
    VEC_DOUBLE orth_basis, m, g_, penalty_gradient;
    double wg_wg, wg_lg;
    double r = epsilon;
    double r0 = 0.0;
    for (unsigned i = 0; i < tree.node_names.size(); i++)
    {
        auto id = tree.node_names[i];
        if (in_vector(ws.untrainable, id))
        {
            continue;
        }
        auto g = tree_loss_mse_gradient_batch_2node(tree, ws, ps, id, datax, datay);
        // feasiable direction
        if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            if (g.size() == 1)
            {
                continue;
            }
            if (norm(ws.weights[id], '2') == 1.0)
            {
                ws.untrainable.push_back(id);
                continue;
            }
            orth_basis.assign(g.size(), 1.0);
            // normalize(g);
            normalize(orth_basis);
            m = vec_mul(orth_basis, -inner_mul(orth_basis, g));
            // grad_err = grad - (grad*orthB)[1] * orthB'
            g_ = g;
            vec_add(g_, m);
            // penalty function p(w) = 1-Sigma(w_i^2)
            // dp/dw_i = -2 * w_i
            // penalty_gradient = vec_mul(ws.weights[id], -2.0);
            penalty_gradient = penalty_func_gradient_3(ws.weights[id]);
            // normalize(penalty_gradient);
            for (unsigned j = 0; j < g_.size(); j++)
            {
                if (g_[j] > 0.0 && ws.weights[id][j] <= 0.0)
                {
                    m.assign(g_.size(), 0.0);
                    m[j] = 1.0;
                    auto vt = vec_mul(orth_basis, -inner_mul(m, orth_basis));
                    vec_add(m, vt);
                    normalize(m);
                    // m 是与边界垂直的向量
                    auto m1 = vec_mul(m, -inner_mul(g_, m));
                    vec_add(g_, m1);
                    g_[j] = 0.0;
                    if (IsPenalty && CAL_NODE_MAP.node_isonehot[tree.node_type[id]])
                    {
                        auto m2 = vec_mul(m, -inner_mul(penalty_gradient, m));
                        vec_add(penalty_gradient, m2);
                        // normalize(penalty_gradient);
                        auto im = -inner_mul(penalty_gradient, g_);
                        if (im < 0.0)
                        {
                            vec_add(g_, vec_mul(penalty_gradient, -im));
                            r0 += (-im);
                        }
                    }
                }
            }
            // penalty_gradient = penalty_func_gradient_2(ws.weights[id]);
            // normalize(penalty_gradient);
            if (IsPenalty && CAL_NODE_MAP.node_isonehot[tree.node_type[id]])
            {
                penalty_gradient = penalty_func_gradient_3(ws.weights[id]);
                // normalize(penalty_gradient);
                m = vec_mul(orth_basis, -inner_mul(orth_basis, penalty_gradient));
                vec_add(penalty_gradient, m);
                wg_wg = inner_mul(penalty_gradient, penalty_gradient);
                wg_lg = inner_mul(penalty_gradient, g_);
                if (std::fabs(wg_wg) <= 1e-8)
                {
                    r = 0.0;
                }
                else
                {
                    r = std::fabs(std::min(wg_lg / wg_wg, 0.0)) + epsilon;
                    r0 += r;
                }
                m = vec_mul(penalty_gradient, r);
                vec_add(g_, m);
            }
            normalize(g_);
        }
        else
        {
            g_ = g;
            normalize(g_);
            vec_mul_inplace(g_, constant_range.second - constant_range.first);
        }
        double ub = 1.0;
        double lb = 0.0;
        auto weight_save = ws.weights[id];
        while (ub - lb > 1e-4)
        {
            double mu1 = lb + 0.382 * (ub - lb);
            double mu2 = lb + 0.618 * (ub - lb);
            // mu1
            ws.weights[id] = weight_save;
            auto L = vec_mul(g_, -mu1);
            vec_add(ws.weights[id], L);
            if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_WEIGHTED_PLUS || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
            {
                onehot(ws.weights[id]);
            }
            if (tree.node_type[id] == CT_CON)
            {
                if (ws.weights[id][0] > constant_range.second)
                {
                    ws.weights[id][0] = constant_range.second;
                }
                if (ws.weights[id][0] < constant_range.first)
                {
                    ws.weights[id][0] = constant_range.first;
                }
                if (std::isnan(ws.weights[id][0]))
                {
                    ws.weights[id][0] = constant_range.second;
                }
            }
            double error1 = loss_func(tree, ws, ps, datax, datay) + r0 * penalty_func_3(ws.weights[id]);
            // mu2
            ws.weights[id] = weight_save;
            L = vec_mul(g_, -mu2);
            vec_add(ws.weights[id], L);
            if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_WEIGHTED_PLUS || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
            {
                onehot(ws.weights[id]);
            }
            if (tree.node_type[id] == CT_CON)
            {
                if (ws.weights[id][0] > constant_range.second)
                {
                    ws.weights[id][0] = constant_range.second;
                }
                if (ws.weights[id][0] < constant_range.first)
                {
                    ws.weights[id][0] = constant_range.first;
                }
                if (std::isnan(ws.weights[id][0]))
                {
                    ws.weights[id][0] = constant_range.second;
                }
            }
            double error2 = loss_func(tree, ws, ps, datax, datay) + r0 * penalty_func_3(ws.weights[id]);
            //
            if (error1 > error2)
                lb = mu1;
            else
                ub = mu2;
        }
        vec_copy(ws.weights[id], weight_save);
        auto L = vec_mul(g_, -lb);
        vec_add(ws.weights[id], L);
        if (tree.node_type[id] == CT_OPS || tree.node_type[id] == CT_VAR || tree.node_type[id] == CT_WEIGHTED_PLUS || tree.node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            onehot(ws.weights[id]);
        }
        if (tree.node_type[id] == CT_CON)
        {
            if (ws.weights[id][0] > constant_range.second)
            {
                ws.weights[id][0] = constant_range.second;
            }
            if (ws.weights[id][0] < constant_range.first)
            {
                ws.weights[id][0] = constant_range.first;
            }
            if (std::isnan(ws.weights[id][0]))
            {
                ws.weights[id][0] = constant_range.second;
            }
        }
    }
}

void TRAIN_TREE(Tree tree, Weights &ws, MAP_GRADIENT g)
{
    auto nodetype = tree.node_type;
    for (auto it : ws.weights)
    {
        if (in_vector(ws.untrainable, it.first))
        {
            continue;
        }else{
            vec_add(ws.weights[it.first], vec_mul(g[it.first], -1.0));
            if (nodetype[it.first] == CT_OPS || nodetype[it.first] == CT_VAR || nodetype[it.first] == CT_WEIGHTED_PLUS || nodetype[it.first] == CT_WEIGHTED_PLUS_2)
            {
                onehot(ws.weights[it.first]);
                if (norm(ws.weights[it.first], '2') == 1.0)
                {
                    ws.untrainable.push_back(it.first);
                    continue;
                }
            }
        }
    }
}