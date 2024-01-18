#include "optimizer.h"

extern calculation_type CAL_TYPE;
extern calculation_node_map CAL_NODE_MAP;

// Momentum
void Momentum(MAP_GRADIENT &v, MAP_GRADIENT &delta, double eta, double alpha)
{
    // v <-- eta * v + alpha * delta
    for (auto it : v)
    {
        vec_mul_inplace(v[it.first], eta);
        vec_add(v[it.first], vec_mul(delta[it.first], alpha));
    }
}

// Adam

VEC_DOUBLE vec_dot(VEC_DOUBLE x, VEC_DOUBLE y)
{
    auto z = x;
    for (int i = 0; i < y.size(); i++)
    {
        z[i] *= y[i];
    }
    return z;
}
VEC_DOUBLE Adam_g(VEC_DOUBLE v, VEC_DOUBLE s, int t, double beta1, double beta2, double eps)
{
    VEC_DOUBLE g = v;
    for (int i = 0; i < v.size(); i++)
    {
        g[i] = v[i] / (1 - std::pow(beta1, double(t))+ eps) / (std::sqrt(s[i] / (1 - std::pow(beta2, double(t)))) + eps);
    }
    return g;
}

void Adam(MAP_GRADIENT &v, MAP_GRADIENT &s, MAP_GRADIENT &delta, MAP_GRADIENT &g, int t, double beta1, double beta2, double eps)
{
    /*
        v <- beta1 * v + (1-beta1) * delta
        s <- beta2 * s + (1-beta2) * delta^2
        g <- v/(1-beta1^t) / sqrt( s/(1-beta2^t)+eps )
    */
    for (auto it : v)
    {
        vec_mul_inplace(v[it.first], beta1);
        vec_add(v[it.first], vec_mul(delta[it.first], 1 - beta1));
        vec_mul_inplace(s[it.first], beta2);
        vec_add(s[it.first], vec_mul(vec_dot(delta[it.first], delta[it.first]), 1 - beta2));
        g[it.first] = Adam_g(v[it.first], s[it.first], t, beta1, beta2, eps);
    }
}

void normalize_grad(MAP_GRADIENT &g, std::map<unsigned, int> node_type)
{
    for (auto it : g)
    {
        if (std::isnan(norm(it.second, '1')))
        {
            g[it.first].assign(g[it.first].size(), 0.0);
        }
        if (node_type[it.first] == CT_OPS || node_type[it.first] == CT_VAR || node_type[it.first] == CT_WEIGHTED_PLUS_2)
        {
            vec_mul_inplace(g[it.first], 1.0 / (norm(g[it.first], '2') + 1e-6));
        }
    }
}

void add_grad(MAP_GRADIENT &g, MAP_GRADIENT src)
{
    for (auto it : src)
    {
        vec_add(g[it.first], src[it.first]);
    }
}

void make_feasible_grad(MAP_GRADIENT &g, std::map<unsigned, int> node_type, std::map<unsigned, VEC_DOUBLE> weights, std::pair<double, double> cr)
{
    for (auto it : g)
    {
        if (node_type[it.first] == CT_OPS || node_type[it.first] == CT_VAR || node_type[it.first] == CT_WEIGHTED_PLUS_2)
        {
            VEC_DOUBLE orth_basis, m;
            orth_basis.assign(it.second.size(), 1.0);
            normalize(orth_basis);
            normalize(g[it.first]);
            m = vec_mul(orth_basis, -inner_mul(orth_basis, g[it.first]));
            // grad_err = grad - (grad*orthB)[1] * orthB'

            vec_add(g[it.first], m);
            for (unsigned j = 0; j < g[it.first].size(); j++)
            {
                if (g[it.first][j] > 0.0 && weights[it.first][j] <= 0.0)
                {
                    m.assign(g[it.first].size(), 0.0);
                    m[j] = 1.0;
                    auto vt = vec_mul(orth_basis, -inner_mul(m, orth_basis));
                    vec_add(m, vt);
                    normalize(m);
                    // m 是与边界垂直的向量
                    auto m1 = vec_mul(m, -inner_mul(g[it.first], m));
                    vec_add(g[it.first], m1);
                    g[it.first][j] = 0.0;
                }
            }
            vec_mul_inplace(g[it.first], 0.01);
        }
        if (node_type[it.first] == CT_CON)
        {
            if (weights[it.first][0] - it.second[0] > cr.second)
            {
                g[it.first][0] = (weights[it.first][0] - cr.second) * 0.9;
            }
            if (weights[it.first][0] - it.second[0] < cr.first)
            {
                g[it.first][0] = (weights[it.first][0] - cr.first) * 0.9;
            }
        }
    }
}

void adaptive_penalty_grad(MAP_GRADIENT &g, std::map<unsigned, int> node_type, std::map<unsigned, VEC_DOUBLE> weights, VEC_DOUBLE(p_func_g)(VEC_DOUBLE x), double eps)
{
    VEC_DOUBLE orth_basis, penalty_gradient, m;
    double wg_wg, wg_lg;
    double r;
    for (auto it : weights)
    {
        if (node_type[it.first] == CT_OPS || node_type[it.first] == CT_VAR || node_type[it.first] == CT_WEIGHTED_PLUS_2)
        {
            penalty_gradient = p_func_g(weights[it.first]);
            normalize(penalty_gradient);
            orth_basis.assign(it.second.size(), 1.0);
            normalize(orth_basis);
            m = vec_mul(orth_basis, -inner_mul(orth_basis, penalty_gradient));
            vec_add(penalty_gradient, m);
            wg_wg = inner_mul(penalty_gradient, penalty_gradient);
            wg_lg = inner_mul(penalty_gradient, g[it.first]);
            if (std::fabs(wg_wg) <= 1e-8)
            {
                r = eps;
            }
            else
            {
                r = std::fabs(std::min(wg_lg / wg_wg, 0.0)) + eps;
            }
            m = vec_mul(penalty_gradient, r);
            vec_add(g[it.first], m);
        }
    }
}

MAP_GRADIENT generate_penalty_grad(std::map<unsigned, VEC_DOUBLE> weights, std::map<unsigned, int> node_type, VEC_DOUBLE(p_func_g)(VEC_DOUBLE x), double gamma)
{
    std::map<unsigned, VEC_DOUBLE> penalty_grad;
    for (auto it : weights)
    {
        if (node_type[it.first] == CT_OPS || node_type[it.first] == CT_VAR || node_type[it.first] == CT_WEIGHTED_PLUS_2)
        {
            penalty_grad.insert({it.first, std::vector<double>(p_func_g(weights[it.first]))});
            vec_mul_inplace(penalty_grad[it.first], gamma);
        }
        else
        {
            penalty_grad.insert({it.first, std::vector<double>(zeros_as(weights[it.first]))});
        }
    }
    return penalty_grad;
}
