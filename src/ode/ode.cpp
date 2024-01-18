#include "wet.h"
#include "tool.h"
#include "forest.h"
#include "ode.h"
#include <algorithm>

extern calculation_type CAL_TYPE;
extern calculation_node_map CAL_NODE_MAP;
/*
    先实现Runge-Kutta格式求解方程
*/
// declare
odeSolution odeSolve_RK45(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat);
odeSolution odeSolve_RK45_withgrad(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat);
odeSolution odeSolve_Euler_withgrad(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat);
odeSolution odeSolve_Euler(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat);
//
odeSolution Solve(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat, std::string method)
{
    if (method == "RK45")
    {
        return odeSolve_RK45(f, x0, tspan, dt, saveat);
    }
    if (method == "Euler"){
        return odeSolve_Euler(f, x0, tspan, dt, saveat);
    }
    return odeSolution();
}

odeSolution Solve_withgrad(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat, std::string method)
{
    if (method == "RK45")
    {
        return odeSolve_RK45_withgrad(f, x0, tspan, dt, saveat);
    }
    if (method== "Euler"){
        return odeSolve_Euler_withgrad(f, x0, tspan, dt, saveat);
    }
    return odeSolution();
}

odeSolution odeSolve_RK45(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat)
{
    if (dt < 0)
    { // 没指定时间间隔，计算100步
        dt = (tspan.second - tspan.first) / 100.0;
    }
    int N = int(floor((tspan.second - tspan.first) / dt));

    VEC_DOUBLE time_point; // 用于计算的时间点，与保存数据的时间点不同 保证精度的最低要求
    for (int i = 0; i < N; i++)
    {
        time_point.push_back(double(i) * dt + tspan.first);
    }
    if (time_point[N - 1] < tspan.second)
    {
        time_point.push_back(tspan.second);
    }
    VEC_DOUBLE tp_;
    tp_.resize(time_point.size() + saveat.size());

    if (saveat.size() == 0)
    {
        saveat = time_point;
        tp_ = time_point;
    }
    else
    {
        // 将time_point和saveat合并 并且排序
        std::merge(time_point.begin(), time_point.end(), saveat.begin(), saveat.end(), tp_.begin());
        std::sort(tp_.begin(), tp_.end());
        tp_.erase(std::unique(tp_.begin(), tp_.end()), tp_.end());
    }

    odeSolution sol;
    sol.t.push_back(tp_[0]);
    sol.u.push_back(VEC_DOUBLE());
    sol.u[0] = x0;
    VEC_DOUBLE xc;
    xc = sol.u[0];
    for (int i = 0; i < int(tp_.size() - 1); i++)
    {
        auto DT = tp_[i + 1] - tp_[i];
        auto K1 = run_forest(f, xc);
        auto x2 = vec_add_r(xc, vec_mul(K1, DT / 2));
        auto K2 = run_forest(f, x2);
        auto x3 = vec_add_r(xc, vec_mul(K2, DT / 2));
        auto K3 = run_forest(f, x3);
        auto x4 = vec_add_r(xc, vec_mul(K3, DT));
        auto K4 = run_forest(f, x4);

        vec_add(xc, vec_mul(K1, DT / 6.0));
        vec_add(xc, vec_mul(K2, 2.0 * DT / 6.0));
        vec_add(xc, vec_mul(K3, 2.0 * DT / 6.0));
        vec_add(xc, vec_mul(K4, DT / 6.0));
        // save data
        if (std::find(saveat.begin(), saveat.end(), tp_[i + 1]) != std::end(saveat))
        {
            sol.u.push_back(VEC_DOUBLE(xc));
            // sol.u[i + 1].assign(f.dim, 0.0);
            // vec_add(sol.u[i + 1], xc);
            sol.t.push_back(tp_[i + 1]);
        }
    }

    return sol;
}

odeSolution odeSolve_RK45_withgrad(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat)
{
    if (dt < 0)
    { // 没指定时间间隔，计算100步
        dt = (tspan.second - tspan.first) / 100.0;
    }
    if ((tspan.second - tspan.first)<dt){
        dt = (tspan.second - tspan.first);
    }
    int N = int(floor((tspan.second - tspan.first) / dt));

    VEC_DOUBLE time_point; // 用于计算的时间点，与保存数据的时间点不同 保证精度的最低要求
    for (int i = 0; i < N; i++)
    {
        time_point.push_back(double(i) * dt + tspan.first);
    }
    if (time_point[N - 1] < tspan.second)
    {
        time_point.push_back(tspan.second);
    }
    VEC_DOUBLE tp_;
    tp_.resize(time_point.size() + saveat.size());

    if (saveat.size() == 0)
    {
        saveat = time_point;
        tp_ = time_point;
    }
    else
    {
        // 将time_point和saveat合并 并且排序
        std::merge(time_point.begin(), time_point.end(), saveat.begin(), saveat.end(), tp_.begin());
        std::sort(tp_.begin(), tp_.end());
        tp_.erase(std::unique(tp_.begin(), tp_.end()), tp_.end());
    }

    odeSolution sol;
    sol.t.push_back(tp_[0]);
    sol.u.push_back(VEC_DOUBLE());
    sol.u[0] = x0;
    VEC_DOUBLE xc;
    xc = sol.u[0];

    // 给grad分配内存
    // 每个time_point的梯度都要保存
    // 在每个计算过程中所有K的梯度都要保存
    // 不要直接用dnode的值，x和K要对所有参数求导数，不仅仅是自身函数的
    std::vector<odeGradient> time_point_grad;
    std::vector<odeGradient> K_grad;

    auto ROOT_ID = f.root_id;
    // K_grad
    for (int i = 0; i < 4; i++)
    {
        K_grad.push_back(odeGradient());
        K_grad[i].dxdv = generate_dxdv_fromForest(f);
    }
    // time_point
    for (long unsigned int i = 0; i < tp_.size(); i++)
    {
        time_point_grad.push_back(odeGradient());
        time_point_grad[i].dxdv = generate_dxdv_fromForest(f);
    }
    sol.grad.clear();
    // initial dx0/dt = 0

    for (int i = 0; i < int(tp_.size() - 1); i++)
    {
        // get d(x_i)/dw

        auto DT = tp_[i + 1] - tp_[i];
        auto K1 = run_forest(f, xc);
        gradient_forest(f, xc);
        // gradient of K1
        K_grad[0].dxdv = generate_dxdv_fromForest(f);
        for (long unsigned int j = 0; j < ROOT_ID.size(); j++)
        {
            auto R_ID = ROOT_ID[j];
            std::map<unsigned, double> dfdx;
            for (long unsigned int k = 0; k < ROOT_ID.size(); k++)
            {
                dfdx.insert({ROOT_ID[k], f.trees_weights.dnode_dvariable[R_ID][ID_N][k]});
            }
            for (auto it : time_point_grad[i].dxdv)
            {
                for (auto it2 : it.second)
                {
                    // df/dx * dx/dw
                    vec_add(K_grad[0].dxdv[R_ID][it2.first], vec_mul(it2.second, dfdx[it.first]));
                }
            }
            for (auto it : f.trees_weights.dnode_dvariable[R_ID])
            {
                vec_add(K_grad[0].dxdv[R_ID][it.first], it.second);
            }
        }

        auto x2 = vec_add_r(xc, vec_mul(K1, DT / 2));
        auto K2 = run_forest(f, x2);
        gradient_forest(f, x2);
        // gradient of K2
        K_grad[1].dxdv = generate_dxdv_fromForest(f);
        for (long unsigned int j = 0; j < ROOT_ID.size(); j++)
        {
            auto R_ID = ROOT_ID[j];
            std::map<unsigned, double> dfdx;
            for (long unsigned int k = 0; k < ROOT_ID.size(); k++)
            {
                dfdx.insert({ROOT_ID[k], f.trees_weights.dnode_dvariable[R_ID][ID_N][k]});
            }
            for (auto it : time_point_grad[i].dxdv)
            {
                for (auto it2 : it.second)
                {
                    // df/dx * dx/dw
                    vec_add(K_grad[1].dxdv[R_ID][it2.first], vec_mul(vec_add_r(it2.second, vec_mul(K_grad[0].dxdv[R_ID][it2.first], DT / 2.0)), dfdx[it.first]));
                }
            }
            for (auto it : f.trees_weights.dnode_dvariable[R_ID])
            {
                vec_add(K_grad[1].dxdv[R_ID][it.first], it.second);
            }
        }

        auto x3 = vec_add_r(xc, vec_mul(K2, DT / 2));
        auto K3 = run_forest(f, x3);
        gradient_forest(f, x3);
        K_grad[2].dxdv = generate_dxdv_fromForest(f);
        for (long unsigned int j = 0; j < ROOT_ID.size(); j++)
        {
            auto R_ID = ROOT_ID[j];
            std::map<unsigned, double> dfdx;
            for (long unsigned int k = 0; k < ROOT_ID.size(); k++)
            {
                dfdx.insert({ROOT_ID[k], f.trees_weights.dnode_dvariable[R_ID][ID_N][k]});
            }
            for (auto it : time_point_grad[i].dxdv)
            {
                for (auto it2 : it.second)
                {
                    // df/dx * dx/dw
                    vec_add(K_grad[2].dxdv[R_ID][it2.first], vec_mul(vec_add_r(it2.second, vec_mul(K_grad[1].dxdv[R_ID][it2.first], DT / 2.0)), dfdx[it.first]));
                }
            }
            for (auto it : f.trees_weights.dnode_dvariable[R_ID])
            {
                vec_add(K_grad[2].dxdv[R_ID][it.first], it.second);
            }
        }

        auto x4 = vec_add_r(xc, vec_mul(K3, DT));
        auto K4 = run_forest(f, x4);
        gradient_forest(f, x3);
        K_grad[3].dxdv = generate_dxdv_fromForest(f);
        for (long unsigned int j = 0; j < ROOT_ID.size(); j++)
        {
            auto R_ID = ROOT_ID[j];
            std::map<unsigned, double> dfdx;
            for (long unsigned int k = 0; k < ROOT_ID.size(); k++)
            {
                dfdx.insert({ROOT_ID[k], f.trees_weights.dnode_dvariable[R_ID][ID_N][k]});
            }
            for (auto it : time_point_grad[i].dxdv)
            {
                for (auto it2 : it.second)
                {
                    // df/dx * dx/dw
                    vec_add(K_grad[3].dxdv[R_ID][it2.first], vec_mul(vec_add_r(it2.second, vec_mul(K_grad[2].dxdv[R_ID][it2.first], DT / 2.0)), dfdx[it.first]));
                }
            }
            for (auto it : f.trees_weights.dnode_dvariable[R_ID])
            {
                vec_add(K_grad[3].dxdv[R_ID][it.first], it.second);
            }
        }
        vec_add(xc, vec_mul(K1, DT / 6.0));
        vec_add(xc, vec_mul(K2, 2.0 * DT / 6.0));
        vec_add(xc, vec_mul(K3, 2.0 * DT / 6.0));
        vec_add(xc, vec_mul(K4, DT / 6.0));

        for (auto it : time_point_grad[i + 1].dxdv)
        {
            // x_it
            for (auto it2 : it.second)
            {
                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], time_point_grad[i].dxdv[it.first][it2.first]);

                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], vec_mul(K_grad[0].dxdv[it.first][it2.first], DT / 6.0));
                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], vec_mul(K_grad[1].dxdv[it.first][it2.first], 2.0 * DT / 6.0));
                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], vec_mul(K_grad[2].dxdv[it.first][it2.first], 2.0 * DT / 6.0));
                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], vec_mul(K_grad[3].dxdv[it.first][it2.first], DT / 6.0));
            }
        }

        // save data
        if (std::find(saveat.begin(), saveat.end(), tp_[i + 1]) != std::end(saveat))
        {
            sol.u.push_back(VEC_DOUBLE(xc));
            sol.t.push_back(tp_[i + 1]);
            sol.grad.push_back(odeGradient());
            sol.grad[sol.grad.size() - 1].dxdv = time_point_grad[i + 1].dxdv;
        }
    }

    return sol;
}

std::map<unsigned, std::map<unsigned, VEC_DOUBLE>> generate_dxdv_fromForest(Forest &f)
{
    // 根据forest的信息分配内存，每个维度的x要对所有参数求导.
    std::map<unsigned, std::map<unsigned, VEC_DOUBLE>> dxdv;
    for (long unsigned int i = 0; i < f.fdim; i++)
    {
        dxdv.insert({f.root_id[i], std::map<unsigned, VEC_DOUBLE>()});
        for (auto it : f.trees_weights.weights)
        {
            dxdv[f.root_id[i]].insert({it.first, VEC_DOUBLE()});
            dxdv[f.root_id[i]][it.first].assign(it.second.size(), 0.0);
        }
    }
    return dxdv;
}

double trees_loss_mse(Forest &f, VEC_DOUBLE time, std::vector<VEC_DOUBLE> x, double dt, std::string method)
{
    std::pair<double, double> tspan;
    tspan.first = time[0];
    tspan.second = time[time.size() - 1];
    auto sol = Solve(f, x[0], tspan, dt, time, method);
    double res = 0.0;
    int count = 0;
    for (int i = 0; i < time.size() - 1; i++)
    {
        double r = dist_2(sol.u[i + 1], x[i + 1]);
        if (!std::isnan(r) && !std::isinf(r)){
            res += r;
            count++;
        }
    }
    res /= double(time.size() - 1);
    printf("Error: %f, Status: %d/%d\n",res,count,time.size()-1);
    return res;
}

std::map<unsigned, VEC_DOUBLE> trees_loss_mse_gradient(Forest &f, VEC_DOUBLE time, std::vector<VEC_DOUBLE> x, double dt, std::string method)
{
    std::map<unsigned, VEC_DOUBLE> loss_gradient = generate_gradient_map(f);
    std::pair<double, double> tspan;
    tspan.first = time[0];
    tspan.second = time[time.size() - 1];
    auto sol = Solve_withgrad(f, x[0], tspan, dt, time, method);

    for (int i = 0; i < time.size() - 1; i++)
    {
        std::map<unsigned, double> data, data_;

        for (int j = 0; j < f.fdim; j++)
        {
            data.insert({f.root_id[j], x[i + 1][j]});
            data_.insert({f.root_id[j], sol.u[i + 1][j]});
        }
        for (auto it : sol.grad[i].dxdv)
        {
            for (auto it2 : it.second)
            {
                // vec_add(loss_gradient[it2.first], vec_mul(it2.second, 2.0 * (data[it.first] - data_[it.first]) / double(time.size() - 1)));
                auto g_ = vec_mul(it2.second, 2.0 * (data_[it.first] - data[it.first]) / double(time.size() - 1));
                if (std::isnan(norm(g_, '2')) || std::isinf(norm(g_, '2')))
                {
                    break;
                }
                else
                {
                    vec_add(loss_gradient[it2.first], g_);
                }
            }
        }
    }
    return loss_gradient;
}

std::map<unsigned, VEC_DOUBLE> trees_loss_search_mse_gradient(Forest &f, VEC_DOUBLE time, std::vector<VEC_DOUBLE> x, double dt,double gamma,std::vector<Spline> &interps, std::string method){
    std::map<unsigned, VEC_DOUBLE> loss_gradient = generate_gradient_map(f);

    for (int kk=0;kk<time.size()-1;kk++) {

        std::vector<double> loss_weight;
        std::vector<std::vector<double>> traindata_x;
        std::vector<double> traindata_t;

        auto time_step = linspace(time[kk], time[kk] + (time[kk + 1] - time[kk]) * gamma, dt);
        std::vector<std::vector<double>> interp_data;
        for (int p = 0; p < f.fdim; p++) {
            interp_data.push_back(interps[p].f(time_step));
        }
        for (int p = 0; p < time_step.size(); p++) {
            std::vector<double> stx;
            for (int k = 0; k < f.fdim; k++) {
                stx.push_back(interp_data[k][p]);
            }
            traindata_x.push_back(VEC_DOUBLE(stx));
            loss_weight.push_back(0.5 + 0.5 * (time_step[p] - time[kk]) / (time[kk + 1] - time[kk]));
        }

        std::pair<double, double> tspan;
        tspan.first = time_step[0];
        tspan.second = time_step[time_step.size() - 1];
        auto sol = Solve_withgrad(f, traindata_x[0], tspan, dt, time_step, method);

        for (int i = 0; i < time_step.size() - 1; i++)
        {
            std::map<unsigned, double> data, data_;

            for (int j = 0; j < f.fdim; j++)
            {
                data.insert({f.root_id[j], traindata_x[i + 1][j]});
                data_.insert({f.root_id[j], sol.u[i + 1][j]});
            }
            for (auto it : sol.grad[i].dxdv)
            {
                for (auto it2 : it.second)
                {
                    // vec_add(loss_gradient[it2.first], vec_mul(it2.second, 2.0 * (data[it.first] - data_[it.first]) / double(time.size() - 1)));
                    auto g_ = vec_mul(it2.second, loss_weight[i]* 2.0 * (data_[it.first] - data[it.first]) / double(time.size() - 1));
                    if (std::isnan(norm(g_, '2')) || std::isinf(norm(g_, '2')))
                    {
                        break;
                    }
                    else
                    {
                        vec_add(loss_gradient[it2.first], g_);
                    }
                }
            }
        }
    }
    return loss_gradient;
}

int trees_train_fixstep(Forest &f, MAP_GRADIENT g, double step_length, double epsilon, bool IsPenalty, std::pair<double, double> constant_range)
{
    std::vector<unsigned> ids;
    for (auto it : f.trees_weights.weights)
    {
        ids.push_back(it.first);
    }
    int s = ids.size();
    std::map<unsigned, int> node_type;

    for (auto it : f.trees)
    {
        node_type.insert(it.node_type.begin(), it.node_type.end());
    }

    for (unsigned i = 0; i < s; i++)
    {
        auto id = ids[i];
        if (in_vector(f.trees_weights.untrainable, id))
        {
            continue;
        }
        VEC_DOUBLE orth_basis, m, g_, penalty_gradient;
        double wg_wg, wg_lg;
        double r;
        // feasiable direction
        if (node_type[id] == CT_OPS || node_type[id] == CT_VAR || node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            if (g[id].size() == 1)
            {
                continue;
            }
            if (norm(g[id], '2') == 1.0)
            {
                f.trees_weights.untrainable.push_back(id);
                continue;
            }
            orth_basis.assign(g[id].size(), 1.0);
            normalize(g[id]);
            normalize(orth_basis);
            m = vec_mul(orth_basis, -inner_mul(orth_basis, g[id]));
            // grad_err = grad - (grad*orthB)[1] * orthB'
            g_ = g[id];
            vec_add(g_, m);
            penalty_gradient = penalty_func_gradient_3(f.trees_weights.weights[id]);
            normalize(penalty_gradient);
            for (unsigned j = 0; j < g_.size(); j++)
            {
                if (g_[j] > 0.0 && f.trees_weights.weights[id][j] <= 0.0)
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
                    if (IsPenalty && CAL_NODE_MAP.node_isonehot[node_type[id]])
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
            if (IsPenalty && CAL_NODE_MAP.node_isonehot[node_type[id]])
            {
                // penalty function p(w) = 1-Sigma(w_i^2)
                // dp/dw_i = -2 * w_i
                penalty_gradient = penalty_func_gradient_3(f.trees_weights.weights[id]);
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
        vec_add(f.trees_weights.weights[id], L);
        if (node_type[id] == CT_OPS || node_type[id] == CT_VAR || node_type[id] == CT_WEIGHTED_PLUS || node_type[id] == CT_WEIGHTED_PLUS_2)
        {
            onehot(f.trees_weights.weights[id]);
        }
        if (node_type[id] == CT_CON)
        {
            if (f.trees_weights.weights[id][0] > constant_range.second)
            {
                f.trees_weights.weights[id][0] = constant_range.second;
            }
            if (f.trees_weights.weights[id][0] < constant_range.first)
            {
                f.trees_weights.weights[id][0] = constant_range.first;
            }
            if (std::isnan(f.trees_weights.weights[id][0]))
            {
                f.trees_weights.weights[id][0] = constant_range.second;
            }
        }
    }
    return 0;
}

odeSolution odeSolve_Euler(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat)
{
    if (dt < 0)
    { // 没指定时间间隔，计算100步
        dt = (tspan.second - tspan.first) / 100.0;
    }
    int N = int(floor((tspan.second - tspan.first) / dt));

    VEC_DOUBLE time_point; // 用于计算的时间点，与保存数据的时间点不同 保证精度的最低要求
    for (int i = 0; i < N; i++)
    {
        time_point.push_back(double(i) * dt + tspan.first);
    }
    if (time_point[N - 1] < tspan.second)
    {
        time_point.push_back(tspan.second);
    }
    VEC_DOUBLE tp_;
    tp_.resize(time_point.size() + saveat.size());

    if (saveat.size() == 0)
    {
        saveat = time_point;
        tp_ = time_point;
    }
    else
    {
        // 将time_point和saveat合并 并且排序
        std::merge(time_point.begin(), time_point.end(), saveat.begin(), saveat.end(), tp_.begin());
        std::sort(tp_.begin(), tp_.end());
        tp_.erase(std::unique(tp_.begin(), tp_.end()), tp_.end());
    }

    odeSolution sol;
    sol.t.push_back(tp_[0]);
    sol.u.push_back(VEC_DOUBLE());
    sol.u[0] = x0;
    VEC_DOUBLE xc;
    xc = sol.u[0];
    for (int i = 0; i < int(tp_.size() - 1); i++)
    {
        auto DT = tp_[i + 1] - tp_[i];
        auto K1 = run_forest(f, xc);

        vec_add(xc, vec_mul(K1, DT ));
        // save data
        if (std::find(saveat.begin(), saveat.end(), tp_[i + 1]) != std::end(saveat))
        {
            sol.u.push_back(VEC_DOUBLE(xc));
            sol.t.push_back(tp_[i + 1]);
        }
    }

    return sol;
}

odeSolution odeSolve_Euler_withgrad(Forest &f, VEC_DOUBLE x0, std::pair<double, double> tspan, double dt, VEC_DOUBLE saveat)
{
    if (dt > tspan.second-tspan.first){
        dt = (tspan.second - tspan.first)/double(saveat.size());
    }
    if (dt < 0)
    { // 没指定时间间隔，计算100步
        dt = (tspan.second - tspan.first) / 10.0;
    }
    int N = int(floor((tspan.second - tspan.first) / dt));

    VEC_DOUBLE time_point; // 用于计算的时间点，与保存数据的时间点不同 保证精度的最低要求
    for (int i = 0; i < N; i++)
    {
        time_point.push_back(double(i) * dt + tspan.first);
    }
    if (time_point[N - 1] < tspan.second)
    {
        time_point.push_back(tspan.second);
    }
    VEC_DOUBLE tp_;
    tp_.resize(time_point.size() + saveat.size());

    if (saveat.size() == 0)
    {
        saveat = time_point;
        tp_ = time_point;
    }
    else
    {
        // 将time_point和saveat合并 并且排序
        std::merge(time_point.begin(), time_point.end(), saveat.begin(), saveat.end(), tp_.begin());
        std::sort(tp_.begin(), tp_.end());
        tp_.erase(std::unique(tp_.begin(), tp_.end()), tp_.end());
    }

    odeSolution sol;
    sol.t.push_back(tp_[0]);
    sol.u.push_back(VEC_DOUBLE());
    sol.u[0] = x0;
    VEC_DOUBLE xc;
    xc = sol.u[0];

    // 给grad分配内存
    // 每个time_point的梯度都要保存
    // 在每个计算过程中所有K的梯度都要保存
    // 不要直接用dnode的值，x和K要对所有参数求导数，不仅仅是自身函数的
    std::vector<odeGradient> time_point_grad;
    std::vector<odeGradient> K_grad;

    auto ROOT_ID = f.root_id;
    // K_grad
    for (int i = 0; i < 1; i++)
    {
        K_grad.push_back(odeGradient());
        K_grad[i].dxdv = generate_dxdv_fromForest(f);
    }
    // time_point
    for (long unsigned int i = 0; i < tp_.size(); i++)
    {
        time_point_grad.push_back(odeGradient());
        time_point_grad[i].dxdv = generate_dxdv_fromForest(f);
    }
    sol.grad.clear();
    // initial dx0/dt = 0

    for (int i = 0; i < int(tp_.size() - 1); i++)
    {
        // get d(x_i)/dw

        auto DT = tp_[i + 1] - tp_[i];
        auto K1 = run_forest(f, xc);
        gradient_forest(f, xc);
        // gradient of K1
        K_grad[0].dxdv = generate_dxdv_fromForest(f);
        for (long unsigned int j = 0; j < ROOT_ID.size(); j++)
        {
            auto R_ID = ROOT_ID[j];
            std::map<unsigned, double> dfdx;
            for (long unsigned int k = 0; k < ROOT_ID.size(); k++)
            {
                dfdx.insert({ROOT_ID[k], f.trees_weights.dnode_dvariable[R_ID][ID_N][k]});
            }
            for (auto it : time_point_grad[i].dxdv)
            {
                for (auto it2 : it.second)
                {
                    // df/dx * dx/dw
                    vec_add(K_grad[0].dxdv[R_ID][it2.first], vec_mul(it2.second, dfdx[it.first]));
                }
            }
            for (auto it : f.trees_weights.dnode_dvariable[R_ID])
            {
                vec_add(K_grad[0].dxdv[R_ID][it.first], it.second);
            }
        }

        vec_add(xc, vec_mul(K1, DT));

        for (auto it : time_point_grad[i + 1].dxdv)
        {
            // x_it
            for (auto it2 : it.second)
            {
                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], time_point_grad[i].dxdv[it.first][it2.first]);
                vec_add(time_point_grad[i + 1].dxdv[it.first][it2.first], vec_mul(K_grad[0].dxdv[it.first][it2.first], DT ));
            }
        }

        // save data
        if (std::find(saveat.begin(), saveat.end(), tp_[i + 1]) != std::end(saveat))
        {
            sol.u.push_back(VEC_DOUBLE(xc));
            sol.t.push_back(tp_[i + 1]);
            sol.grad.push_back(odeGradient());
            sol.grad[sol.grad.size() - 1].dxdv = time_point_grad[i + 1].dxdv;
        }
    }

    return sol;
}