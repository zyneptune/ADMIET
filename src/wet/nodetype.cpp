#include "wet.h"
#include "tool.h"
extern TreeOpSet TreeOps;
node_calculation_func calculation_node_map::operator()(int i)
{
    return node_calculation[i];
}

node_gradient_func calculation_node_map::operator[](int i)
{
    return node_gradient[i];
}

calculation_node_map::calculation_node_map()
{
    node_type = std::vector<int>();
    node_calculation = std::map<int, node_calculation_func>();
}

///////////////////////
// register nodes //
void node_cal_func_example(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
}
void node_grad_func_example(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
}
void node_print_func_example(Tree tree, Weights &ws, unsigned node_id)
{
}

calculation_type CAL_TYPE;
calculation_node_map CAL_NODE_MAP;

void node_cal_func_var(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ps.node_values[node_id][0] = 0.0;
    for (unsigned i = 0; i < ws.weights[node_id].size(); i++)
    {
        ps.node_values[node_id][0] += ws.weights[node_id][i] * input[i];
    }
}
void node_grad_func_var(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    if (weight_id == ID_N)
    {
        for (unsigned i = 0; i < ws.dnode_dvariable[node_id][weight_id].size(); i++)
        {
            ws.dnode_dvariable[node_id][weight_id][i] = ws.weights[node_id][i];
        }
    }
    if (weight_id == node_id)
    {
        for (unsigned i = 0; i < ws.dnode_dvariable[node_id][weight_id].size(); i++)
        {
            ws.dnode_dvariable[node_id][weight_id][i] = input[i];
        }
    }
}

void node_print_func_var(Tree &tree, Weights &ws, unsigned node_id)
{
    printf("x%d", max_idx(ws.weights[node_id]) + 1);
}

void node_cal_func_con(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ps.node_values[node_id][0] = ws.weights[node_id][0];
}
void node_grad_func_con(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    if (weight_id == node_id)
    {
        ws.dnode_dvariable[node_id][weight_id][0] = 1.0;
    }
}

void node_print_func_con(Tree &tree, Weights &ws, unsigned node_id)
{
    printf("%f", ws.weights[node_id][0]);
}

void node_cal_func_ops(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ps.node_values[node_id][0] = 0.0;
    std::vector<double> conn_value;
    conn_value.assign(conn.size(), 0.0);
    for (unsigned i = 0; i < conn.size(); i++)
    {
        conn_value[i] = ps.node_values[conn[i]][0];
    }
    for (unsigned i = 0; i < ws.weights[node_id].size(); i++)
    {
        ps.node_values[node_id][0] += ws.weights[node_id][i] * (TreeOps.funcs[i].func_ptr(conn_value));
    }
}
void node_grad_func_ops(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    // v = weight' * ops(conn)
    // ps.node_values[node_id][0] = 0.0;
    std::vector<double> conn_value;
    conn_value.assign(conn.size(), 0.0);
    ws.dnode_dvariable[node_id][weight_id].assign(ws.weights[weight_id].size(), 0.0);
    for (unsigned i = 0; i < conn.size(); i++)
    {
        conn_value[i] = ps.node_values[conn[i]][0];
    }
    if (node_id != weight_id)
    {
        for (unsigned i = 0; i < TreeOps.funcs.size(); i++)
        {
            VEC_DOUBLE vg = TreeOps.funcs[i].func_gradient_ptr(conn_value);
            for (unsigned j = 0; j < conn.size(); j++)
            {
                auto t = vec_mul(ws.dnode_dvariable[conn[j]][weight_id], vg[j] * ws.weights[node_id][i]);
                if (t.size() == 0)
                    continue;
                vec_add(ws.dnode_dvariable[node_id][weight_id], t);
            }
        }
    }
    else
    {
        for (unsigned i = 0; i < ws.dnode_dvariable[node_id][weight_id].size(); i++)
        {
            ws.dnode_dvariable[node_id][weight_id][i] = (TreeOps.funcs[i].func_ptr(conn_value));
        }
    }
}

void node_print_func_ops(Tree &tree, Weights &ws, unsigned node_id)
{
    printf("(");
    auto child_id1 = tree.connection_down[node_id][0];
    CAL_NODE_MAP.node_print[tree.node_type[child_id1]](tree, ws, child_id1);
    auto opi = max_idx(ws.weights[node_id]);
    printf("%s", TreeOps.funcs[opi].name.c_str());
    child_id1 = tree.connection_down[node_id][1];
    CAL_NODE_MAP.node_print[tree.node_type[child_id1]](tree, ws, child_id1);
    printf(")");
}

void node_cal_func_plus2(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ps.node_values[node_id][0] = 0.0;
    for (unsigned i = 0; i < ws.weights[node_id].size(); i++)
    {
        ps.node_values[node_id][0] += ws.weights[node_id][i] * ps.node_values[conn[i]][0];
    }
}
void node_grad_func_plus2(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ws.dnode_dvariable[node_id][weight_id].assign(ws.weights[weight_id].size(), 0.0);
    if (node_id != weight_id)
    {
        auto m = ws.dnode_dvariable[conn[0]][weight_id];
        vec_mul_inplace(m, ws.weights[node_id][0]);
        vec_add(ws.dnode_dvariable[node_id][weight_id], m);
        m = ws.dnode_dvariable[conn[1]][weight_id];
        vec_mul_inplace(m, ws.weights[node_id][1]);
        vec_add(ws.dnode_dvariable[node_id][weight_id], m);
    }
    else
    {
        for (unsigned i = 0; i < ws.dnode_dvariable[node_id][weight_id].size(); i++)
        {
            ws.dnode_dvariable[node_id][weight_id][i] = ps.node_values[conn[i]][0];
        }
    }
}

void node_print_func_plus2(Tree &tree, Weights &ws, unsigned node_id)
{
    /*
    auto idx = max_idx(ws.weights[node_id]);
    auto child_id = tree.connection_down[node_id][idx];
    CAL_NODE_MAP.node_print[tree.node_type[child_id]](tree, ws, child_id);
    */
    printf("(");
    auto child_id = tree.connection_down[node_id][0];
    if (ws.weights[node_id][0] == 0.0){
        child_id = tree.connection_down[node_id][1];
        CAL_NODE_MAP.node_print[tree.node_type[child_id]](tree, ws, child_id);
    }else if (ws.weights[node_id][1] == 0.0){
        printf("%f", ws.weights[node_id][0] * ws.weights[child_id][0]);
    }else{
        printf("%f", ws.weights[node_id][0] * ws.weights[child_id][0]);
        printf("+");
        printf("%f*", ws.weights[node_id][1]);
        child_id = tree.connection_down[node_id][1];
        CAL_NODE_MAP.node_print[tree.node_type[child_id]](tree, ws, child_id);
    }
    printf(")");
}

void node_cal_func_mul2(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ps.node_values[node_id][0] = ps.node_values[conn[0]][0] * ps.node_values[conn[1]][0];
}
void node_grad_func_mul2(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    if (weight_id == node_id)
    {
        ws.dnode_dvariable[node_id][weight_id][0] = 0.0;
    }
    else
    {
        ws.dnode_dvariable[node_id][weight_id].assign(ws.weights[weight_id].size(), 0.0);
        auto m = ws.dnode_dvariable[conn[0]][weight_id];
        vec_add(ws.dnode_dvariable[node_id][weight_id], vec_mul(m, ps.node_values[conn[1]][0]));
        m = ws.dnode_dvariable[conn[1]][weight_id];
        vec_add(ws.dnode_dvariable[node_id][weight_id], vec_mul(m, ps.node_values[conn[0]][0]));
    }
}

void node_print_func_mul2(Tree &tree, Weights &ws, unsigned node_id)
{
    printf("(");
    auto child_id = tree.connection_down[node_id][0];
    CAL_NODE_MAP.node_print[tree.node_type[child_id]](tree, ws, child_id);
    child_id = tree.connection_down[node_id][1];
    printf("*");
    CAL_NODE_MAP.node_print[tree.node_type[child_id]](tree, ws, child_id);
    printf(")");
}

void node_cal_func_plus(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ps.node_values[node_id][0] = 0.0;
    for (auto it : conn){
        ps.node_values[node_id][0] += ps.node_values[it][0];
    }
}
void node_grad_func_plus(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ws.dnode_dvariable[node_id][weight_id].assign(ws.weights[weight_id].size(), 0.0);
    if (node_id != weight_id)
    {
        for (auto it : conn){
            vec_add(ws.dnode_dvariable[node_id][weight_id],ws.dnode_dvariable[it][weight_id]);
        }
    }
    else
    {
        for (unsigned i = 0; i < ws.dnode_dvariable[node_id][weight_id].size(); i++)
        {
            ws.dnode_dvariable[node_id][weight_id][i] = ps.node_values[conn[i]][0];
        }
    }
}
void node_print_func_plus(Tree &tree, Weights &ws, unsigned node_id)
{
    printf("(");
    auto child_id = tree.connection_down[node_id];
    for (unsigned i=0;i<child_id.size();i++){
        CAL_NODE_MAP.node_print[tree.node_type[child_id[i]]](tree, ws, child_id[i]);
        if (i != child_id.size()-1){
            printf("+");
        }
    }
    printf(")");
}

void node_cal_func_minus(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    // ps.node_values[node_id][0] = 0.0;
    ps.node_values[node_id][0] = ps.node_values[conn[0]][0] - ps.node_values[conn[1]][0];
}
void node_grad_func_minus(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input)
{
    ws.dnode_dvariable[node_id][weight_id].assign(ws.weights[weight_id].size(), 0.0);
    if (node_id != weight_id)
    {
        vec_add(ws.dnode_dvariable[node_id][weight_id],ws.dnode_dvariable[conn[0]][weight_id]);
        vec_add(ws.dnode_dvariable[node_id][weight_id], vec_mul(ws.dnode_dvariable[conn[1]][weight_id],-1.0));
    }
}
void node_print_func_minus(Tree &tree, Weights &ws, unsigned node_id)
{
    printf("(");
    auto child_id = tree.connection_down[node_id];
    for (unsigned i=0;i<child_id.size();i++){
        CAL_NODE_MAP.node_print[tree.node_type[child_id[i]]](tree, ws, child_id[i]);
        if (i != child_id.size()-1){
            printf("-");
        }
    }
    printf(")");
}

/////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

void register_node(int input_dim)
{
    setTreeOpsFromFile("ops_setting.ini");
    // CT_CON
    CAL_NODE_MAP.node_type.push_back(CT_CON);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_CON, node_cal_func_con));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_CON, node_grad_func_con));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_CON, true));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_CON, 0));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_CON, node_print_func_con));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_CON, 1));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_CON, false));

    // CT_VAR
    CAL_NODE_MAP.node_type.push_back(CT_VAR);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_VAR, node_cal_func_var));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_VAR, node_grad_func_var));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_VAR, true));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_VAR, 0));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_VAR, node_print_func_var));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_VAR, input_dim));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_VAR, true));

    // CT_OPS
    CAL_NODE_MAP.node_type.push_back(CT_OPS);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_OPS, node_cal_func_ops));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_OPS, node_grad_func_ops));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_OPS, true));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_OPS, 2));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_OPS, node_print_func_ops));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_OPS, TreeOps.funcs.size()));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_OPS, true));

    // CT_PLUS2
    CAL_NODE_MAP.node_type.push_back(CT_WEIGHTED_PLUS_2);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_WEIGHTED_PLUS_2, node_cal_func_plus2));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_WEIGHTED_PLUS_2, node_grad_func_plus2));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_WEIGHTED_PLUS_2, true));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_WEIGHTED_PLUS_2, 2));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_WEIGHTED_PLUS_2, node_print_func_plus2));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_WEIGHTED_PLUS_2, 2));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_WEIGHTED_PLUS_2, true));

    // CT_MUL2
    CAL_NODE_MAP.node_type.push_back(CT_MUL2);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_MUL2, node_cal_func_mul2));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_MUL2, node_grad_func_mul2));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_MUL2, false));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_MUL2, 2));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_MUL2, node_print_func_mul2));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_MUL2, 1));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_MUL2, true));

    // CT_PLUS
    CAL_NODE_MAP.node_type.push_back(CT_PLUS);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_PLUS, node_cal_func_plus));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_PLUS, node_grad_func_plus));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_PLUS, false));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_PLUS, 0));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_PLUS, node_print_func_plus));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_MUL2, 0));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_MUL2, false));

    // CT_MINUS
    CAL_NODE_MAP.node_type.push_back(CT_MINUS);
    CAL_NODE_MAP.node_calculation.insert(std::pair<int, node_calculation_func>(CT_MINUS, node_cal_func_minus));
    CAL_NODE_MAP.node_gradient.insert(std::pair<int, node_gradient_func>(CT_MINUS, node_grad_func_minus));
    CAL_NODE_MAP.node_istrainable.insert(std::pair<int, bool>(CT_MINUS, false));
    CAL_NODE_MAP.node_children_num.insert(std::pair<int, int>(CT_MINUS, 2));
    CAL_NODE_MAP.node_print.insert(std::pair<int, node_print_func>(CT_MINUS, node_print_func_minus));
    CAL_NODE_MAP.node_weight_dim.insert(std::pair<int, int>(CT_MINUS, 0));
    CAL_NODE_MAP.node_isonehot.insert(std::pair<int, bool>(CT_MINUS, false));
}

void print_tree(Tree tree, Weights &ws)
{
    CAL_NODE_MAP.node_print[tree.node_type[tree.node_names[0]]](tree, ws, tree.node_names[0]);
}