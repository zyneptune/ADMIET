#ifndef WET_H
#define WET_H

#include <vector>
#include <map>
#include <string>
#include <algorithm>
// id standard:
// for node: natural number from 0 to N.
// for weights and constants and variales
// ID_N for variables
#define ID_N 100
typedef std::vector<double> VEC_DOUBLE;
typedef std::vector<unsigned> VEC_ID;
enum calculation_type
{
    CT_VAR,           // variables
    CT_CON,           // constants
    CT_OPS,           // operators
    CT_PLUS,          // ...+...+...+... type
    CT_WEIGHTED_PLUS, // w1f1 + w2f2+..., Sigma(wi) = 1.0
    CT_WEIGHTED_PLUS_2,
    CT_MUL2,
    CT_MINUS
};

struct Tree
{
    std::map<unsigned, VEC_ID> connection_down;
    std::map<unsigned, VEC_ID> connection_gradient; // record the node which have influence.
    std::map<unsigned, int> node_type;
    VEC_ID node_names;
    VEC_ID terminal_nodes;
    VEC_ID operator_nodes;
    std::vector<VEC_ID> layer_nodes;
};

struct Weights
{
    std::map<unsigned, VEC_DOUBLE> weights;
    std::map<unsigned, std::map<unsigned, VEC_DOUBLE>> dnode_dvariable;
    VEC_ID untrainable;
};

struct Progress
{
    std::map<unsigned, VEC_DOUBLE> node_values;
};

struct TreeOp
{
    int input_num;
    std::string name;
    double (*func_ptr)(VEC_DOUBLE);
    std::vector<double> (*func_gradient_ptr)(VEC_DOUBLE);
};

struct TreeOpSet
{
    std::vector<TreeOp> funcs;
    int input_num;
};

typedef void (*node_calculation_func)(Progress &, Weights &, unsigned, VEC_ID &, VEC_DOUBLE &);
typedef void (*node_gradient_func)(Progress &, Weights &, unsigned, unsigned, VEC_ID &, VEC_DOUBLE &);
typedef void (*node_print_func)(Tree &, Weights &, unsigned);
class calculation_node_map
{
public:
    std::vector<int> node_type;
    std::map<int, node_calculation_func> node_calculation;
    std::map<int, node_gradient_func> node_gradient;
    std::map<int, bool> node_istrainable;
    std::map<int, int> node_children_num;
    std::map<int, node_print_func> node_print;
    std::map<int, int> node_weight_dim;
    std::map<int, bool> node_isonehot;
    calculation_node_map();
    node_calculation_func operator()(int);
    node_gradient_func operator[](int);
};
void register_node(int);
//////////////////////////////////////////////////////////////////////////

void run_node(Progress &ps, Weights &ws, unsigned node_id, VEC_ID &conn, VEC_DOUBLE &input, int type);
double run_tree(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input);
void gradient_tree(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input);
void gradient_node(Progress &ps, Weights &ws, unsigned node_id, unsigned weight_id, VEC_ID &conn, VEC_DOUBLE &input, int type);
void gradient_tree_top2node(Tree tree, Weights &ws, Progress &ps, unsigned node_id, VEC_DOUBLE input);

// build tree
Tree build_tree_standard(unsigned head_length);
Weights generate_weights_from_tree(Tree tree, int input_dim, int init_method, std::pair<double, double> constant_range, int randseed);
Progress generate_progress_from_tree(Tree tree);
Progress *generate_progress_pointer_from_tree(Tree tree);
Weights *generate_weights_pointer_from_tree(Tree tree, int input_dim, int init_method, std::pair<double, double> constant_range, int randseed);
Weights merge_weights(Weights w1, Weights w2);
Progress merge_progress(Progress g1, Progress g2);
std::map<unsigned, VEC_DOUBLE> generate_gradient_map(Weights weights);
// loss functions
double tree_loss_mse(Tree tree, Weights &ws, Progress &ps, VEC_DOUBLE input, VEC_DOUBLE y);
double tree_loss_mse_batch(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> input, VEC_DOUBLE y);
std::map<unsigned, VEC_DOUBLE> tree_loss_mse_gradient_batch(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> input, VEC_DOUBLE y);
VEC_DOUBLE tree_loss_mse_gradient_batch_2node(Tree tree, Weights &ws, Progress &ps, unsigned node_id, std::vector<VEC_DOUBLE> input, VEC_DOUBLE y);
// training
typedef std::map<unsigned, VEC_DOUBLE> MAP_GRADIENT;
int tree_train_fixstep(Tree tree, Weights &ws, MAP_GRADIENT g, double step_length, double epsilon, bool IsPenalty, std::pair<double, double> c_r);
typedef double (*batch_loss_func)(Tree, Weights &, Progress &, std::vector<VEC_DOUBLE>, VEC_DOUBLE);
void tree_train_linearsearch(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> datax, VEC_DOUBLE datay, batch_loss_func loss_func, double epsilon, bool IsPenalty, std::pair<double, double> constant_range);

double penalty_func_1(Tree tree, Weights &ws);
VEC_DOUBLE penalty_func_gradient_3(VEC_DOUBLE x);
double penalty_func_3(VEC_DOUBLE x);
// optimization module
double tree_loss(Tree tree,Weights &ws,Progress &ps, std::vector<VEC_DOUBLE> x, VEC_DOUBLE y, double(error_func)(double x, double y));
std::map<unsigned, VEC_DOUBLE> tree_loss_gradient(Tree tree, Weights &ws, Progress &ps, std::vector<VEC_DOUBLE> x, VEC_DOUBLE y, double(error_func_g)(double x, double y));
void TRAIN_TREE(Tree tree, Weights &ws, MAP_GRADIENT g);

// double penalty_func_2(VEC_DOUBLE x);
//  output
void print_tree(Tree tree, Weights &ws);

// inline operators
void setTreeOpsFromFile(std::string filename);
inline double plus(VEC_DOUBLE x);
inline VEC_DOUBLE plus_gradient(VEC_DOUBLE x);
inline double minus(VEC_DOUBLE x);
inline VEC_DOUBLE minus_gradient(VEC_DOUBLE x);
inline double multi(VEC_DOUBLE x);
inline VEC_DOUBLE multi_gradient(VEC_DOUBLE x);
inline double protect_divide(VEC_DOUBLE x);
inline VEC_DOUBLE protect_divide_gradient(VEC_DOUBLE x);
inline double operator_example(VEC_DOUBLE x);
inline VEC_DOUBLE example_gradient(VEC_DOUBLE x);
inline double hill2(VEC_DOUBLE x);
inline VEC_DOUBLE hill2_gradient(VEC_DOUBLE x);

#endif