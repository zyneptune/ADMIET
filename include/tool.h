#include <cmath>
#include <vector>

double norm(std::vector<double>, char);
std::vector<double> dot_divide(std::vector<double>, double);
unsigned max_idx(std::vector<double>);

void vec_add(std::vector<double> &x1, std::vector<double> x2);
std::vector<double> vec_add_r(std::vector<double> x1, std::vector<double> x2);
std::vector<double> vec_mul(std::vector<double> x1, double a);
void vec_mul_inplace(std::vector<double> &x, double a);
void vec_rand_init(std::vector<double> &x, int randseed);
void vec_average_init(std::vector<double> &x);
void vec_copy(std::vector<double> &x1, std::vector<double> x2);
void onehot(std::vector<double> &x);
double dist_2(std::vector<double> x, std::vector<double> y);
void normalize(std::vector<double> &x);
double inner_mul(std::vector<double> &x1, std::vector<double> &x2);
std::vector<double> zeros_as(std::vector<double> src);
unsigned get_id();
std::vector<std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>> DataLoader(std::vector<std::vector<double>> datax,std::vector<std::vector<double>> datay, int batch_num, bool shuffle);
std::vector<double> linspace(double u1, double u2, int num);
std::vector<double> linspace(double u1, double u2, double dt);
// template <typename T>
bool in_vector(std::vector<unsigned> x, unsigned y);
