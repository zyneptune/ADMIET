
#include "tool.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
double norm(std::vector<double> x, char type)
{
    double sum_ = 0.0;
    switch (type)
    {
    case '2':
        for (unsigned i = 0; i < x.size(); i++)
        {
            sum_ += std::pow(x[i], 2.0);
        }
        sum_ = std::sqrt(sum_);
        break;
    case '1':
        for (unsigned i = 0; i < x.size(); i++)
        {
            sum_ += std::fabs(x[i]);
        }
        break;
    default:
        for (unsigned i = 0; i < x.size(); i++)
        {
            sum_ += std::pow(x[i], 2.0);
        }
        sum_ = std::sqrt(sum_);
        break;
    }
    return sum_;
}

std::vector<double> dot_divide(std::vector<double> x, double a)
{
    std::vector<double> x2 = x;
    for (unsigned i = 0; i < x2.size(); i++)
    {
        x2[i] /= a;
    }
    return x2;
}

unsigned max_idx(std::vector<double> x)
{
    unsigned i = 0;
    double s = x[0];
    for (unsigned j = 0; j < x.size(); j++)
    {
        if (x[j] >= s)
        {
            i = j;
            s = x[j];
        }
    }
    return i;
}

void vec_add(std::vector<double> &x1, std::vector<double> x2)
{
    if (x2.size() == 0)
    {
        return;
    }
    for (unsigned i = 0; i < x1.size(); i++)
    {
        x1[i] += x2[i];
    }
}

std::vector<double> vec_add_r(std::vector<double> x1, std::vector<double> x2)
{
    std::vector<double> re = x1;
    vec_add(re, x2);
    return re;
}

void vec_copy(std::vector<double> &x1, std::vector<double> x2)
{
    for (int i = 0; i < x1.size(); i++)
    {
        x1[i] = x2[i];
    }
}

void onehot(std::vector<double> &x)
{
    double sum_ = 0.0;
    for (unsigned i = 0; i < x.size(); i++)
    {
        if (x[i] <= 1e-4)
        {
            x[i] = 0.0;
        }
    }
    sum_ = norm(x, '1');
    for (unsigned i = 0; i < x.size(); i++)
    {
        x[i] = x[i] / sum_;
    }
}

std::vector<double> vec_mul(std::vector<double> x1, double a)
{
    std::vector<double> x2 = x1;
    for (unsigned i = 0; i < x2.size(); i++)
    {
        x2[i] *= a;
    }
    return x2;
}

void vec_mul_inplace(std::vector<double> &x, double a)
{
    for (unsigned i = 0; i < x.size(); i++)
    {
        x[i] *= a;
    }
}

void vec_rand_init(std::vector<double> &x, int randseed)
{
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(0, 1.0); // 左闭右闭区间
    e.seed(randseed);
    for (unsigned i = 0; i < x.size(); i++)
    {
        x[i] = u(e);
    }
    onehot(x);
}

void vec_average_init(std::vector<double> &x)
{
    double l = double(x.size());
    if (l == 0)
        return;
    for (unsigned i = 0; i < x.size(); i++)
    {
        x[i] = 1.0 / l;
    }
}

// template <typename T>
bool in_vector(std::vector<unsigned> x, unsigned y)
{
    for (unsigned i = 0; i < x.size(); i++)
    {
        if (x[i] == y)
        {
            return true;
        }
    }
    return false;
}

double dist_2(std::vector<double> x, std::vector<double> y)
{
    std::vector<double> z = vec_add_r(x, vec_mul(y, -1.0));
    return std::pow(norm(z, '2'), 2.0);
}

void normalize(std::vector<double> &x)
{
    double sum_ = 0.0;
    for (unsigned i = 0; i < x.size(); i++)
    {
        sum_ += (x[i] * x[i]);
    }
    sum_ = std::sqrt(sum_);
    if (sum_ > 1e-8)
    {
        for (unsigned i = 0; i < x.size(); i++)
        {
            x[i] = x[i] / sum_;
        }
    }
}

double inner_mul(std::vector<double> &x1, std::vector<double> &x2)
{
    double sum_ = 0.0;
    for (unsigned i = 0; i < x1.size(); i++)
    {
        sum_ += (x1[i] * x2[i]);
    }
    return sum_;
}

std::vector<double> zeros_as(std::vector<double> src)
{
    std::vector<double> r;
    r.assign(src.size(), 0.0);
    return r;
}

unsigned get_id(){
    static unsigned int a = 10000000;
    return a++;
}



std::vector<std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>> DataLoader(std::vector<std::vector<double>> datax,std::vector<std::vector<double>> datay, int batch_num, bool shuffle){
    std::vector<std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>> dataloader;
    for (int i = 0;i<batch_num;i++){
        dataloader.push_back(std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>({{},{}}));
    }
    if (shuffle){
        std::vector<unsigned> num;
        for (int j=0;j<datax.size();j++){
            num.push_back(j);
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(num.begin(),num.end(),g);
        for (int j=0;j<datax.size();j++){
            dataloader[j%batch_num].first.push_back(std::vector<double>(datax[num[j]]));
            dataloader[j%batch_num].second.push_back(std::vector<double>(datay[num[j]]));
        }
    }else{
        for (int j=0;j<datax.size();j++){
            dataloader[j%batch_num].first.push_back(std::vector<double>(datax[j]));
            dataloader[j%batch_num].second.push_back(std::vector<double>(datay[j]));
        }
    }
    return dataloader;
}

std::vector<double> linspace(double u1, double u2, int num){
    std::vector<double> t;
    double dt = (u2-u1) / double(num-1);
    for (int i=0;i<num;i++){
        t.push_back(u1+double(i)*dt);
    }
    return t;
}

std::vector<double> linspace(double u1,double u2, double dt){
    std::vector<double> t;
    double ct = u1;
    while (ct < u2){
        t.push_back(ct);
        ct+=dt;
    }
    t.push_back(u2);
    return t;
}