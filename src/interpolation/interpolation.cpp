#include "interpolation.h"
#include <Eigen/Dense>
#include <utility>

// test
//#include <iostream>

using namespace Eigen;

Spline::Spline(std::vector<double> datax, std::vector<double> datay, int SplineType)
{
    int size = datax.size();
    if (size < 3)
    {
        abort();
    }
    this->size_i = size;
    // sort, increase
    std::vector<double> tempy;
    //this->x = (double *)malloc(sizeof(double) * size);
    //tempy = (double *)malloc(sizeof(double) * size);

//    for (int i = 0; i < size; i++)
//    {
//        this->x[i] = datax[i];
//        tempy[i] = datay[i];
//    }
    this->x = std::vector<double>(datax);
    tempy = std::vector<double>(std::move(datay));

    for (int i = 0; i < size - 1; i++)
    {
        double t1, t2;
        for (int j = 0; j < size - 1 - i; j++)
        {
            if (this->x[j] > this->x[j + 1])
            {
                t1 = this->x[j];
                this->x[j] = this->x[j + 1];
                this->x[j + 1] = t1;

                t2 = tempy[j];
                tempy[j] = tempy[j + 1];
                tempy[j + 1] = t2;
            }
        }
    }

//    double *h = (double *)malloc(sizeof(double) * (size - 1));
//    double *yy = (double *)malloc(sizeof(double) * (size - 1));
    std::vector<double> h,yy;
    for (int i = 0; i < size - 1; i++)
    {
        h.push_back( this->x[i + 1] - this->x[i]);
        yy.push_back(tempy[i + 1] - tempy[i]);
    }
    // Ax = b
    Eigen::MatrixXd A(size, size);
    A.setZero();
    Eigen::VectorXd bb(size);
    bb.setZero();
    for (int i = 1; i < size - 1; i++)
    {
        A(i, i - 1) = h[i - 1];
        A(i, i) = 2 * (h[i - 1] + h[i]);
        A(i, i + 1) = h[i];
        bb(i) = 6 * (yy[i] / h[i] - yy[i - 1] / h[i - 1]);
    }
    bb(0) = 0;
    bb(size - 1) = 0;
    if (SplineType == 1)
    {
        A(0, 0) = 1;
        A(size - 1, size - 1) = 1;
    }
    if (SplineType == 2)
    {
        A(0, 0) = 2 * h[0];
        A(0, 1) = h[0];
        A(size - 1, size - 1) = 2 * h[size - 1];
        A(size - 1, size - 2) = h[size - 1];
    }
    if (SplineType != 1 && SplineType != 2)
    {
        A(0, 0) = -h[1];
        A(0, 1) = h[0] + h[1];
        A(0, 2) = -h[0];
        A(size - 1, size - 1) = 2 * h[size - 2];
        A(size - 1, size - 2) = h[size - 1] + h[size - 2];
        A(size - 1, size - 2) = -h[size - 1];
    }
    Eigen::VectorXd m = A.colPivHouseholderQr().solve(bb);
    // coefficients

//    this->a = (double *)malloc(sizeof(double) * (size - 1));
//    this->b = (double *)malloc(sizeof(double) * (size - 1));
//    this->c = (double *)malloc(sizeof(double) * (size - 1));
//    this->d = (double *)malloc(sizeof(double) * (size - 1));
//    std::vector<double> a,b,c,d;
//    //test
//    std::cout << A << std::endl<< std::endl<< std::endl;
//    std::cout << bb << std::endl<< std::endl << std::endl;
//    std::cout << m << std::endl << std::endl << std::endl;
    for (int i = 0; i < size - 1; i++)
    {
        this->a.push_back( tempy[i]);
        this->b.push_back( (tempy[i + 1] - tempy[i]) / h[i] - h[i] / 2.0 * m[i] - h[i] / 6.0 * (m[i + 1] - m[i]));
        this->c.push_back(  m[i] / 2.0);
        this->d.push_back( (m[i + 1] - m[i]) / (6 * h[i]));
    }
//    free(h);
//    free(yy);
//    free(tempy);
}

double Spline::f(double x1)
{
    if (x1 < x[0] || x1 > x[size_i - 1])
    {
        printf("%f is out of range(%f,%f)!\n", x1, x[0], x[size_i - 1]);
        abort();
    }
    for (int i = 0; i < size_i - 1; i++)
    {
        if (x1 >= x[i] && x1 <= x[i + 1])
        {
            double p = x1 - x[i];
            return a[i] + b[i] * p + c[i] * p * p + d[i] * p * p * p;
        }
    }
}

std::vector<double> Spline::f(std::vector<double> x1)
{
    std::vector<double> t;
    for (int i = 0; i < x1.size(); i++)
    {
        t.push_back(f(x1[i]));
    }
    return t;
}

double Spline::f_d(double x1)
{
    if (x1 < x[0] || x1 > x[size_i - 1])
    {
        abort();
    }
    for (int i = 0; i < size_i - 1; i++)
    {
        if (x1 >= x[i] && x1 <= x[i + 1])
        {
            double p = x1 - x[i];
            return b[i] + 2 * c[i] * p + 3 * d[i] * p * p;
        }
    }
}

std::vector<double> Spline::f_d(std::vector<double> x1)
{
    std::vector<double> t;
    for (int i = 0; i < x1.size(); i++)
    {
        t.push_back(f_d(x1[i]));
    }
    return t;
}

double BSpline::biq(double x,std::vector<double> ts_,int i,int order){
    if (order==1){
        if (x > ts_[i] && x <= ts_[i+order]){
            return 1.0;
        }else{
            return 0.0;
        }
    }else{
        if (x>ts_[i] && x <= ts_[i+order]){
            return (x - ts_[i]) / (ts_[i+order-1] - ts_[i]) * biq(x, ts_, i, order - 1) + (ts_[i+order] - x) / (ts_[i+order] - ts_[i+1]) * biq(x, ts_, i + 1, order - 1);
        }
        else{
            return 0.0;
        }
    }
    return 0.0;
}
double BSpline::biq_d(double x,std::vector<double> ts_,int i,int order){
    if (order == 1){
        return 0.0;
    }else{
        if (x>ts_[i] && x <= ts_[i+order]){
            return double(order) / (ts_[i+order-1] - ts_[i]) * biq(x, ts_, i, order - 1) - double(order) / (ts_[i+order] - ts_[i+1]) * biq(x, ts_, i + 1, order - 1);
        }else{
            return 0.0;
        }
    }
}

double BSpline::bvalue(double x,std::vector<double> ts_,std::vector<double> w_,int order){
    int l = ts_.size() - order;
    double sum = 0.0;
    for (int i=0;i<l;i++){
        sum += biq(x,ts_,i,order) * w_[i];
    }
    return sum;
}

double BSpline::bvalue_d(double x,std::vector<double> ts_,std::vector<double> w_,int order){
    int l = ts_.size() - order;
    double sum = 0.0;
    for (int i=0;i<l;i++){
        sum += biq_d(x,ts_,i,order) * w_[i];
    }
    return sum;
}

double BSpline::f(double x1){
    return this->bvalue(x1,this->ts,this->weights,this->order);
}
double BSpline::f_d(double x1){
    return this->bvalue_d(x1,this->ts,this->weights,this->order);
}

std::vector<double> BSpline::f(std::vector<double> x1)
{
    std::vector<double> t;
    for (int i = 0; i < x1.size(); i++)
    {
        t.push_back(f(x1[i]));
    }
    return t;
}

std::vector<double> BSpline::f_d(std::vector<double> x1)
{
    std::vector<double> t;
    for (int i = 0; i < x1.size(); i++)
    {
        t.push_back(f_d(x1[i]));
    }
    return t;
}

BSpline::BSpline(std::vector<double> datax, std::vector<double> datay, int order,int cpoints,double gamma){
    // Use Bspline method to fit f(x)=y
    // Bmatrix
    // ts_
    int size = datax.size();
    if (size < 3)
    {
        abort();
    }
    // sort, increase
    std::vector<double> tempy;
    auto x = std::vector<double>(datax);
    tempy = std::vector<double>(std::move(datay));
    for (int i = 0; i < size - 1; i++) {
        double t1, t2;
        for (int j = 0; j < size - 1 - i; j++) {
            if (x[j] > x[j + 1]) {
                t1 = x[j];
                x[j] = x[j + 1];
                x[j + 1] = t1;
                t2 = tempy[j];
                tempy[j] = tempy[j + 1];
                tempy[j + 1] = t2;
            }
        }
    }
    int l = cpoints - order;
    int m = size;
    this->ts.assign(cpoints,x[0]);
    double dx = (x[x.size()-1] - x[0])/double(cpoints-1);
    for (int i=0;i<cpoints-1;i++){
        this->ts[i+1] = this->ts[i] + dx;
    }
    this->order = order;
    Eigen::MatrixXd Bm(m, l);
    Eigen::MatrixXd G(l,l);
    Eigen::VectorXd b(m);
    for (int i=0;i<m;i++){
        b(i) = tempy[i];
        for (int j=0;j<l;j++){
            Bm(i,j) = biq(datax[i],this->ts,j,order);
        }
    }
    for (int i=0;i<l-1;i++){
        G(i,i) = 1.0 * gamma;
        G(i,i+1) = -1.0 * gamma;
    }
    G(l-1,l-1) = 1.0 * gamma;
    // Solve Bm * w = datay
    // smooth gamma
    // min ||Bm * w - datay|| + gamma ||Gw||
    auto w = (Bm.transpose() * Bm + G.transpose() * G).inverse() * Bm.transpose() * b;
    this->weights.clear();
    for (int i=0;i<cpoints;i++){
        this->weights.push_back(w(i));
    }
}
