#ifndef INTERPS
#define INTERPS
#include <vector>

class Spline
{
public:
    Spline(std::vector<double> datax, std::vector<double> datay, int SplineType = 1);
    double f(double x1);
    std::vector<double>  f(std::vector<double> x1);
    double f_d(double x1);
    std::vector<double>  f_d(std::vector<double> x1);

private:
    int size_i;
    std::vector<double> x;
    std::vector<double> a, b, c, d;
};

class BSpline{
public:
    BSpline(std::vector<double> datax, std::vector<double> datay, int order,int cpoints,double gamma);
    double f(double x1);
    std::vector<double>  f(std::vector<double> x1);
    double f_d(double x1);
    std::vector<double>  f_d(std::vector<double> x1);
private:
    std::vector<double> ts;
    std::vector<double> weights;
    int order;
    double biq(double x,std::vector<double> ts_,int i,int order);
    double biq_d(double x,std::vector<double> ts_,int i,int order);
    double bvalue(double x,std::vector<double> ts_,std::vector<double> w_,int order);
    double bvalue_d(double x,std::vector<double> ts_,std::vector<double> w_,int order);
};
#endif