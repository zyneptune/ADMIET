#include <iostream>
#include "wet.h"
#include "tool.h"
#include "read.h"
#include "forest.h"
#include "cmdlines.h"
#include "ode.h"
#include "optimizer.h"
#include "interpolation.h"

int main(int argc, char *argv[])
{
    //if (argc < 2)
    //    return -1;
    //std::string filename(argv[1]);
    //std::cout << filename << std::endl;
    //setTreeOpsFromFile(filename);
    std::vector<double> x = {1.0,2.0,3.0,4.0,6.0};
    std::vector<double> y = {1.0,4.0,9.0,16.0,36.0};
    auto bb = BSpline(x,y,4,6,0.5);
    for (int i=0;i<100;i++){
        std::cout << bb.f(double(i)*0.1) << std::endl;
    }
    return 0;
}