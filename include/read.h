#ifndef READ_H
#define READ_H
#include <vector>
#include <string>
#include "forest.h"
typedef std::vector<double> VECTOR;
typedef std::vector<VECTOR> MATRIX;

MATRIX readmatrix(std::string filename);
VECTOR readvector(std::string filename);
Forest parse_forest(std::string filepath);
#endif
