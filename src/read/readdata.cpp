#include "read.h"
#include <armadillo>
MATRIX readmatrix(std::string filename)
{
	arma::mat mat;
	mat.load(filename, arma::auto_detect);
	auto m = mat.n_rows;
	auto n = mat.n_cols;
	MATRIX mat_;
	for (auto i = 0; i < m; i++) {
		mat_.push_back(std::vector<double>());
		for (auto j = 0; j < n; j++) {
			mat_[i].push_back(mat(i, j));
		}
	}
	return mat_;
}

VECTOR readvector(std::string filename)
{
	arma::mat mat;
	mat.load(filename, arma::auto_detect);
	auto m = mat.n_rows;
	auto n = mat.n_cols;
	VECTOR mat_;
	for (auto j = 0; j < m; j++) {
		mat_.push_back(mat(j, 0));
	}
	return mat_;
}