#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "helpers.h"
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

void color_deconv(int b, int g, int r) {
	Vector3d I;
	I << r, g, b;
	I.unaryExpr([] (double &c) {c = log10(c);});
	Matrix3d Q;
	Q << 0.65, 0.704, 0.285,
		 0.072, 0.99, 0.105,
		 0.6218, 0, 0.7831;
	I*Q.inverse();
}

#endif