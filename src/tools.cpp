#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
      cout << "Estimation vector size error." << endl;
      return rmse;
  }

  // accumulate squared residuals
  for (int i=0; i < estimations.size(); ++i) {
  // ... your code here
  VectorXd residual = estimations[i] - ground_truth[i];
  residual = residual.array() * residual.array();
  rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (px == 0 && py == 0) {
      cout << "Divide by zero!" << endl;
      return Hj;
  }

  // compute the Jacobian matrix
  Hj << px/sqrt(px*px+py*py), py/sqrt(px*px+py*py), 0, 0,
      -1*py/(px*px+py*py), px/(px*px+py*py), 0, 0,
      py*(vx*py-vy*px)/pow(px*px+py*py, 3/2), px*(vy*px-vx*py)/pow(px*px+py*py, 3/2), px/sqrt(px*px+py*py), py/sqrt(px*px+py*py);

  return Hj;
}
