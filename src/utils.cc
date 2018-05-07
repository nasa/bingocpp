/*!
 * \file utils.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains utility functions for doing and testing
 * sybolic regression problems in the bingo package
 */

#include "BingoCpp/utils.h"
#include <numeric>


std::vector<Eigen::ArrayXXd> calculate_partials(Eigen::ArrayXXd x) {
  std::vector<int> break_points;
  break_points.push_back(0);

  for (int i = 0; i < x.rows(); ++i) {
    if (isnan(x(i))) {
      break_points.push_back(i);
    }
  }

  break_points.push_back(x.rows());
  std::vector<Eigen::ArrayXXd> x_all_vec;
  int x_rows = 0;
  int x_cols = 0;
  std::vector<Eigen::ArrayXXd> times_deriv_all_vec;
  int t_rows = 0;
  int t_cols = 0;

  for (int i = 0; i < break_points.size() - 1; ++i) {
    int start = break_points[i];
    int row_size = break_points[i + 1] - start;
    Eigen::ArrayXXd x_seg(row_size, x.cols());
    x_seg = x.block(start, 0, row_size, x.cols());
    Eigen::ArrayXXd time_deriv(x_seg.rows(), x_seg.cols());

    for (int j = 0; j < x_seg.cols(); ++j) {
      Eigen::ArrayXXd temp(x_seg.rows(), 1);
      temp = x_seg.block(0, j, x_seg.rows(), 1);
      time_deriv.block(0, j, x_seg.rows(), 1) = savitzky_golay(temp, 7, 3, 1);
    }

    Eigen::ArrayXXd x_seg_no_edge(x_seg.rows() - 7, x_seg.cols());
    x_seg_no_edge = x_seg.block(3, 0, x_seg_no_edge.rows(), x_seg.cols());
    Eigen::ArrayXXd time_deriv_no_edge(time_deriv.rows() - 7, time_deriv.cols());
    time_deriv_no_edge = time_deriv.block(3, 0, time_deriv_no_edge.rows(),
                                          time_deriv.cols());
    x_all_vec.push_back(x_seg_no_edge);
    x_rows += x_seg_no_edge.rows();
    x_cols += x_seg_no_edge.cols();
    times_deriv_all_vec.push_back(time_deriv_no_edge);
    t_rows += time_deriv_no_edge.rows();
    t_cols += time_deriv_no_edge.cols();
  }

  Eigen::ArrayXXd x_all(x_rows, x_cols);
  Eigen::ArrayXXd times_deriv_all(t_rows, t_cols);
  int x_start = 0;

  for (int i = 0; i < x_all_vec.size(); ++i) {
    x_all.block(x_start, 0, x_all_vec[i].rows(), x_all.cols()) = x_all_vec[i];
    x_start += x_all_vec[i].rows();
  }

  int t_start = 0;

  for (int i = 0; i < times_deriv_all_vec.size(); ++i) {
    times_deriv_all.block(t_start, 0, times_deriv_all_vec[i].rows(),
                          times_deriv_all.cols()) = times_deriv_all_vec[i];
    t_start += times_deriv_all_vec[i].rows();
  }

  std::vector<Eigen::ArrayXXd> temp;
  temp.push_back(x_all);
  temp.push_back(times_deriv_all);
  return temp;
}

double GramPoly(double gp_i, double gp_m, double gp_k, double gp_s) {
  double gram_poly = 0;

  if (gp_k > 0) {
    gram_poly = (4. * gp_k - 2.) / (gp_k * (2. * gp_m - gp_k + 1.)) *
                (gp_i * GramPoly(gp_i, gp_m, gp_k - 1., gp_s) +
                 gp_s * GramPoly(gp_i, gp_m, gp_k - 1., gp_s - 1.)) -
                ((gp_k - 1.) * (2. * gp_m + gp_k)) /
                (gp_k * (2. * gp_m - gp_k + 1.)) *
                GramPoly(gp_i, gp_m, gp_k - 2, gp_s);

  } else {
    if (gp_k == 0 && gp_s == 0) {
      gram_poly = 1.;

    } else {
      gram_poly = 0.;
    }
  }

  return gram_poly;
}

double GenFact(double a, double b) {
  int fact = 1;

  for (int i = a - b + 1; i < a + 1; ++i) {
    fact *= i;
  }

  return fact;
}

double GramWeight(double gw_i, double gw_t, double gw_m, double gw_n,
                  double gw_s) {
  double weight = 0;

  for (int i = 0; i < gw_n + 1; ++i) {
    weight += (2. * i + 1.) * GenFact(2. * gw_m, i) /
              GenFact(2. * gw_m + i + 1, i + 1) *
              GramPoly(gw_i, gw_m, i, 0) *
              GramPoly(gw_t, gw_m, i, gw_s);
  }

  return weight;
}

Eigen::ArrayXXd savitzky_golay(Eigen::ArrayXXd y, int window_size, int order,
                               int deriv) {
  int m = (window_size - 1) / 2;
  // fill weights
  Eigen::ArrayXXd weights(2 * m + 1, 2 * m + 1);

  for (int i = m * -1; i < m + 1; ++i) {
    for (int j = m * -1; j < m + 1; ++j) {
      weights(i + m, j + m) = GramWeight(i, j, m, order, deriv);
    }
  }

  // convolution
  int y_center = 0;
  int w_ind = 0;
  int y_len = y.rows();
  Eigen::ArrayXXd f(y_len, 1);

  for (int i = 0; i < y_len; ++i) {
    if (i < m) {
      y_center = m;
      w_ind = i;

    } else if (y_len - i <= m) {
      y_center = y_len - m - 1;
      w_ind = 2 * m + 1 - (y_len - i);

    } else {
      y_center = i;
      w_ind = m;
    }

    f(i) = 0;

    for (int j = m * -1; j < m + 1; ++j) {
      f(i) += y(y_center + j) * weights(j + m, w_ind);
    }
  }

  return f;
}