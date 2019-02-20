#ifndef TEST_TESTING_UTILS_H_
#define TEST_TESTING_UTILS_H_

#include <Eigen/Dense>

namespace testutils {
  const double TESTING_TOL = 1e-7;

  struct AGraphValues {
  Eigen::ArrayXXd x_vals;
  Eigen::VectorXd constants;
  AGraphValues() {}
  AGraphValues(Eigen::ArrayXXd &x, Eigen::VectorXd &c) : 
    x_vals(x), constants(c) {}
  };

  namespace {
    inline double difference(double val_1, double val_2) {
      if ((std::isnan(val_1) && std::isnan(val_2)) ||
          (val_1 == INFINITY && val_2 == INFINITY) ||
          (val_1 == -INFINITY && val_2 == -INFINITY)) {
        return 0;
      } else {
        return val_1 - val_2;
      }
    }

    inline bool non_comparable_matrices(const Eigen::ArrayXXd &array1,
                                  const Eigen::ArrayXXd &array2) {
      int rows_array1 = array1.rows();
      int rows_array2 = array2.rows();
      int cols_array1 = array1.cols();
      int cols_array2 = array2.cols();
      return (!(rows_array1 > 0) || 
              !(rows_array2 > 0) || 
              !(cols_array1 > 0) || 
              !(cols_array2 > 0) || 
              (rows_array1 != rows_array2) || 
              (cols_array1 != cols_array2));
    }
  } // namespace

  inline bool almost_equal(const Eigen::ArrayXXd &array1, const Eigen::ArrayXXd &array2) {
    if (non_comparable_matrices(array1, array2))
      return false;

    int rows_array1 = array1.rows();
    int cols_array1 = array1.cols();
    Eigen::MatrixXd matrix_diff = Eigen::MatrixXd(rows_array1, cols_array1);
    for (int row = 0; row < rows_array1; row++) {
      for (int col = 0; col < cols_array1; col++) {
        matrix_diff(row, col) = difference(array1(row, col), array1(row, col));
      }
    }
    double frobenius_norm = matrix_diff.norm();
    return (frobenius_norm < TESTING_TOL ? true : false);
  }

  inline AGraphValues init_agraph_vals(double begin, double end, int num_points) {
    Eigen::VectorXd constants = Eigen::VectorXd(2);
    constants << 10, 3.14;

    Eigen::ArrayXXd x_vals(num_points, 2);
    x_vals.col(0) = Eigen::ArrayXd::LinSpaced(num_points, begin, end);
    x_vals.col(1) = Eigen::ArrayXd::LinSpaced(num_points, end, -begin);

    return AGraphValues(x_vals, constants);
  }

  inline std::vector<Eigen::ArrayXXd> init_op_evals_x0(const AGraphValues &sample_agraph_1_values) {
    Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals.col(0);
    double constant = sample_agraph_1_values.constants[0];
    Eigen::ArrayXXd c_0 = constant * Eigen::ArrayXd::Ones(x_0.rows());

    std::vector<Eigen::ArrayXXd> op_evals_x0 = std::vector<Eigen::ArrayXXd>();
    op_evals_x0.push_back(x_0);
    op_evals_x0.push_back(c_0);
    op_evals_x0.push_back(x_0+x_0);
    op_evals_x0.push_back(x_0-x_0);
    op_evals_x0.push_back(x_0*x_0);
    op_evals_x0.push_back(x_0/x_0);
    op_evals_x0.push_back(x_0.sin());
    op_evals_x0.push_back(x_0.cos());
    op_evals_x0.push_back(x_0.exp());
    op_evals_x0.push_back(x_0.abs().log());
    op_evals_x0.push_back(x_0.abs().pow(x_0));
    op_evals_x0.push_back(x_0.abs());
    op_evals_x0.push_back(x_0.abs().sqrt());

    return op_evals_x0;
  }

  inline std::vector<Eigen::ArrayXXd> init_op_x_derivs(const AGraphValues &sample_agraph_1_values) {
    Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals.col(0);
    std::vector<Eigen::ArrayXXd> op_x_derivs = std::vector<Eigen::ArrayXXd>();
    int size = x_0.rows();

    auto last_nan = [](Eigen::ArrayXXd array) {
      array(array.rows() - 1, array.cols() -1) = std::nan("1");
      Eigen::ArrayXXd modified_array = array;
      return modified_array;
    };
    op_x_derivs.push_back(Eigen::ArrayXd::Ones(size));
    op_x_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_x_derivs.push_back(2.0  * Eigen::ArrayXd::Ones(size));
    op_x_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_x_derivs.push_back(2.0 * x_0);
    op_x_derivs.push_back(last_nan(Eigen::ArrayXd::Zero(size)));
    op_x_derivs.push_back(x_0.cos());
    op_x_derivs.push_back(-x_0.sin());
    op_x_derivs.push_back(x_0.exp());
    op_x_derivs.push_back(1.0 / x_0);
    op_x_derivs.push_back(last_nan(x_0.abs().pow(x_0)*(x_0.abs().log()
                         + Eigen::ArrayXd::Ones(size))));
    op_x_derivs.push_back(x_0.sign());
    op_x_derivs.push_back(0.5 * x_0.sign() / x_0.abs().sqrt());

    return op_x_derivs;
  }

  inline std::vector<Eigen::ArrayXXd> init_op_c_derivs(const AGraphValues &sample_agraph_1_values) {
    int size = sample_agraph_1_values.x_vals.rows();
    Eigen::ArrayXXd c_1 = sample_agraph_1_values.constants[1] * Eigen::ArrayXd::Ones(size);
    std::vector<Eigen::ArrayXXd> op_c_derivs = std::vector<Eigen::ArrayXXd>();

    op_c_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_c_derivs.push_back(Eigen::ArrayXd::Ones(size));
    op_c_derivs.push_back(2.0  * Eigen::ArrayXd::Ones(size));
    op_c_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_c_derivs.push_back(2.0 * c_1);
    op_c_derivs.push_back((Eigen::ArrayXd::Zero(size)));
    op_c_derivs.push_back(c_1.cos());
    op_c_derivs.push_back(-c_1.sin());
    op_c_derivs.push_back(c_1.exp());
    op_c_derivs.push_back(1.0 / c_1);
    op_c_derivs.push_back(c_1.abs().pow(c_1)*(c_1.abs().log()
                          + Eigen::ArrayXd::Ones(size)));
    op_c_derivs.push_back(c_1.sign());
    op_c_derivs.push_back(0.5 * c_1.sign() / c_1.abs().sqrt());

    return op_c_derivs;
  }
} // testutils

#endif // INCLUDE_BINGO_TESTING_UTILS_H_
