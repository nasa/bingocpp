#ifndef INCLUDE_BINGOCPP_BACKEND_NODES_H_
#define INCLUDE_BINGOCPP_BACKEND_NODES_H_

#include <vector>

#include <Eigen/Dense>

namespace bingo {
namespace backend {

typedef Eigen::ArrayXXd (
  *forward_operator_function)(
    int, int, const Eigen::ArrayXXd&,
    const Eigen::VectorXd&, std::vector<Eigen::ArrayXXd>&
);

typedef void (
  *reverse_operator_function)(
    int, int, int,
    const std::vector<Eigen::ArrayXXd>&, std::vector<Eigen::ArrayXXd>&
);

/*
 * Maps param1, param2, x, constants, and forward eval to the correct
 * forward eval function corresponding to the operation node.
 */
Eigen::ArrayXXd forward_eval_function(int node, int param1, int param2,
                                      const Eigen::ArrayXXd& x, 
                                      const Eigen::VectorXd& constants,
                                      std::vector<Eigen::ArrayXXd>& forward_eval);
/*
 * Maps reverse_index, param1, param2, forward evaluation stack and 
 * revese evaluation stack to the corresponding operation node.
 */
void reverse_eval_function(int node, int reverse_index, int param1, int param2,
                           const std::vector<Eigen::ArrayXXd>& forward_eval,
                           std::vector<Eigen::ArrayXXd>& reverse_eval);
}
} // namespace bingo

#endif