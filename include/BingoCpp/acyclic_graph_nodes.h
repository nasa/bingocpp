/*!
 * \file acyclic_graph_nodes.hh
 *
 * \author Ethan Adams
 * \date 2/9/2018
 *
 * This is the header file to hold the Operation abstract class
 * and all implementations of that class. Also holds the OperatorInterface
 * class, which includes a map to keep the Operations in
 */

#ifndef INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_
#define INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_

#include <vector>

#include <Eigen/Dense>

namespace agraphnodes{
  typedef void (
    *forward_operator_function)(
      const Eigen::ArrayX3i&, const Eigen::ArrayXXd&,
      const Eigen::VectorXd&, std::vector<Eigen::ArrayXXd>&, std::size_t
  );
  typedef void (
    *derivative_operator_function)(
      const Eigen::ArrayX3i &, const int,
      const std::vector<Eigen::ArrayXXd> &,
      std::vector<Eigen::ArrayXXd> &, int
  );
  
  void forward_eval_function(int node, const Eigen::ArrayX3i &stack,
                             const Eigen::ArrayXXd &x,
                             const Eigen::VectorXd &constants,
                             std::vector<Eigen::ArrayXXd> &buffer,
                             std::size_t result_location);

  void derivative_eval_function(int node, const Eigen::ArrayX3i &stack,
                                const int command_index,
                                const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                int dependency);
} //agraphnodes

#endif  // INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_
