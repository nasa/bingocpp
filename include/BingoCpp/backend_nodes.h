/*
 * Copyright 2018 United States Government as represented by the Administrator 
 * of the National Aeronautics and Space Administration. No copyright is claimed 
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0 
 * (the "License"); you may not use this file except in compliance with the 
 * License. You may obtain a copy of the License at  
 * http://www.apache.org/licenses/LICENSE-2.0. 
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations under 
 * the License.
*/
#ifndef INCLUDE_BINGOCPP_BACKEND_NODES_H_
#define INCLUDE_BINGOCPP_BACKEND_NODES_H_

#include <vector>

#include <Eigen/Dense>

namespace bingo {

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
} // namespace bingo

#endif
