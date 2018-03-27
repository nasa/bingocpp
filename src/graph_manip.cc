/*!
 * \file graph_manip.cc
 *
 * \author Ethan Adams
 * \date 2/26/2018
 *
 * This file contains the cpp version of AGraphCpp.py
 */

#include <iostream>

#include "BingoCpp/graph_manip.h"
#include "BingoCpp/acyclic_graph_nodes.h"
#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/fitness_metric.h"



AcyclicGraph::AcyclicGraph() {
  stack = Eigen::ArrayX3d(0, 3);
  constants = Eigen::VectorXd(0);
  fitness = std::vector<double>();
  fit_set = false;
}

AcyclicGraph::AcyclicGraph(const AcyclicGraph &ag) {
  stack = ag.stack;
  constants = ag.constants;
  fitness = ag.fitness;
  fit_set = ag.fit_set;
}

AcyclicGraph AcyclicGraph::copy() {
  AcyclicGraph temp = AcyclicGraph();
  temp.stack = stack;
  temp.constants = constants;
  temp.fitness = fitness;
  temp.fit_set = fit_set;
  return temp;
}

bool AcyclicGraph::needs_optimization() {
  std::set<int> util = utilized_commands();
  std::set<int>::iterator it;

  for (it = util.begin(); it != util.end(); ++it) {
    if (stack(*it, 0) == 1 && stack(*it, 1) == -1 ||
        stack(*it, 0) == 1 && stack(*it, 1) >= constants.size()) {
      return true;
    }
  }

  return false;
}

void AcyclicGraph::set_constants(Eigen::VectorXd con) {
  constants = con;
}

int AcyclicGraph::count_constants() {
  std::set<int> util = utilized_commands();
  std::set<int>::iterator it;
  int const_num = 0;

  for (it = util.begin(); it != util.end(); ++it) {
    if (stack(*it, 0) == 1) {
      stack(*it, 1) = const_num;
      stack(*it, 2) = const_num;
      const_num += 1;
    }
  }

  return const_num;
}

Eigen::ArrayXXd AcyclicGraph::evaluate(Eigen::ArrayXXd &eval_x) {
  return SimplifyAndEvaluate(stack, eval_x, constants);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> AcyclicGraph::evaluate_deriv(
  Eigen::ArrayXXd &eval_x) {
  return SimplifyAndEvaluateWithDerivative(stack, eval_x, constants);
}

std::string AcyclicGraph::latexstring() {
  std::set<int> util = utilized_commands();
  std::set<int>::iterator it = util.begin();
  std::vector<std::string> strings;
  std::ostringstream stream;

  for (int i = 0; i < stack.rows(); ++i) {
    std::string temp = "";

    if (util.count(i) == 1) {
      if (stack(*it, 0) == 0)
        stream << oper_interface.operator_map[stack(*it, 0)]->get_print()
               << "_" << stack(*it, 1);

      else if (stack(*it, 0) == 1) {
        if (stack(*it, 1) == -1) {
          stream << "0";

        } else {
          stream << constants[stack(*it, 1)];
        }

      } else if (stack(*it, 0) == 2)
        stream << strings[stack(*it, 1)] << " + " <<
               strings[stack(*it, 2)];

      else if (stack(*it, 0) == 3)
        stream << strings[stack(*it, 1)] << " - (" <<
               strings[stack(*it, 2)] << ")";

      else if (stack(*it, 0) == 4)
        stream << "(" << strings[stack(*it, 1)] << ")(" <<
               strings[stack(*it, 2)] << ")";

      else if (stack(*it, 0) == 5)
        stream << "\\frac{" << strings[stack(*it, 1)] << "}{" <<
               strings[stack(*it, 2)] << "}";

      else if (stack(*it, 0) == 6 || stack(*it, 0) == 7 || stack(*it, 0) == 8
               || stack(*it, 0) == 9 || stack(*it, 0) == 12)
        stream << "\\" << oper_interface.operator_map[stack(*it, 0)]->get_print()
               << "{" << strings[stack(*it, 1)] << "}";

      else if (stack(*it, 0) == 10)
        stream << "(" << strings[stack(*it, 1)] << ")^{(" <<
               strings[stack(*it, 2)] << ")}";

      else if (stack(*it, 0) == 11) {
        stream << "|{" << strings[stack(*it, 1)] << "}|";
      }

      ++it;

    } else {
      stream << "";
    }

    temp = stream.str();
    strings.push_back(temp);
    stream.str(std::string());
  }

  return strings.back();
}

std::set<int> AcyclicGraph::utilized_commands() {
  std::set<int> util;
  util.insert(stack.rows() - 1);

  for (int i = stack.rows() - 1; i >= 0; --i) {
    if (stack(i, 0) > 1 && util.count(i) == 1) {
      util.insert(stack(i, 1));
      util.insert(stack(i, 2));
    }
  }

  return util;
}

int AcyclicGraph::complexity() {
  return utilized_commands().size();
}

std::string AcyclicGraph::print_stack() {
  std::ostringstream out;
  out << "---full stack---\n";

  for (int i = 0; i < stack.rows(); ++i) {
    out << std::left << std::setw(4) << i;
    out << "<= ";

    if (stack(i, 0) == 0)
      out << oper_interface.operator_map[stack(i, 0)]->get_print()
          << stack(i, 1) << std::endl;

    else if (stack(i, 0) == 1) {
      if (stack(i, 1) == -1) {
        out << oper_interface.operator_map[stack(i, 0)]->get_print();

      } else {
        out << constants[stack(i, 1)];
      }

      out << std::endl;

    } else {
      out << "(" << stack(i, 1) << ") "
          << oper_interface.operator_map[stack(i, 0)]->get_print()
          << " (" << stack(i, 2) << ")\n";
    }
  }

  out << "---small stack---\n";
  std::set<int>::iterator it;
  std::set<int> util = utilized_commands();

  for (it = util.begin(); it != util.end(); ++it) {
    out << std::left << std::setw(4) << *it;
    out << "<= ";

    if (stack(*it, 0) == 0)
      out << oper_interface.operator_map[stack(*it, 0)]->get_print()
          << stack(*it, 1) << std::endl;

    else if (stack(*it, 0) == 1) {
      if (stack(*it, 1) == -1) {
        out << oper_interface.operator_map[stack(*it, 0)]->get_print();

      } else {
        out << constants[stack(*it, 1)];
      }

      out << std::endl;

    } else {
      out << "(" << stack(*it, 1) << ") "
          << oper_interface.operator_map[stack(*it, 0)]->get_print()
          << " (" << stack(*it, 2) << ")\n";
    }
  }

  return out.str();
}

AcyclicGraphManipulator::AcyclicGraphManipulator(int nvars, int ag_size,
    int nloads,
    float float_lim, float terminal_prob) {
  this->nvars = nvars;
  this->ag_size = ag_size;
  this->nloads = nloads;
  this->float_lim = float_lim;
  this->terminal_prob = terminal_prob;
  node_type_vec.push_back(0);
  node_type_vec.push_back(1);

  for (int i = 1; i < nvars; ++i) {
    term_vec.push_back(0);
  }
}

void AcyclicGraphManipulator::add_node_type(int node_type) {
  bool exists = false;

  for (int i = 0; i < node_type_vec.size(); ++i) {
    if (node_type == node_type_vec[i]) {
      exists = true;
    }
  }

  if (!exists) {
    node_type_vec.push_back(node_type);

    if (node_type <= 1) {
      term_vec.push_back(node_type_vec.size());

    } else {
      op_vec.push_back(node_type_vec.size());
    }
  }
}

AcyclicGraph AcyclicGraphManipulator::generate() {
  AcyclicGraph indv = AcyclicGraph();
  Eigen::ArrayX3d array(ag_size, 3);

  for (int i = 0; i < ag_size; ++i) {
    float r = static_cast <float> (rand()) / static_cast < float> (RAND_MAX);

    if (i < nloads || r < terminal_prob) {
      std::vector<int> vec = rand_terminal();
      array(i, 0) = vec[0];
      array(i, 1) = vec[1];
      array(i, 2) = vec[2];

    } else {
      std::vector<int> vec = rand_operator(i);
      array(i, 0) = vec[0];
      array(i, 1) = vec[1];
      array(i, 2) = vec[2];
    }
  }

  indv.stack = array;
  return indv;
}

std::pair<Eigen::ArrayX3d, Eigen::VectorXd> AcyclicGraphManipulator::dump(
  AcyclicGraph &indv) {
  std::pair<Eigen::ArrayX3d, Eigen::VectorXd> temp(indv.stack, indv.constants);
  return temp;
}

AcyclicGraph AcyclicGraphManipulator::load(
  std::pair<Eigen::ArrayX3d, Eigen::VectorXd> indv_list) {
  AcyclicGraph temp;
  temp.stack = indv_list.first;
  temp.constants = indv_list.second;
  return temp;
}

std::vector<AcyclicGraph> AcyclicGraphManipulator::crossover(
  AcyclicGraph &parent1, AcyclicGraph &parent2) {
  int c_point = rand() % ag_size;
  std::vector<AcyclicGraph> temp;
  AcyclicGraph c1 = AcyclicGraph(parent1);
  AcyclicGraph c2 = AcyclicGraph(parent2);
  c1.stack(c_point, 0) = parent2.stack(c_point, 0);
  c1.stack(c_point, 1) = parent2.stack(c_point, 1);
  c1.stack(c_point, 2) = parent2.stack(c_point, 2);
  c2.stack(c_point, 0) = parent1.stack(c_point, 0);
  c2.stack(c_point, 1) = parent1.stack(c_point, 1);
  c2.stack(c_point, 2) = parent1.stack(c_point, 2);
  c1.fitness = std::vector<double>();
  c2.fitness = std::vector<double>();
  c1.fit_set = false;
  c2.fit_set = false;
  temp.push_back(c1);
  temp.push_back(c2);
  return temp;
}

AcyclicGraph AcyclicGraphManipulator::mutation(AcyclicGraph &indv) {
  std::set<int> util = indv.utilized_commands();
  int loc = rand() % util.size();
  std::set<int>::iterator it = util.begin();

  for (int i = 0; i < loc; ++i) {
    ++it;
  }

  int mut_point = *it;
  int orig_node_type = indv.stack(mut_point, 0);
  int new_param1 = indv.stack(mut_point, 1);
  int new_param2 = indv.stack(mut_point, 2);
  float r = static_cast <float> (rand()) / static_cast < float> (RAND_MAX);
  std::vector<int> vec;

  if (r < 0.4 && mut_point > nloads) {
    float ran = static_cast <float> (rand()) / static_cast < float> (RAND_MAX);
    int temp_node = -1;
    int temp_p1 = 0;
    int temp_p2 = 0;

    while (temp_node < 0) {
      if (ran < terminal_prob) {
        vec = rand_terminal();
        temp_node = vec[0];
        temp_p1 = vec[1];
        temp_p2 = vec[2];

      } else {
        vec = rand_operator(mut_point);
        temp_node = vec[0];
        temp_p1 = vec[1];
        temp_p2 = vec[2];
      }

      if (temp_node == orig_node_type && orig_node_type > 1) {
        temp_node = -1;
      }
    }

    indv.stack(mut_point, 0) = temp_node;
    indv.stack(mut_point, 1) = temp_p1;
    indv.stack(mut_point, 2) = temp_p2;

  } else if (r < 0.8) {
    if (orig_node_type <= 1) {
      new_param1 = mutate_terminal_param(orig_node_type);
      new_param2 = new_param1;

    } else {
      vec = rand_operator_params(2, mut_point);
      new_param1 = vec[0];
      new_param2 = vec[1];
    }

    indv.stack(mut_point, 1) = new_param1;
    indv.stack(mut_point, 2) = new_param2;

  } else {
    if (orig_node_type > 1) {
      int ran = rand() % 2;
      int pruned_param = -2;

      if (ran == 0) {
        pruned_param = new_param1;

      } else if (ran == 1) {
        pruned_param = new_param2;
      }

      for (int i = mut_point; i < indv.stack.rows(); ++i) {
        int p0 = indv.stack(i, 1);
        int p1 = indv.stack(i, 2);

        if (p0 == mut_point) {
          p0 = pruned_param;
        }

        if (p1 == mut_point) {
          p1 = pruned_param;
        }

        indv.stack(i, 1) = p0;
        indv.stack(i, 2) = p1;
      }
    }
  }

  indv.fitness = std::vector<double>();
  indv.fit_set = false;
  return indv;
}

int AcyclicGraphManipulator::distance(AcyclicGraph &indv1,
                                      AcyclicGraph &indv2) {
  return (indv1.stack - indv2.stack).sum();
}

std::vector<int> AcyclicGraphManipulator::rand_operator_params(int arity,
    int stack_location) {
  std::vector<int> temp;

  if (stack_location > 1) {
    for (int i = 0; i < arity; ++i) {
      temp.push_back(rand() % stack_location);
    }

  } else {
    for (int i = 0; i < arity; ++i) {
      temp.push_back(0);
    }
  }

  return temp;
}

int AcyclicGraphManipulator::rand_operator_type() {
  return node_type_vec[rand() % op_vec.size() + 2];
}

std::vector<int> AcyclicGraphManipulator::rand_operator(int stack_location) {
  std::vector<int> temp;
  int node_type = rand_operator_type();
  std::vector<int> temp2 = rand_operator_params(2, stack_location);
  temp.push_back(node_type);
  temp.push_back(temp2[0]);
  temp.push_back(temp2[1]);
  return temp;
}

int AcyclicGraphManipulator::rand_terminal_param(int terminal) {
  if (terminal == 0) {
    return (rand() % nvars);

  } else {
    return -1;
  }
}

int AcyclicGraphManipulator::mutate_terminal_param(int terminal) {
  if (terminal == 0) {
    return (rand() % nvars);

  } else {
    return -1;
  }
}

std::vector<int> AcyclicGraphManipulator::rand_terminal() {
  std::vector<int> temp;
  int node = node_type_vec[rand() % term_vec.size()];
  int param = rand_terminal_param(node);
  temp.push_back(node);
  temp.push_back(param);
  temp.push_back(param);
  return temp;
}