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
  stack = Eigen::ArrayX3i(0, 3);
  constants = Eigen::VectorXd(0);
  simple_stack = Eigen::ArrayX3i(0, 3);
  fitness = std::vector<double>();
  fit_set = false;
  needs_opt = false;
  opt_rate = 0;
  genetic_age = 0;
}

AcyclicGraph::AcyclicGraph(const AcyclicGraph &ag) {
  stack = ag.stack;
  constants = ag.constants;
  simple_stack = ag.simple_stack;
  fitness = ag.fitness;
  fit_set = ag.fit_set;
  needs_opt = ag.needs_opt;
  opt_rate = ag.opt_rate;
  genetic_age = ag.genetic_age;
}

AcyclicGraph AcyclicGraph::copy() {
  AcyclicGraph temp = AcyclicGraph();
  temp.stack = stack;
  temp.constants = constants;
  temp.simple_stack = simple_stack;
  temp.fitness = fitness;
  temp.fit_set = fit_set;
  temp.needs_opt = needs_opt;
  temp.opt_rate = opt_rate;
  temp.genetic_age = genetic_age;
  return temp;
}

bool AcyclicGraph::needs_optimization() {
  if (opt_rate == 0) {
    for (int i = 0; i < simple_stack.rows(); ++i) {
      if (simple_stack(i, 0) == 1 && simple_stack(i, 1) == -1 ||
          simple_stack(i, 0) == 1 && simple_stack(i, 1) >= constants.size()) {
        return true;
      }
    }

    return false;

  } else {
    return needs_opt;
  }
}

void AcyclicGraph::set_constants(Eigen::VectorXd con) {
  constants = con;
}

int AcyclicGraph::count_constants() {
  if (opt_rate == 0) {
    std::set<int> util = utilized_commands();
    std::set<int>::iterator it = util.begin();
    int const_num = 0;

    for (int i = 0; i < simple_stack.rows(); ++i, ++it) {
      if (simple_stack(i, 0) == 1) {
        simple_stack(i, 1) = const_num;
        simple_stack(i, 2) = const_num;
        stack(*it, 1) = const_num;
        stack(*it, 2) = const_num;
        const_num += 1;
      }
    }

    return const_num;

  } else {
    return constants.size();
  }
}

Eigen::ArrayXXd AcyclicGraph::evaluate(Eigen::ArrayXXd &eval_x) {
  return Evaluate(simple_stack, eval_x, constants);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> AcyclicGraph::evaluate_deriv(
  Eigen::ArrayXXd &eval_x) {
  return EvaluateWithDerivative(simple_stack, eval_x, constants);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>
AcyclicGraph::evaluate_with_const_deriv(
  Eigen::ArrayXXd &eval_x) {
  return EvaluateWithDerivative(simple_stack, eval_x, constants, false);
}


std::string AcyclicGraph::latexstring() {
  std::vector<std::string> strings;
  std::ostringstream stream;

  for (int i = 0; i < simple_stack.rows(); ++i) {
    std::string temp = "";

    switch ((int)simple_stack(i, 0)) {
      case 0:
        stream << oper_interface.operator_map[simple_stack(i, 0)]->get_print()
               << "_" << simple_stack(i, 1);
        break;

      case 1:
        if (simple_stack(i, 1) == -1) {
          stream << "0";

        } else {
          stream << constants[simple_stack(i, 1)];
        }

        break;

      case 2:
        stream << strings[simple_stack(i, 1)] << " + " <<
               strings[simple_stack(i, 2)];
        break;

      case 3:
        stream << strings[simple_stack(i, 1)] << " - (" <<
               strings[simple_stack(i, 2)] << ")";
        break;

      case 4:
        stream << "(" << strings[simple_stack(i, 1)] << ")(" <<
               strings[simple_stack(i, 2)] << ")";
        break;

      case 5:
        stream << "\\frac{" << strings[simple_stack(i, 1)] << "}{" <<
               strings[simple_stack(i, 2)] << "}";
        break;

      case 6:
      case 7:
      case 8:
      case 9:
      case 12:
        stream << "\\" << oper_interface.operator_map[simple_stack(i, 0)]
               ->get_print() << "{" << strings[simple_stack(i, 1)] << "}";
        break;

      case 10:
        stream << "(" << strings[simple_stack(i, 1)] << ")^{(" <<
               strings[simple_stack(i, 2)] << ")}";
        break;

      case 11:
        stream << "|{" << strings[simple_stack(i, 1)] << "}|";
        break;

      default:
        stream << "";
        break;
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
  return simple_stack.rows();
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

  for (int i = 0; i < simple_stack.rows(); ++i) {
    out << std::left << std::setw(4) << i;
    out << "<= ";

    if (simple_stack(i, 0) == 0)
      out << oper_interface.operator_map[simple_stack(i, 0)]->get_print()
          << simple_stack(i, 1) << std::endl;

    else if (simple_stack(i, 0) == 1) {
      if (simple_stack(i, 1) == -1) {
        out << oper_interface.operator_map[simple_stack(i, 0)]->get_print();

      } else {
        out << constants[simple_stack(i, 1)];
      }

      out << std::endl;

    } else {
      out << "(" << simple_stack(i, 1) << ") "
          << oper_interface.operator_map[simple_stack(i, 0)]->get_print()
          << " (" << simple_stack(i, 2) << ")\n";
    }
  }

  return out.str();
}

AcyclicGraphManipulator::AcyclicGraphManipulator(int nvars, int ag_size,
    int nloads, float float_lim, float terminal_prob, int opt_rate) {
  this->nvars = nvars;
  this->ag_size = ag_size;
  this->nloads = nloads;
  this->float_lim = float_lim;
  this->terminal_prob = terminal_prob;
  this->opt_rate = opt_rate;
  num_node_types = 0;
  add_node_type(0);

  for (int i = 1; i < nvars; ++i) {
    term_vec.push_back(0);
  }

  add_node_type(1);
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
      term_vec.push_back(num_node_types);

    } else {
      op_vec.push_back(num_node_types);
    }

    num_node_types++;
  }
}

AcyclicGraph AcyclicGraphManipulator::generate() {
  AcyclicGraph indv = AcyclicGraph();
  Eigen::ArrayX3i array(ag_size, 3);
  float r = 0;
  std::vector<int> vec;

  for (int i = 0; i < ag_size; ++i) {
    r = static_cast <float> (rand()) / static_cast < float> (RAND_MAX);

    if (i < nloads || r < terminal_prob) {
      vec = rand_terminal();
      array(i, 0) = vec[0];
      array(i, 1) = vec[1];
      array(i, 2) = vec[2];

    } else {
      vec = rand_operator(i);
      array(i, 0) = vec[0];
      array(i, 1) = vec[1];
      array(i, 2) = vec[2];
    }
  }

  indv.stack = array;
  indv.opt_rate = opt_rate;
  simplify_stack(indv);
  return indv;
}

void AcyclicGraphManipulator::simplify_stack(AcyclicGraph &indv) {
  std::set<int> util = indv.utilized_commands();
  std::map<int, int> reduced;
  Eigen::ArrayX3i temp(util.size(), 3);

  if (opt_rate == 0) {
    int i = 0;

    for (std::set<int>::iterator it = util.begin(); it != util.end(); ++it) {
      reduced[*it] = i;
      temp(i, 0) = indv.stack(*it, 0);
      int arity = indv.oper_interface.operator_map[temp(i, 0)]->get_arity();

      if (arity == 0) {
        temp(i, 1) = indv.stack(*it, 1);
        temp(i, 2) = indv.stack(*it, 2);

      } else {
        temp(i, 1) = reduced[indv.stack(*it, 1)];
        temp(i, 2) = reduced[indv.stack(*it, 2)];
      }

      ++i;
    }

    indv.simple_stack = temp;

  } else {
    std::set<int>::iterator it = util.begin();
    int j = 0;
    int const_num = 0;

    for (int i = 0; i < indv.stack.rows(); ++i) {
      if (i == *it) {
        reduced[*it] = j;
        temp(j, 0) = indv.stack(*it, 0);

        if (temp(j, 0) == 0) {
          temp(j, 1) = indv.stack(*it, 1);
          temp(j, 2) = indv.stack(*it, 2);

        } else if (temp(j, 0) == 1) {
          if (indv.stack(*it, 1) == -1) {
            indv.needs_opt = true;
          }

          temp(j, 1) = const_num;
          temp(j, 2) = const_num;
          indv.stack(*it, 1) = const_num;
          indv.stack(*it, 2) = const_num;
          const_num += 1;

        } else {
          temp(j, 1) = reduced[indv.stack(*it, 1)];
          temp(j, 2) = reduced[indv.stack(*it, 2)];
        }

        ++j;
        ++it;

      } else {
        if (indv.stack(i, 0) == 1) {
          indv.stack(i, 1) = -1;
          indv.stack(i, 2) = -1;
        }
      }
    }

    indv.simple_stack = temp;

    if (const_num > indv.count_constants()) {
      indv.constants.resize(const_num);
      indv.needs_opt = true;
    }
  }
}

std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int>
AcyclicGraphManipulator::dump(AcyclicGraph &indv) {
  std::pair<Eigen::ArrayX3i, Eigen::VectorXd> temp(indv.stack, indv.constants);
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> dumped(temp,
      indv.genetic_age);
  return dumped;
}

AcyclicGraph AcyclicGraphManipulator::load(
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> indv_list) {
  AcyclicGraph temp = AcyclicGraph();
  temp.stack = indv_list.first.first;
  temp.constants = indv_list.first.second;
  temp.genetic_age = indv_list.second;
  simplify_stack(temp);
  return temp;
}

std::vector<AcyclicGraph> AcyclicGraphManipulator::crossover(
  AcyclicGraph &parent1, AcyclicGraph &parent2) {
  int cross = (rand() % (ag_size - 1)) + 1;
  std::vector<AcyclicGraph> temp;
  AcyclicGraph child1 = AcyclicGraph(parent1);
  AcyclicGraph child2 = AcyclicGraph(parent2);
  int parent_1_rows = parent1.stack.rows() - cross;
  int parent_2_rows = parent2.stack.rows() - cross;
  child1.stack.block(cross, 0, parent_1_rows, parent1.stack.cols()) =
    parent2.stack.block(cross, 0, parent_2_rows, parent2.stack.cols());
  child2.stack.block(cross, 0, parent_2_rows, parent2.stack.cols()) =
    parent1.stack.block(cross, 0, parent_1_rows, parent1.stack.cols());
  int max_gen_age = std::max(parent1.genetic_age, parent2.genetic_age);
  child1.genetic_age = max_gen_age;
  child2.genetic_age = max_gen_age;
  child1.fitness = std::vector<double>();
  child2.fitness = std::vector<double>();
  child1.fit_set = false;
  child2.fit_set = false;
  simplify_stack(child1);
  simplify_stack(child2);

  if (opt_rate == 2) {
    int child_1_cons = child1.count_constants();
    int child_2_cons = child2.count_constants();

    if (child_1_cons > 0 || child_2_cons > 0) {
      Eigen::VectorXd temp_const_vec_1(child_1_cons);
      Eigen::VectorXd temp_const_vec_2(child_2_cons);
      int i = 0;
      int const_1_loc = 0;
      int const_2_loc = 0;

      while (i < cross) {
        if (child1.stack(i, 0) == 1 && child1.stack(i, 1) != -1) {
          int con = parent1.stack(i, 1);

          if (con == -1) {
            child1.needs_opt = true;
            temp_const_vec_1(const_1_loc) = 0;

          } else {
            temp_const_vec_1(const_1_loc) = parent1.constants(con);
          }

          ++const_1_loc;
        }

        if (child2.stack(i, 0) == 1 && child2.stack(i, 1) != -1) {
          int con = parent2.stack(i, 1);

          if (con == -1) {
            child2.needs_opt = true;
            temp_const_vec_2(const_2_loc) = 0;

          } else {
            temp_const_vec_2(const_2_loc) = parent2.constants(con);
          }

          ++const_2_loc;
        }

        ++i;
      }

      while (i < parent1.stack.rows()) {
        if (child2.stack(i, 0) == 1 && child2.stack(i, 1) != -1) {
          int con = parent1.stack(i, 1);

          if (con == -1) {
            child2.needs_opt = true;
            temp_const_vec_2(const_2_loc) = 0;

          } else {
            temp_const_vec_2(const_2_loc) = parent1.constants(con);
          }

          ++const_2_loc;
        }

        if (child1.stack(i, 0) == 1 && child1.stack(i, 1) != -1) {
          int con = parent2.stack(i, 1);

          if (con == -1) {
            child1.needs_opt = true;
            temp_const_vec_1(const_1_loc) = 0;

          } else {
            temp_const_vec_1(const_1_loc) = parent2.constants(con);
          }

          ++const_1_loc;
        }

        ++i;
      }

      child1.set_constants(temp_const_vec_1);
      child2.set_constants(temp_const_vec_2);
    }
  }

  if (opt_rate == 3 || opt_rate == 5) {
    if (child1.count_constants() > 0) {
      child1.needs_opt = true;
    }

    if (child2.count_constants() > 0) {
      child2.needs_opt = true;
    }
  }

  temp.push_back(child1);
  temp.push_back(child2);
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
    int temp_node = 0;
    int temp_p1 = 0;
    int temp_p2 = 0;
    bool new_type_found = false;

    while (!new_type_found) {
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

      if (temp_node != orig_node_type || orig_node_type <= 1) {
        new_type_found = true;
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
      int pruned_param = 0;

      if (ran == 0) {
        pruned_param = new_param1;
      }

      // if random is 1
      else {
        pruned_param = new_param2;
      }

      for (int i = mut_point; i < indv.stack.rows(); ++i) {
        if (indv.stack(i, 0) > 1 && (mut_point == indv.stack(i, 1) ||
                                     mut_point == indv.stack(i, 2))) {
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
  }

  indv.fitness = std::vector<double>();
  indv.fit_set = false;
  simplify_stack(indv);

  if ((opt_rate == 4 || opt_rate == 5) && indv.count_constants() > 0) {
    indv.needs_opt = true;
  }

  return indv;
}

int AcyclicGraphManipulator::distance(AcyclicGraph &indv1,
                                      AcyclicGraph &indv2) {
  int tot = 0;

  for (int i = 0; i < indv1.stack.rows(); ++i) {
    if (indv1.stack(i, 0) != indv2.stack(i, 0)) {
      tot++;
    }

    if (indv1.stack(i, 1) != indv2.stack(i, 1)) {
      tot++;
    }

    if (indv1.stack(i, 2) != indv2.stack(i, 2)) {
      tot++;
    }
  }

  return tot;
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
  return node_type_vec[op_vec[rand() % op_vec.size()]];
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
  int node = node_type_vec[term_vec[rand() % term_vec.size()]];
  int param = rand_terminal_param(node);
  temp.push_back(node);
  temp.push_back(param);
  temp.push_back(param);
  return temp;
}