/*!
 * \file acyclic_graph.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the functions associated with an acyclic graph
 * representation of a symbolic equation.
 */
#include <iostream>
#include <iomanip>
#include <vector>

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/backend.h"
#include "BingoCpp/acyclic_graph_nodes.h"

namespace bingo {

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

const char *AcyclicGraph::stack_print_map[13] = {
  "X",
  "C",
  "+",
  "-",
  "*",
  "/",
  "sin",
  "cos",
  "exp",
  "log",
  "pow",
  "abs",
  "sqrt"
};

const bool AcyclicGraph::is_arity_2_map[13] = {
  false,
  false,
  true,
  true,
  true,
  true,
  false,
  false,
  false,
  false,
  true,
  false,
  false
};

const bool AcyclicGraph::is_terminal_map[13] = {
  true,
  true,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false
};

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

// void AcyclicGraph::input_constants() {
//   std::set<int> util = utilized_commands();
//   std::set<int>::iterator it = util.begin();
//   int const_num = 0;
//   int j = 0;
//   for (int i = 0; i < stack.rows(); ++i) {
//     if (i == *it) {
//       if (stack(i, 0) == 1) {
//         simple_stack(j, 1) = const_num;
//         simple_stack(j, 2) = const_num;
//         stack(*it, 1) = const_num;
//         stack(*it, 2) = const_num;
//         const_num += 1;
//       }
//       ++it;
//       ++j;
//     }
//     else
//       if (stack(i, 0) == 1) {
//         stack(i, 1) = -1;
//         stack(i, 2) = -1;
//       }
//   }

//   // if (const_num != count_constants()) {
//   //   Eigen::VectorXd temp(const_num);
//   //   constants = temp;
//   // }
//   if (const_num != count_constants())
//     // std::cout << "Constants are too big\n";
//     constants.conservativeResize(const_num);
// }

Eigen::ArrayXXd AcyclicGraph::evaluate(Eigen::ArrayXXd &eval_x) {
  return backend::evaluate(simple_stack, eval_x, constants);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> AcyclicGraph::evaluate_deriv(
  Eigen::ArrayXXd &eval_x) {
  return backend::evaluateWithDerivative(simple_stack, eval_x, constants);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>
AcyclicGraph::evaluate_with_const_deriv(
  Eigen::ArrayXXd &eval_x) {
  return backend::evaluateWithDerivative(simple_stack, eval_x, constants, false);
}


// TODO: remove iterator interface
std::string AcyclicGraph::latexstring() {
  std::vector<std::string> strings;
  std::ostringstream stream;

  for (int i = 0; i < simple_stack.rows(); ++i) {
    std::string temp = "";

    switch ((int)simple_stack(i, 0)) {
      case 0:
        stream << AcyclicGraph::get_print(simple_stack(i, 0))
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
        stream << "\\" << AcyclicGraph::get_print(simple_stack(i, 0))
               << "{" << strings[simple_stack(i, 1)] << "}";
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
      out << AcyclicGraph::get_print(stack(i, 0))
          << stack(i, 1) << std::endl;

    else if (stack(i, 0) == 1) {
      if (stack(i, 1) == -1) {
        out << AcyclicGraph::get_print(stack(i, 0));

      } else {
        out << constants[stack(i, 1)];
      }

      out << std::endl;

    } else {
      out << "(" << stack(i, 1) << ") "
          << AcyclicGraph::get_print(stack(i, 0))
          << " (" << stack(i, 2) << ")\n";
    }
  }

  out << "---small stack---\n";

  for (int i = 0; i < simple_stack.rows(); ++i) {
    out << std::left << std::setw(4) << i;
    out << "<= ";

    if (simple_stack(i, 0) == 0)
      out << AcyclicGraph::get_print(simple_stack(i, 0))
          << simple_stack(i, 1) << std::endl;

    else if (simple_stack(i, 0) == 1) {
      if (simple_stack(i, 1) == -1) {
        out << AcyclicGraph::get_print(simple_stack(i, 0));

      } else {
        out << constants[simple_stack(i, 1)];
      }

      out << std::endl;

    } else {
      out << "(" << simple_stack(i, 1) << ") "
          << AcyclicGraph::get_print(simple_stack(i, 0))
          << " (" << simple_stack(i, 2) << ")\n";
    }
  }

  return out.str();
}

bool AcyclicGraph::has_arity_two(int node) {
  return is_arity_2_map[node];
}

bool AcyclicGraph::is_terminal(int node) {
  return is_terminal_map[node];
}

const char *AcyclicGraph::get_print(int node) {
  return stack_print_map[node];
}
} // namespace bingo 