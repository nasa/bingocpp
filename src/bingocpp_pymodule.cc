/*!
 * \file bingocpp_pymodule.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the python bindings of the BingoCpp library.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "BingoCpp/acyclic_graph.h"
#include "graph_manip.cc"
#include "fitness_metric.cc"

double add(double i, double j) {
  return i + j;
}

namespace py = pybind11;

// class PyFitnessMetric : public FitnessMetric {
// public:
//   using FitnessMetric::FitnessMetric;
//   float evaluate_metric(AGraphCpp &indv, Eigen::ArrayXXd &eval_x, 
//                                          Eigen::ArrayXXd &eval_y) override 
//                        { PYBIND11_OVERLOAD(float, FitnessMetric,
//                        evaluate_metric, indv, eval_x, eval_y);}
// };

// class PyStandardRegression : public StandardRegression {
// public:
//   using StandardRegression::StandardReression;
//   float evaluate_metric(AGraphCpp &indv, Eigen::ArrayXXd &eval_x, 
//                                          Eigen::ArrayXXd &eval_y) override 
//                        { PYBIND11_OVERLOAD(float, FitnessMetric,
//                        evaluate_metric, indv, eval_x, eval_y);}
// };


PYBIND11_MODULE(bingocpp, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring
  m.def("evaluate", &Evaluate, "evaluate");
  m.def("simplify_and_evaluate", &SimplifyAndEvaluate,
        "evaluate after simplification");
  m.def("evaluate_with_derivative", &EvaluateWithDerivative,
        "evaluate with derivative");
  m.def("simplify_and_evaluate_with_derivative",
        &SimplifyAndEvaluateWithDerivative,
        "evaluate with derivative after simplification");
  
  m.def("rand_init", &rand_init);
  
  py::class_<AGraphCpp>(m, "AGraphCpp")
    .def(py::init<>())
    .def(py::init<AGraphCpp &>())
    .def_readwrite("stack", &AGraphCpp::stack)
    .def("needs_optimization", &AGraphCpp::needs_optimization)
    .def("set_constants", &AGraphCpp::set_constants)
    .def("count_constants", &AGraphCpp::count_constants)
    .def("evaluate", &AGraphCpp::evaluate)
    .def("evaluate_deriv", &AGraphCpp::evaluate_deriv)
    .def("latexstring", &AGraphCpp::latexstring)
    .def("utilized_commands", &AGraphCpp::utilized_commands)
    .def("complexity", &AGraphCpp::complexity)
    .def("__str__", &AGraphCpp::print_stack); 
            
  py::class_<AGraphCppManipulator>(m, "AGraphCppManipulator")
    .def(py::init<int &, int &, int &, float &, float &>(), 
       py::arg("nvars")=3, py::arg("ag_size")=15, py::arg("nloads")=1, 
       py::arg("float_lim")=10.0, py::arg("terminal_prob")=0.1)
    .def("add_node_type", &AGraphCppManipulator::add_node_type)
    .def("generate", &AGraphCppManipulator::generate)
    .def("crossover", &AGraphCppManipulator::crossover)
    .def("mutation", &AGraphCppManipulator::mutation)
    .def("distance", &AGraphCppManipulator::distance)
    .def("rand_operator_params", &AGraphCppManipulator::rand_operator_params)
    .def("rand_operator_type", &AGraphCppManipulator::rand_operator_type)
    .def("rand_operator", &AGraphCppManipulator::rand_operator) 
    .def("rand_terminal_param", &AGraphCppManipulator::rand_operator_params)
    .def("mutate_terminal_param", &AGraphCppManipulator::rand_operator_type)
    .def("rand_terminal", &AGraphCppManipulator::rand_operator); 

  py::class_<FitnessMetric>(m, "FitnessMetric")
     .def(py::init<>())
     .def("evaluate_metric", &FitnessMetric::evaluate_metric);

  py::class_<StandardRegression, FitnessMetric>(m, "StandardRegression")
     .def(py::init<>())
     .def("evaluate_vector", &StandardRegression::evaluate_vector);
}





