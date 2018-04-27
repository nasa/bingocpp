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
#include "training_data.cc"

double add(double i, double j) {
  return i + j;
}

namespace py = pybind11;

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
  py::class_<AcyclicGraph>(m, "AcyclicGraph")
  .def(py::init<>())
  .def(py::init<AcyclicGraph &>())
  .def_readwrite("stack", &AcyclicGraph::stack)
  .def_readwrite("constants", &AcyclicGraph::constants)
  .def_readwrite("fitness", &AcyclicGraph::fitness)
  .def_readwrite("fit_set", &AcyclicGraph::fit_set)
  .def_readwrite("genetic_age", &AcyclicGraph::genetic_age)
  .def("copy", &AcyclicGraph::copy)
  .def("needs_optimization", &AcyclicGraph::needs_optimization)
  .def("set_constants", &AcyclicGraph::set_constants)
  .def("count_constants", &AcyclicGraph::count_constants)
  .def("evaluate", &AcyclicGraph::evaluate)
  .def("evaluate_deriv", &AcyclicGraph::evaluate_deriv)
  .def("evaluate_with_const_deriv", &AcyclicGraph::evaluate_with_const_deriv)
  .def("latexstring", &AcyclicGraph::latexstring)
  .def("utilized_commands", &AcyclicGraph::utilized_commands)
  .def("complexity", &AcyclicGraph::complexity)
  .def("__str__", &AcyclicGraph::print_stack);
  py::class_<AcyclicGraphManipulator>(m, "AcyclicGraphManipulator")
  .def(py::init<int &, int &, int &, float &, float &, int &>(),
       py::arg("nvars") = 3, py::arg("ag_size") = 15, py::arg("nloads") = 1,
       py::arg("float_lim") = 10.0, py::arg("terminal_prob") = 0.1,
       py::arg("opt_rate") = 0)
  .def("add_node_type", &AcyclicGraphManipulator::add_node_type)
  .def("generate", &AcyclicGraphManipulator::generate)
  .def("simplify_stack", &AcyclicGraphManipulator::simplify_stack)
  .def("dump", &AcyclicGraphManipulator::dump)
  .def("load", &AcyclicGraphManipulator::load)
  .def("crossover", &AcyclicGraphManipulator::crossover)
  .def("mutation", &AcyclicGraphManipulator::mutation)
  .def("distance", &AcyclicGraphManipulator::distance)
  .def("rand_operator_params", &AcyclicGraphManipulator::rand_operator_params)
  .def("rand_operator_type", &AcyclicGraphManipulator::rand_operator_type)
  .def("rand_operator", &AcyclicGraphManipulator::rand_operator)
  .def("rand_terminal_param", &AcyclicGraphManipulator::rand_operator_params)
  .def("mutate_terminal_param", &AcyclicGraphManipulator::rand_operator_type)
  .def("rand_terminal", &AcyclicGraphManipulator::rand_operator);
  py::class_<FitnessMetric>(m, "FitnessMetric")
  //  .def(py::init<>())
  .def("evaluate_fitness", &FitnessMetric::evaluate_fitness)
  .def("optimize_constants", &FitnessMetric::optimize_constants);
  py::class_<StandardRegression, FitnessMetric>(m, "StandardRegression")
  .def(py::init<>())
  .def("evaluate_fitness_vector", &StandardRegression::evaluate_fitness_vector);
  py::class_<ImplicitRegression, FitnessMetric>(m, "ImplicitRegression")
  .def(py::init<int &, bool &, double &>(),
       py::arg("required_params") = 0, py::arg("normalize_dot") = false,
       py::arg("acceptable_nans") = 0.1)
  .def("evaluate_fitness_vector", &ImplicitRegression::evaluate_fitness_vector);
  py::class_<TrainingData>(m, "TrainingData");
  py::class_<ExplicitTrainingData, TrainingData>(m, "ExplicitTrainingData")
  .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd &>())
  .def("__getitem__", &ExplicitTrainingData::get_item)
  .def("size", &ExplicitTrainingData::size);
  py::class_<ImplicitTrainingData, TrainingData>(m, "ImplicitTrainingData")
  .def(py::init<Eigen::ArrayXXd &>())
  .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd &>())
  .def("__getitem__", &ImplicitTrainingData::get_item)
  .def("size", &ImplicitTrainingData::size);
  m.def("calculate_partials", &calculate_partials);
  m.def("savitzky_golay", &savitzky_golay);
  m.def("GenFact", &GenFact);
  m.def("GramPoly", &GramPoly);
  m.def("GramWeight", &GramWeight);
}





