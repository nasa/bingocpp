/*!
 * \file bingocpp_pymodule.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the python bindings of the BingoCpp library.
 * 
 * Notices
 * -------
 * Copyright 2018 United States Government as represented by the Administrator of 
 * the National Aeronautics and Space Administration. No copyright is claimed in 
 * the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *  
 * 
 * Disclaimers
 * -----------
 * No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
 * ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
 * TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
 * IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
 * FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
 * FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
 * SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
 * ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
 * RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
 * RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
 * DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF 
 * PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 * 
 * Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
 * UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
 * PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
 * LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
 * INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE 
 * OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED 
 * STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR 
 * RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY 
 * SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/backend.h"
#include "BingoCpp/graph_manip.h"
#include "BingoCpp/fitness_metric.h"
#include "BingoCpp/training_data.h"
#include "BingoCpp/utils.h"

double add(double i, double j) {
  return i + j;
}

namespace py = pybind11;
using namespace bingo;

PYBIND11_MODULE(bingocpp, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring
  m.def("is_cpp", &backend::IsCpp, "is the backend c++");
  m.def("evaluate", &backend::Evaluate, "evaluate");
  m.def("simplify_and_evaluate", &backend::SimplifyAndEvaluate,
        "evaluate after simplification");
  m.def("evaluate_with_derivative", &backend::EvaluateWithDerivative,
        "evaluate with derivative");
  m.def("simplify_and_evaluate_with_derivative",
        &backend::SimplifyAndEvaluateWithDerivative,
        "evaluate with derivative after simplification");
  m.def("get_utilized_commands",
        &backend::GetUtilizedCommands,
        "get the commands that are utilized in a stack");
  m.def("simplify_stack", &backend::SimplifyStack,
        "simplify stack to only utilized commands");
        
        
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
//   .def("input_constants", &AcyclicGraph::input_constants)
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
  .def_readwrite("x", &ExplicitTrainingData::x)
  .def_readwrite("y", &ExplicitTrainingData::y)
  .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd &>())
  .def("__getitem__", &ExplicitTrainingData::get_item)
  .def("size", &ExplicitTrainingData::size);
  py::class_<ImplicitTrainingData, TrainingData>(m, "ImplicitTrainingData")
  .def_readwrite("x", &ImplicitTrainingData::x)
  .def_readwrite("dx_dt", &ImplicitTrainingData::dx_dt)
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





