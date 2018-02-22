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
//#include <pybind11/stl_bind.h>

//PYBIND11_MAKE_OPAQUE(CommandStack);

double add(double i, double j) {
  return i + j;
}

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
  
  //pybind11::bind_vector<CommandStack>(m, "CommandStack");
}



