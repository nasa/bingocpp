/*!
 * \file symbolic_regression_pymodule.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the python bindings of the symbolic_regression library.
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

#include <Eigen/Dense> 

#include "bingocpp/agraph/agraph.h"
#include "bingocpp/explicit_regression.h"
#include "bingocpp/implicit_regression.h"
#include "bingocpp/utils.h"


namespace py = pybind11;
using namespace bingo;

// Todo: Rename symbolic_regression to _symbolic_regression to indicate private
// module
PYBIND11_MODULE(symbolic_regression, m) {
  m.doc() = "The symbolic regression module";  // optional module docstring


  
  py::class_<ImplicitTrainingData>(m, "ImplicitTrainingData")
    .def(py::init<Eigen::ArrayXXd &>())
    .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd &>())
    .def_readwrite("x", &ImplicitTrainingData::x)
    .def_readwrite("dx_dt", &ImplicitTrainingData::dx_dt)
    .def("__getitem__", 
         (ImplicitTrainingData *(ImplicitTrainingData::*)(int))
         &ImplicitTrainingData::GetItem)
    .def("__getitem__",
         (ImplicitTrainingData *(ImplicitTrainingData::*)(const std::vector<int>&))
         &ImplicitTrainingData::GetItem)
    .def("__len__", &ImplicitTrainingData::Size)
    .def("__getstate__", &ImplicitTrainingData::DumpState)
    .def("__setstate__", [](ImplicitTrainingData &td, const ImplicitTrainingDataState &state) {
            new (&td) ImplicitTrainingData(state); });
  
  py::class_<ExplicitTrainingData>(m, "ExplicitTrainingData")
    .def(py::init<Eigen::ArrayXXd &, Eigen::ArrayXXd&>())
    .def_readwrite("x", &ExplicitTrainingData::x)
    .def_readwrite("y", &ExplicitTrainingData::y)
    .def("__getitem__", 
         (ExplicitTrainingData *(ExplicitTrainingData::*)(int))
         &ExplicitTrainingData::GetItem)
    .def("__getitem__",
         (ExplicitTrainingData *(ExplicitTrainingData::*)(const std::vector<int>&))
         &ExplicitTrainingData::GetItem)
    .def("__len__", &ExplicitTrainingData::Size)
    .def("__getstate__", &ExplicitTrainingData::DumpState)
    .def("__setstate__", [](ExplicitTrainingData &td, const ExplicitTrainingDataState &state) {
            new (&td) ExplicitTrainingData(state); });
  
  py::class_<ExplicitRegression>(m, "ExplicitRegression")
    .def(py::init<ExplicitTrainingData *, std::string &, bool &>(),
        py::arg("training_data"),
        py::arg("metric")="mae",
        py::arg("relative")=false)
    .def_property("eval_count",
                  &ExplicitRegression::GetEvalCount,
                  &ExplicitRegression::SetEvalCount)
    .def("__call__", &ExplicitRegression::EvaluateIndividualFitness)
    .def("evaluate_fitness_vector", &ExplicitRegression::EvaluateFitnessVector)
    .def("__getstate__", &ExplicitRegression::DumpState)
    .def("__setstate__", [](ExplicitRegression &r, const ExplicitRegressionState &state) {
            new (&r) ExplicitRegression(state); });
  
  py::class_<ImplicitRegression>(m, "ImplicitRegression")
    .def(py::init<ImplicitTrainingData *, int &, std::string &>(),
         py::arg("training_data"),
         py::arg("required_params") = -1,
         py::arg("metric") = "mae")
    .def_property("eval_count",
                  &ImplicitRegression::GetEvalCount,
                  &ImplicitRegression::SetEvalCount)
    .def("__call__", &ImplicitRegression::EvaluateIndividualFitness)
    .def("evaluate_fitness_vector", &ImplicitRegression::EvaluateFitnessVector)
    .def("__getstate__", &ImplicitRegression::DumpState)
    .def("__setstate__", [](ImplicitRegression &r, const ImplicitRegressionState &state) {
            new (&r) ImplicitRegression(state); });
}