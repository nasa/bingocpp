/*!
 * \file acyclic_graph_nodes.hh
 *
 * \author Ethan Adams
 * \date 2/9/2018
 *
 * This is the header file to hold the Operation abstract class
 * and all implementations of that class. Also holds the OperatorInterface
 * class, which includes a map to keep the Operation in
 */

#ifndef INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_
#define INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_

#include <Eigen/Dense>
#include <Eigen/Core>

#include <set>
#include <map>
#include <utility>
#include <vector>
#include <string>

typedef std::vector< std::pair<int, std::vector<int> > > CommandStack;
typedef std::pair<int, std::vector<int> > SingleCommand;
typedef Eigen::Ref<Eigen::ArrayXXd,
        0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ArrayByRef;

/*! \class Operation
 *  \brief Abstract class for an operation.
 *
 *  This is the abstract class for the Operation being performed.
 *
 *  \note Operators include : X_Load, C_Load, Addition, Subtraction,
 *        Multiplication, Division, sin, cos, exp, log, pow, abs, sqrt
 *
 *  \fn virtual int get_arity()
 *  \fn virtual std::string get_print()
 *  \fn void evaluate(const std::vector<int> &command,
 *                               const Eigen::ArrayXXd &x,
 *                               const std::vector<double> &constants,
 *                               std::vector<Eigen::ArrayXXd> &buffer,
 *                               std::size_t result_location)
 *  \fn virtual void deriv_evaluate(const std::vector<int> &command,
 *                                  const int command_index,
 *                                  const std::vector<Eigen::ArrayXXd> &forward_buffer,
 *                                  std::vector<Eigen::ArrayXXd> &reverse_buffer,
 *                                  int dependency)
 */
class Operation {
 public:
  //!\brief Returns how many parameters the operation requires.
  virtual int get_arity() = 0;

  //! \brief Returns the string representation of the operator.
  virtual std::string get_print() = 0;

  /*! \brief evaluates single command stack, returns array to be saved in buffer.
   *
   *  \param[in] command The parameters to be input to buffer. std::vector<int>
   *  \param[in] x Input variables to the acyclic graph. Eigen::ArrayXXd
   *  \param[in] constants Constants used in the command. std::vector<double>
   *  \param[in/out] buffer Vector of Eigen arrays for the buffer.
   *  \param[in] result_location location to use with buffer.
   */
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location) = 0;

  /*! \brief Computes reverse autodiff partial of a command stack.
   *
   *  \param[in] command The parameters to be input to buffer. std::vector<int>
   *  \param[in] command_index Index of command in the command; also the location of
   *                          the result to be placed in the reverse buffer.
   *  \param[in] forward_buffer Vector of Eigen arrays for the forward buffer.
   *  \param[in\out] forward_buffer Vector of Eigen arrays for the forward buffer.
   *  \param[in] dependency Int for location of where dependency is located in buffer.
   */
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) = 0;
};


/*! \class X_Load
 *  \brief This class loads the X.
 */

class X_Load: public Operation {
 public:
  int get_arity() {
    return 0;
  }
  std::string get_print() {
    return "X";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  X_Load();
};

/*! \class C_Load
 *  \brief This class loads the constant.
 */

class C_Load: public Operation {
 public:
  int get_arity() {
    return 0;
  }
  std::string get_print() {
    return "C";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  C_Load();
};

/*! \class Addition
 *  \brief This class performs addition.
 */

class Addition: public Operation {
 public:
  int get_arity() {
    return 2;
  }
  std::string get_print() {
    return "+";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Addition();
};

/*! \class Subtraction
 *  \brief This class performs subtraction.
 */

class Subtraction: public Operation {
 public:
  int get_arity() {
    return 2;
  }
  std::string get_print() {
    return "-";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Subtraction();
};

/*! \class Multiplication
 *  \brief This class performs multiplication.
 */

class Multiplication: public Operation {
 public:
  int get_arity() {
    return 2;
  }
  std::string get_print() {
    return "*";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Multiplication();
};

/*! \class Division
 *  \brief This class performs division.
 */

class Division: public Operation {
 public:
  int get_arity() {
    return 2;
  }
  std::string get_print() {
    return "/";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Division();
};

/*! \class Sin
 *  \brief This class performs sine.
 */

class Sin: public Operation {
 public:
  int get_arity() {
    return 1;
  }
  std::string get_print() {
    return "sin";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Sin();
};

/*! \class Cos
 *  \brief This class performs cosine.
 */

class Cos: public Operation {
 public:
  int get_arity() {
    return 1;
  }
  std::string get_print() {
    return "cos";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Cos();
};

/*! \class Exp
 *  \brief This class performs an exponential.
 */

class Exp: public Operation {
 public:
  int get_arity() {
    return 1;
  }
  std::string get_print() {
    return "exp";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Exp();
};

/*! \class Log
 *  \brief This class performs a safe log.
 */

class Log: public Operation {
 public:
  int get_arity() {
    return 1;
  }
  std::string get_print() {
    return "log";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Log();
};

/*! \class Power
 *  \brief This class performs power.
 */

class Power: public Operation {
 public:
  int get_arity() {
    return 2;
  }
  std::string get_print() {
    return "pow";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Power();
};

/*! \class Absolute
 *  \brief This class performs an absolute.
 */

class Absolute: public Operation {
 public:
  int get_arity() {
    return 1;
  }
  std::string get_print() {
    return "abs";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Absolute();
};

/*! \class Sqrt
 *  \brief This class performs square root.
 */

class Sqrt: public Operation {
 public:
  int get_arity() {
    return 1;
  }
  std::string get_print() {
    return "sqrt";
  }
  virtual void evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location);
  virtual void deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency);
  Sqrt();
};

/*! \class OperatorInterface
 *
 *  \brief Populate and hold the map of operations
 *
 *  \var stat std::map<int, Operation*> operator_map
 *  \brief Map that holds the int location to the Operation
 *
 *  \fn static std::map<int, Operation*> create_op_map()
 *  \brief Populates the map with each implementation of operation.
 */
class OperatorInterface {
 public:
  static std::map<int, Operation*> operator_map;
  static std::map<int, Operation*> create_op_map();
};
#endif  // INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_
