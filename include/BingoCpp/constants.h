#ifndef INCLUDE_BINGOCPP_CONSTANTS_H_
#define INCLUDE_BINGOCPP_CONSTANTS_H_

namespace bingo {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

enum ArrayProps : unsigned int {
  kNodeIdx = 0,
  kOp1 = 1,
  kOp2 = 2,
  kArrayCols = 3
};

enum Op : signed int {
  LOAD_X = 0,
  LOAD_C = 1,
  C_OPTIMIZE = -1,
  ADD = 2,
  SUB = 3,
  MULT = 4,
  DIV = 5,
  SIN = 6,
  COS = 7,
  EXP = 8,
  LOG = 9,
  POW = 10,
  ABS = 11,
  SQRT = 12
};
}
#endif
