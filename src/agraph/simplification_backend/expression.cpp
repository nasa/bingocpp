#include <bingocpp/agraph/simplification_backend/expression.h>

#include <iostream>


namespace bingo {
namespace simplification_backend {

TermExpression::TermExpression(const int op, const int param):
  Expression(op), param_(param){}



std::ostream &operator<<(std::ostream &strm, const TermExpression& expr){
  if (expr.op_ == kVariable) {
    strm << "X";
  } else if (expr.op_ == kConstant) {
    strm << "C";
  }
  return strm << expr.param_;
}


} // namespace simplification_backend
} // namespace bingo