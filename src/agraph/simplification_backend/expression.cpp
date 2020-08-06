#include <bingocpp/agraph/simplification_backend/expression.h>

#include <iostream>


namespace bingo {
namespace simplification_backend {

TermExpression::TermExpression(const Op op, const int param):
  param_(param){ op_ = op; }



std::ostream &TermExpression::print(std::ostream &strm) const {
  if (op_ == kVariable) {
    strm << "X";
  } else if (op_ == kConstant) {
    strm << "C";
  }
  return strm << param_;
}

bool TermExpression::equal(const Expression& other) const {
  const TermExpression& cast_other =
      static_cast<const TermExpression&>(other);
  return param_ == cast_other.param_;
}




OpExpression::OpExpression(const Op op,
                           const std::vector<std::shared_ptr<Expression>> params):
  params_(params){ op_ = op; }

bool OpExpression::IsConstantValued() const  {
  for (auto i : params_ ) {
    if ( !(i->IsConstantValued()) ) { return false; };
  }
  return true;
};


bool OpExpression::equal(const Expression& other) const {
  const OpExpression& cast_other =
      static_cast<const OpExpression&>(other);
  if (params_.size() != cast_other.params_.size()) { return false; }
  for (unsigned int i=0; i < params_.size(); ++i) {
    if (*params_[i] != *(cast_other.params_[i])) { return false; }
  }
  return true;
}


std::ostream &OpExpression::print(std::ostream &strm) const {
  strm << op_ << "(";
  for (auto i : params_ ) {
    strm << *i << ", ";
  }
  return strm << ")";
}


} // namespace simplification_backend
} // namespace bingo