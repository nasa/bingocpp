#include <bingocpp/agraph/simplification_backend/expression.h>
#include <bingocpp/agraph/simplification_backend/constant_expressions.h>

#include <iostream>


namespace bingo {
namespace simplification_backend {

TermExpression::TermExpression(const Op operatr, const int operand):
  operand_(operand){ operator_ = operatr; }



std::ostream &TermExpression::print(std::ostream &strm) const {
  if (operator_ == kVariable) {
    strm << "X";
  } else if (operator_ == kConstant) {
    strm << "C";
  }
  return strm << operand_;
}

bool TermExpression::equal(const Expression& other) const {
  const TermExpression& cast_other =
      static_cast<const TermExpression&>(other);
  return operand_ == cast_other.operand_;
}

inline std::shared_ptr<const Expression> TermExpression::GetExponent() const
  { return kOne; }




OpExpression::OpExpression(const Op operatr,
                           const std::vector<std::shared_ptr<Expression>> operands):
  operands_(operands){ operator_ = operatr; }

bool OpExpression::IsConstantValued() const  {
  for (auto i : operands_ ) {
    if ( !(i->IsConstantValued()) ) { return false; };
  }
  return true;
};


bool OpExpression::equal(const Expression& other) const {
  const OpExpression& cast_other =
      static_cast<const OpExpression&>(other);
  if (operands_.size() != cast_other.operands_.size()) { return false; }
  for (unsigned int i=0; i < operands_.size(); ++i) {
    if (*operands_[i] != *(cast_other.operands_[i])) { return false; }
  }
  return true;
}


std::ostream &OpExpression::print(std::ostream &strm) const {
  strm << operator_ << "(";
  for (auto i : operands_ ) {
    strm << *i << ", ";
  }
  return strm << ")";
}

std::shared_ptr<const Expression> OpExpression::GetBase() const {
  if (operator_ == kPower) { return operands_[0]; }
  return shared_from_this();
}


std::shared_ptr<const Expression> OpExpression::GetExponent() const {
  if (operator_ == kPower) { return operands_[1]; }
  return kOne;
}


} // namespace simplification_backend
} // namespace bingo