/*
 * Copyright 2018 United States Government as represented by the Administrator
 * of the National Aeronautics and Space Administration. No copyright is claimed
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
*/
#ifndef INCLUDE_BINGOCPP_SIMPLIFICATION_BACKEND_EXPRESSION_H
#define INCLUDE_BINGOCPP_SIMPLIFICATION_BACKEND_EXPRESSION_H

#include <vector>
#include <iostream>

#include <bingocpp/agraph/operator_definitions.h>

namespace bingo {
namespace simplification_backend {

class Expression {
  public:
    Expression(int op) : op_(op) {}
    virtual ~Expression() = default;

    int GetOperator() const { return op_; }
    virtual bool IsZero() const = 0;
    virtual bool IsOne() const = 0;
    virtual bool IsConstantValued() const = 0;
    //virtual std::vector<std::string> DependsOn() const = 0;
    //virtual Expression GetBase() const = 0;
    //virtual Expression GetExponent() const = 0;
    //virtual Expression GetTerm() const = 0;
    //virtual Expression GetCoefficient() const = 0;


 protected:
   int op_;
};


class TermExpression : public Expression {
  public:
    TermExpression(const int op, const int param);
    virtual ~TermExpression() = default;

    inline bool IsZero() const {
      return op_ == kInteger && param_ == 0;
    }
    inline bool IsOne() const {
      return op_ == kInteger && param_ == 1;
    }
    inline bool IsConstantValued() const {
      return op_ != kVariable; // op_ == kConstant || op_ == kInteger;
    }

    //std::vector<std::string> DependsOn() const;
    //Expression GetBase() const;
    //Expression GetExponent() const;
    //Expression GetTerm() const;
    //Expression GetCoefficient() const;



    inline bool operator==(const TermExpression& other) const {
      return op_ == other.op_ && param_ == other.param_;
    }
    inline bool operator!=(const TermExpression& other) const {
      return !(*this == other);
    }

    friend std::ostream &operator<<(std::ostream &strm,
                                    const TermExpression& expr);


  private:
    int param_;

};

/*
class OpExpression : public Expression {
  public:
    OpExpression(const int op, const std::vector<Expression> params);
    virtual ~OpExpression() = default;

    bool IsConstantValued() const;
    std::vector<std::string> DependsOn() const;
    Expression GetBase() const;
    Expression GetExponent() const;
    Expression GetTerm() const;
    Expression GetCoefficient() const;

  private:
    std::vector<Expression> params_;

}
*/



} // namespace simplification_backend
} // namespace bingo
#endif