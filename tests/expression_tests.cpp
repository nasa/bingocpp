#include <memory>
#include <gtest/gtest.h>
#include <iostream>

#include <bingocpp/agraph/simplification_backend/expression.h>
#include <bingocpp/agraph/operator_definitions.h>


using namespace bingo;
using namespace simplification_backend;
namespace {

class ExpressionTest : public testing::Test {
  protected:
    std::shared_ptr<Expression> zero =
        std::make_shared<TermExpression>(kInteger, 0);
    std::shared_ptr<Expression> one =
        std::make_shared<TermExpression>(kInteger, 1);
    std::shared_ptr<Expression> x0 =
        std::make_shared<TermExpression>(kVariable, 0);
    std::shared_ptr<Expression> c0 =
        std::make_shared<TermExpression>(kConstant, 0);

    std::vector<std::shared_ptr<Expression>> x0squared_params =
        {std::make_shared<TermExpression>(kVariable, 0),
         std::make_shared<TermExpression>(kVariable, 0)} ;
    std::shared_ptr<Expression> x0squared =
        std::make_shared<OpExpression>(kMultiplication, x0squared_params);

    std::vector<std::shared_ptr<Expression>> threec0_params =
        {std::make_shared<TermExpression>(kConstant, 0),
         std::make_shared<TermExpression>(kConstant, 0),
         std::make_shared<TermExpression>(kConstant, 0)} ;
    std::shared_ptr<Expression> threec0 =
        std::make_shared<OpExpression>(kAddition, threec0_params);

    std::vector<std::shared_ptr<Expression>> c0x0_params =
        {std::make_shared<TermExpression>(kConstant, 0),
         std::make_shared<TermExpression>(kVariable, 0)} ;
    std::shared_ptr<Expression> c0x0 =
        std::make_shared<OpExpression>(kAddition, c0x0_params);

    std::vector<std::shared_ptr<Expression>> x0cubed_params =
        {std::make_shared<TermExpression>(kVariable, 0),
         std::make_shared<TermExpression>(kInteger, 3)} ;
    std::shared_ptr<Expression> x0cubed =
        std::make_shared<OpExpression>(kPower, x0cubed_params);

};

TEST_F(ExpressionTest, ExpressionOperator) {
    EXPECT_EQ (kInteger, zero->GetOperator());
    EXPECT_EQ (kInteger, one->GetOperator());
    EXPECT_EQ (kVariable, x0->GetOperator());
    EXPECT_EQ (kConstant, c0->GetOperator());
    EXPECT_EQ (kMultiplication, x0squared->GetOperator());
}

TEST_F(ExpressionTest, ExpressionIsOne) {
    EXPECT_TRUE (one->IsOne());
    EXPECT_TRUE (!zero->IsOne());
    EXPECT_TRUE (!x0->IsOne());
    EXPECT_TRUE (!c0->IsOne());
    EXPECT_TRUE (!x0squared->IsOne());
}

TEST_F(ExpressionTest, ExpressionIsZero) {
    EXPECT_TRUE (zero->IsZero());
    EXPECT_TRUE (!one->IsZero());
    EXPECT_TRUE (!x0->IsZero());
    EXPECT_TRUE (!c0->IsZero());
    EXPECT_TRUE (!x0squared->IsZero());
}

TEST_F(ExpressionTest, ExpressionIsConstantValued) {
    EXPECT_TRUE (zero->IsConstantValued());
    EXPECT_TRUE (one->IsConstantValued());
    EXPECT_TRUE (!x0->IsConstantValued());
    EXPECT_TRUE (c0->IsConstantValued());
    EXPECT_TRUE (!x0squared->IsConstantValued());
    EXPECT_TRUE (threec0->IsConstantValued());
}

TEST_F(ExpressionTest, ExpressionEquality) {
    std::shared_ptr<Expression> zero_duplicate =
        std::make_shared<TermExpression>(kInteger, 0);
    EXPECT_EQ (*zero, *zero_duplicate);
    EXPECT_NE (*zero, *one);
    EXPECT_NE (*zero, *x0);
    EXPECT_NE (*zero, *c0);
    EXPECT_NE (*zero, *x0squared);
    EXPECT_NE (*x0squared, *zero);
    EXPECT_NE (*x0squared, *threec0);
    EXPECT_NE (*x0squared, *c0x0);
}

TEST_F(ExpressionTest, ExpressionBase) {
    EXPECT_EQ (*x0, *(x0cubed->GetBase()) );
    EXPECT_EQ (*c0, *(c0->GetBase()) );
    EXPECT_EQ (*c0x0, *(c0x0->GetBase()) );
}

TEST_F(ExpressionTest, ExpressionExponent) {
    std::shared_ptr<Expression> three =
        std::make_shared<TermExpression>(kInteger, 3);
    EXPECT_EQ (*three, *(x0cubed->GetExponent()) );
    EXPECT_EQ (*one, *(c0->GetExponent()) );
    EXPECT_EQ (*one, *(c0x0->GetExponent()) );
}

} // namespace