#include <gtest/gtest.h>

#include <bingocpp/agraph/simplification_backend/expression.h>
#include <bingocpp/agraph/operator_definitions.h>


using namespace bingo;
using namespace simplification_backend;
namespace {

class ExpressionTest : public testing::Test {
 public:
  TermExpression zero = TermExpression(kInteger, 0);
  TermExpression one = TermExpression(kInteger, 1);
  TermExpression x0 = TermExpression(kVariable, 0);
  TermExpression c0 = TermExpression(kConstant, 0);
};

TEST_F(ExpressionTest, ExpressionOperator) {
    EXPECT_EQ (kInteger, zero.GetOperator());
    EXPECT_EQ (kInteger, one.GetOperator());
    EXPECT_EQ (kVariable, x0.GetOperator());
    EXPECT_EQ (kConstant, c0.GetOperator());
}

TEST_F(ExpressionTest, ExpressionIsOne) {
    EXPECT_TRUE (one.IsOne());
    EXPECT_TRUE (!zero.IsOne());
    EXPECT_TRUE (!x0.IsOne());
    EXPECT_TRUE (!c0.IsOne());
}

TEST_F(ExpressionTest, ExpressionIsZero) {
    EXPECT_TRUE (zero.IsZero());
    EXPECT_TRUE (!one.IsZero());
    EXPECT_TRUE (!x0.IsZero());
    EXPECT_TRUE (!c0.IsZero());
}

TEST_F(ExpressionTest, ExpressionIsConstantValued) {
    EXPECT_TRUE (zero.IsConstantValued());
    EXPECT_TRUE (one.IsConstantValued());
    EXPECT_TRUE (!x0.IsConstantValued());
    EXPECT_TRUE (c0.IsConstantValued());
}

TEST_F(ExpressionTest, ExpressionEquality) {
    TermExpression zero_duplicate = TermExpression(kInteger, 0);
    EXPECT_EQ (zero, zero_duplicate);
    EXPECT_NE (zero, one);
    EXPECT_NE (zero, x0);
    EXPECT_NE (zero, c0);
}

} // namespace