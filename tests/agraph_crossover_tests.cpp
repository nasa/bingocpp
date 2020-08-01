#include <gtest/gtest.h>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/agraph/agraph_crossover.h>

#include "test_fixtures.h"
#include "testing_utils.h"

using namespace bingo;

namespace {

class AGraphCrossoverTest : public ::testing::Test {
 public:
  AGraph sample_agraph_1;
  AGraph sample_agraph_2;

  void SetUp() {
    sample_agraph_1 = testutils::init_sample_agraph_1();
    sample_agraph_2 = testutils::init_sample_agraph_2();
  }

  void TearDown() {}
};

TEST_F(AGraphCrossoverTest, test_single_point_crossover) {
  AGraphCrossover crossover;
  CrossoverChildren children = crossover.Crossover(sample_agraph_1, sample_agraph_2);

  bool found_crossover_point = false;
  for (int i = 0; i < sample_agraph_1.GetCommandArray().rows(); i ++) {
    Eigen::ArrayX3i parent_1_op = sample_agraph_1.GetCommandArray().row(i);
    Eigen::ArrayX3i parent_2_op = sample_agraph_2.GetCommandArray().row(i);
    Eigen::ArrayX3i child_1_op = children.first.GetCommandArray().row(i); 
    Eigen::ArrayX3i child_2_op = children.second.GetCommandArray().row(i); 
    if (!found_crossover_point) {
      if (parent_1_op.isApprox(child_1_op)) {
        ASSERT_TRUE(parent_2_op.isApprox(child_2_op));
      } else if (parent_2_op.isApprox(child_1_op)) {
        found_crossover_point = true;
      } else {
        FAIL() << "Genes do not match parents!\n";
      }
    }

    if (found_crossover_point) {
      ASSERT_TRUE(parent_2_op.isApprox(child_1_op));
      ASSERT_TRUE(parent_1_op.isApprox(child_2_op));
    }
  }
}

TEST_F(AGraphCrossoverTest, modified_genetic_age) {
  AGraphCrossover crossover;
  CrossoverChildren children = crossover.Crossover(sample_agraph_1,
                                                   sample_agraph_2);

  int oldest_parent_age = std::max(sample_agraph_1.GetGeneticAge(),
                                   sample_agraph_2.GetGeneticAge());
  ASSERT_TRUE(children.first.GetGeneticAge() == oldest_parent_age);
  ASSERT_TRUE(children.second.GetGeneticAge() == oldest_parent_age);
}

TEST_F(AGraphCrossoverTest, crossover_resets_fitness) {
  ASSERT_TRUE(sample_agraph_1.IsFitnessSet());
  ASSERT_TRUE(sample_agraph_2.IsFitnessSet());

  AGraphCrossover crossover;
  CrossoverChildren children = crossover.Crossover(sample_agraph_1,
                                                   sample_agraph_2);
  ASSERT_FALSE(children.first.IsFitnessSet());
  ASSERT_FALSE(children.second.IsFitnessSet());
  ASSERT_DOUBLE_EQ(children.first.GetFitness(), 1e9);
  ASSERT_DOUBLE_EQ(children.second.GetFitness(), 1e9);
}
}