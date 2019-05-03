#include <iostream>

#include "BingoCpp/agraph_crossover.h"

namespace bingo {
namespace {

// TODO: If we use mask to simplify stack, we can move the uniform
// distro to be a member variable instead of creating a new instance each
// call. Basically if CommandArray size is constant throughout the 
// exeuction, we can optimize this.
int find_random_int(std::mt19937& engine, int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(engine);
}

void perform_crossover(AGraph& child, AGraph& parent, int cross_point) {
  Eigen::ArrayX3i child_array = child.getCommandArray();
  Eigen::ArrayX3i parent_array = parent.getCommandArray();
  int num_modified_rows = parent_array.rows() - cross_point;

  // Hard coded substitute 2 for row length
  child_array.block(cross_point, 0, num_modified_rows, child_array.cols()) =
    parent_array.block(cross_point, 0, num_modified_rows, parent_array.cols());
  
  child.setCommandArray(child_array);
}
} // namespace

AGraphCrossover::AGraphCrossover() {
  std::random_device rd;
  engine_ = std::mt19937(rd()); 
}

AGraphCrossover::AGraphCrossover(std::mt19937::result_type seed) {
  engine_ = std::mt19937(seed);
}

AGraphCrossover::AGraphCrossover(const AGraphCrossover& agc) {
  engine_ = agc.engine_;
}

CrossoverChildren AGraphCrossover::crossover(AGraph& parent_1,
                                             AGraph& parent_2) {
  AGraph child_1 = parent_1.copy();
  AGraph child_2 = parent_2.copy();

  size_t agraph_size = parent_1.getCommandArray().rows();
  int cross_point = find_random_int(engine_, 1, agraph_size - 1);
  perform_crossover(child_1, parent_2, cross_point);
  perform_crossover(child_2, parent_1, cross_point);

  int child_age = std::max(parent_1.getGeneticAge(), parent_2.getGeneticAge());
  child_1.setGeneticAge(child_age);
  child_2.setGeneticAge(child_age);

  return std::make_pair(child_1, child_2);
}
} // namespace bingo