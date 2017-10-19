#include <iostream>
#include <stdio.h>
#include <math.h>
#include "gtest/gtest.h"
#include "BingoCpp/AGraph.hh"

namespace {

TEST(ExampleTest, ExampleSqrt2) {
  ASSERT_EQ(do_sqrt(3.0), sqrt(3.0));
}

TEST(ExampleTest2, ExampleSqrt3) {
  ASSERT_EQ(do_sqrt(3.0), sqrt(3.0));
}

} //namespace
