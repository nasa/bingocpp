/*!
 * \file driver.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the main function for BingoCpp.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>

#include <iostream>

#include "BingoCpp/version.h"
#include "BingoCpp/acyclic_graph.h"


int test_eig() {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
}


int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stdout, "%s Version %d.%d\n", argv[0], Tutorial_VERSION_MAJOR,
            Tutorial_VERSION_MINOR);
    fprintf(stdout, "Usage: %s number\n", argv[0]);
    return 1;
  }

  test_eig();
  return 0;
}



