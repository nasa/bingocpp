# BingoCpp
master: [![Build Status](https://travis-ci.com/nasa/bingocpp.svg?branch=master)](https://travis-ci.com/nasa/bingocpp) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingocpp/badge.svg?branch=master)](https://coveralls.io/github/nasa/bingocpp?branch=master)

develop: [![Build Status](https://travis-ci.com/nasa/bingocpp.svg?branch=develop)](https://travis-ci.com/nasa/bingocpp) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingocpp/badge.svg?branch=develop)](https://coveralls.io/github/nasa/bingocpp?branch=develop) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/ccd11c4092544eaca355722cea87272e)](https://www.codacy.com/app/bingo_developers/bingocpp?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nasa/bingocpp&amp;utm_campaign=Badge_Grade)

## General
Bingocpp is part of the open source package Bingo for performing symbolic 
regression.  Bingocpp contains the c++ implementation of a portion of the code 
within bingo.


## Installation
The build process requires:
- A recent version of cmake
- git

Installation can be performed using the typical out of source build flow:
```bash
mkdir <path to source dir>/build
cd <path to source dir>/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Python Bindings
The python bindings that are needed for integration with bingo can be made by 
running `make pybind` in the build directory.

### Tests
Several unit and integration tests can be made upon building.  You can run the 
testing suite by running `make gtest` in the build directory.


## Notices
Copyright 2018 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. No copyright is claimed in 
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 

## Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF 
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE 
OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED 
STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR 
RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY 
SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

