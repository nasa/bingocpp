/*!
 * \file training_data.h
 *
 * \author Ethan Adams
 * \date 
 *
 * This file contains the cpp version of training_data.py
 */

#ifndef INCLUDE_BINGOCPP_TRAINING_DATA_H_
#define INCLUDE_BINGOCPP_TRAINING_DATA_H_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <list>

struct TrainingData {
    public:
    /*! \brief 
    *
    *  \param[in] 
    *  \return 
    */
    // TrainingData() { }
    virtual TrainingData* get_item(std::list<int> items) = 0;
    virtual int size() = 0;
};

struct ExplicitTrainingData : TrainingData {
    public:
    Eigen::ArrayXXd x;
    Eigen::ArrayXXd y;
    // ExplicitTrainingData() : TrainingData() {}
    ExplicitTrainingData(Eigen::ArrayXXd vx, Eigen::ArrayXXd vy);
    ExplicitTrainingData* get_item(std::list<int> items);
    int size() { return x.rows(); }
};

struct ImplicitTrainingData : TrainingData {
    public:
    Eigen::ArrayXXd x;
    Eigen::ArrayXXd dx_dt;
    // ImplicitTrainingData : TrainingData() {}
    ImplicitTrainingData(Eigen::ArrayXXd vx, Eigen::ArrayXXd vdx_dt);
    ImplicitTrainingData* get_item(std::list<int> items);
    int size() { return x.rows(); }
};


#endif