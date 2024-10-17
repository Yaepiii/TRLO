/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/
#ifndef MY_PCL_TUTORIAL_IMM_UKF_JPDAF_H
#define MY_PCL_TUTORIAL_IMM_UKF_JPDAF_H

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace pcl;

void getOriginPoints(Eigen::Matrix3f pose);

Eigen::VectorXd getCpFromBbox(PointCloud<PointXYZ> bBox);

void immUkfJpdaf(vector<PointCloud<PointXYZ>> bBoxes, double timestamp, 
	PointCloud<PointXYZ>& targets, vector<vector<double>>& targetVandYaw, 
	vector<int>& trackManage, vector<bool>& isStaticVec,
	vector<bool>& isVisVec, vector<PointCloud<PointXYZ>>& visBB);




#endif /* MY_PCL_TUTORIAL_IMM_UKF_JPDAF_H */
