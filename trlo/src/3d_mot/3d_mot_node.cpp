/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/
#include <ros/ros.h>

#include <vector>
#include <iostream>
#include <math.h>

// PCL specific includes
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include "pcl_ros/impl/transforms.hpp"

#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include "tf/transform_datatypes.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <visualization_msgs/Marker.h>

#include "3d_mot/imm_ukf_jpda.h"

ros::Subscriber sub;
ros::Subscriber sub_odom;
ros::Publisher pub_track_box;
ros::Publisher vis_pub;
ros::Publisher vis_pub2;

tf::TransformListener* tran;

std::deque<nav_msgs::Odometry> odom_queue;

pcl::PointCloud<pcl::PointXYZ>::Ptr color_point(new pcl::PointCloud<pcl::PointXYZ>());

int counta = 0;
ros::Time last_time;

// 回调函数-- 话题 track_box
void  track_box (const jsk_recognition_msgs::BoundingBoxArrayPtr &input){


  counta ++;
  cout << "Frame: "<<counta << "----------------------------------------"<< endl;

  // convert local to global-------------------------
  double timestamp = input->header.stamp.toSec();
  ros::Time input_time = input->header.stamp;

  int box_num = input->boxes.size();

  vector<PointCloud<PointXYZ>> bBoxes;
  PointCloud<PointXYZ> oneBbox;
//  bBoxes.header = input.header;
  for(int i = 0; i < box_num; i++)
  {
      PointXYZ o;
      o.x = input->boxes[i].pose.position.x - input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y - input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z - input->boxes[i].dimensions.z / 2;
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x + input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y - input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z - input->boxes[i].dimensions.z / 2; 
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x - input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y + input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z - input->boxes[i].dimensions.z / 2;   
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x + input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y + input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z - input->boxes[i].dimensions.z / 2;  
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x - input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y - input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z + input->boxes[i].dimensions.z / 2;  
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x + input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y - input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z + input->boxes[i].dimensions.z / 2;  
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x - input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y + input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z + input->boxes[i].dimensions.z / 2;  
      oneBbox.push_back(o);
      o.x = input->boxes[i].pose.position.x + input->boxes[i].dimensions.y / 2;
      o.y = input->boxes[i].pose.position.y + input->boxes[i].dimensions.x / 2;
      o.z = input->boxes[i].pose.position.z + input->boxes[i].dimensions.z / 2;  
      oneBbox.push_back(o);
      bBoxes.push_back(oneBbox);
      oneBbox.clear();
  }
  
  // std::cout << "!!!!!input->boxes[i].pose.position.x: " << input->boxes[0].pose.position.x << std::endl;

  tf::StampedTransform transform;

  // static bool first = 1;
  // if (first)
  //   last_time = input_time;
  //   first = 0;
  //   return;

  // while (odom_queue.front().header.stamp != last_time) {
  //   odom_queue.pop_front();
  // }
  if (odom_queue.empty())
    return;
  // std::cout << odom_queue.front().pose.pose.position.x << "!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

  // transform.setOrigin( tf::Vector3(odom_queue.front().pose.pose.position.x, odom_queue.front().pose.pose.position.y, 0.0) );
  // tf::Quaternion q(odom_queue.front().pose.pose.orientation.x, odom_queue.front().pose.pose.orientation.y, odom_queue.front().pose.pose.orientation.z, odom_queue.front().pose.pose.orientation.w);
  
  // transform.setRotation( q );
  // tran->setTransform(transform);

  
  PointCloud<PointXYZ> newBox;
  for(int i = 0; i < box_num; i++ ){
    bBoxes[i].header.frame_id = "robot/odom";

    // tran->waitForTransform("robot/odom", "robot/base_link", last_time, ros::Duration(10.0));
    // pcl_ros::transformPointCloud("robot/odom", bBoxes[i], newBox, *tran);
    Eigen::Matrix4f last_T;
    Eigen::Quaternionf last_q(odom_queue.front().pose.pose.orientation.w, odom_queue.front().pose.pose.orientation.x, odom_queue.front().pose.pose.orientation.y, odom_queue.front().pose.pose.orientation.z);
    last_T.block<3,3>(0,0) = last_q.toRotationMatrix();
    last_T.block<3,1>(0,3) = Eigen::Vector3f(odom_queue.front().pose.pose.position.x, odom_queue.front().pose.pose.position.y, 0.0);
    pcl::transformPointCloud(bBoxes[i], newBox, last_T);
    bBoxes[i] = newBox;
    bBoxes[i].header.frame_id = "robot/odom";
    // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
  }

  for (int i = 0; i < bBoxes.size(); i++) {
    *color_point += bBoxes[i];
  }

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*color_point, cloud_msg);
  cloud_msg.header = input->header;
  cloud_msg.header.frame_id = "robot/odom";
  pub_track_box.publish(cloud_msg);

  //end converting----------------------------------------
  PointCloud<PointXYZ> targetPoints;
  vector<vector<double>> targetVandYaw;
  vector<int> trackManage;  // trackManage???  大量具有相关不确定性的跟踪对象需要有效地实施跟踪管理。跟踪管理的主要目的是动态限制虚假跟踪列表的数量（从而防止错误的数据关联），并在丢失检测的情况下保持对象跟踪
  vector<bool> isStaticVec;
  vector<bool> isVisVec;
  vector<PointCloud<PointXYZ>> visBBs;
  // std::cout << "!!!!!input->boxes[i].pose.position.x: " << bBoxes[0][0] << std::endl;
  immUkfJpdaf(bBoxes, timestamp, targetPoints, targetVandYaw, trackManage, isStaticVec, isVisVec, visBBs);

// //   cout << "size is "<<visBBs.size()<<endl;
//   // cout << "x1:"<<visBBs[0][0].x<<"y1:"<<visBBs[0][0].y<<endl;
//   // cout << "x2:"<<visBBs[0][1].x<<"y2:"<<visBBs[0][1].y<<endl;
//   // cout << "x3:"<<visBBs[0][2].x<<"y3:"<<visBBs[0][2].y<<endl;
//   // cout << "x4:"<<visBBs[0][3].x<<"y4:"<<visBBs[0][3].y<<endl;


  assert(targetPoints.size() == trackManage.size());
  assert(targetPoints.size()== targetVandYaw.size());

  Eigen::Matrix4f last_T;
  Eigen::Quaternionf last_q(odom_queue.front().pose.pose.orientation.w, odom_queue.front().pose.pose.orientation.x, odom_queue.front().pose.pose.orientation.y, odom_queue.front().pose.pose.orientation.z);
  last_T.block<3,3>(0,0) = last_q.toRotationMatrix().inverse();
  Eigen::Vector3f t(odom_queue.front().pose.pose.position.x, odom_queue.front().pose.pose.position.y, 0.0);
  t = -last_T.block<3,3>(0,0) * t;
  last_T.block<3,1>(0,3) = t;

  // converting from global to ego tf for visualization
  // processing targetPoints
  PointCloud<PointXYZ> egoTFPoints;
  targetPoints.header.frame_id = "robot/base_link";
  // pcl_ros::transformPointCloud("/velodyne", targetPoints, egoTFPoints, *tran);
  pcl::transformPointCloud(targetPoints, egoTFPoints, last_T);

  //processing visBBs
  PointCloud<PointXYZ> visEgoBB;
  for(int i = 0; i < visBBs.size(); i++){
    visBBs[i].header.frame_id = "robot/base_link";
    // pcl_ros::transformPointCloud("/velodyne", visBBs[i], visEgoBB, *tran);
    pcl::transformPointCloud(visBBs[i], visEgoBB, last_T);

    visBBs[i] = visEgoBB;
  }
  //end converting to ego tf-------------------------





  // tracking arrows visualizing start---------------------------------------------
  for(int i = 0; i < targetPoints.size(); i++){
    visualization_msgs::Marker arrowsG;
    arrowsG.lifetime = ros::Duration(0.1);
    if(trackManage[i] == 0 ) {
      continue;
    }
    if(isVisVec[i] == false ) {
      continue;
    }
    if(isStaticVec[i] == true){
      continue;
    }
//    arrowsG.header.frame_id = "/velo_link";
    arrowsG.header.frame_id = "robot/base_link";
    
    arrowsG.header.stamp= input_time;
    arrowsG.ns = "arrows";
    arrowsG.action = visualization_msgs::Marker::ADD;
    arrowsG.type =  visualization_msgs::Marker::ARROW;
    // green  设置颜色
    arrowsG.color.g = 1.0f; // 绿色
    // arrowsG.color.r = 1.0f; // 红色
    arrowsG.color.a = 1.0;  
    arrowsG.id = i;
    geometry_msgs::Point p;
    // assert(targetPoints[i].size()==4);
    p.x = egoTFPoints[i].x;
    p.y = egoTFPoints[i].y;
    p.z = -1.73/2;
    double tv   = targetVandYaw[i][0];
    double tyaw = targetVandYaw[i][1];

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    arrowsG.pose.position.x = p.x;
    arrowsG.pose.position.y = p.y;
    arrowsG.pose.position.z = p.z;

    // convert from 3 angles to quartenion
    tf::Matrix3x3 obs_mat;
    obs_mat.setEulerYPR(tyaw, 0, 0); // yaw, pitch, roll
    tf::Quaternion q_tf;
    obs_mat.getRotation(q_tf);
    arrowsG.pose.orientation.x = q_tf.getX();
    arrowsG.pose.orientation.y = q_tf.getY();
    arrowsG.pose.orientation.z = q_tf.getZ();
    arrowsG.pose.orientation.w = q_tf.getW();

    // Set the scale of the arrowsG -- 1x1x1 here means 1m on a side
    arrowsG.scale.x = tv;
    arrowsG.scale.y = 0.1;
    arrowsG.scale.z = 0.1;

    vis_pub2.publish(arrowsG);  // 发布箭头消息
  }














  // tracking points visualizing start---------------------------------------------
  
  visualization_msgs::Marker pointsY, pointsG, pointsR, pointsB;
  pointsY.header.frame_id = pointsG.header.frame_id = pointsR.header.frame_id = pointsB.header.frame_id = "robot/base_link";
//  pointsY.header.frame_id = pointsG.header.frame_id = pointsR.header.frame_id = pointsB.header.frame_id = "velo_link";
  
  pointsY.header.stamp= pointsG.header.stamp= pointsR.header.stamp =pointsB.header.stamp = input_time;
  pointsY.ns= pointsG.ns = pointsR.ns =pointsB.ns=  "points";
  pointsY.action = pointsG.action = pointsR.action = pointsB.action = visualization_msgs::Marker::ADD;
  pointsY.pose.orientation.w = pointsG.pose.orientation.w  = pointsR.pose.orientation.w =pointsB.pose.orientation.w= 1.0;

  pointsY.id = 1;
  pointsG.id = 2;
  pointsR.id = 3;
  pointsB.id = 4;
  pointsY.type = pointsG.type = pointsR.type = pointsB.type = visualization_msgs::Marker::POINTS;

  // POINTS markers use x and y scale for width/height respectively
  pointsY.scale.x =pointsG.scale.x =pointsR.scale.x = pointsB.scale.x=0.5;
  pointsY.scale.y =pointsG.scale.y =pointsR.scale.y = pointsB.scale.y = 0.5;

  // yellow（红绿蓝混合为黄）
  pointsY.color.r = 1.0f;
  pointsY.color.g = 1.0f;
  pointsY.color.b = 0.0f;
  pointsY.color.a = 1.0;

  // green
  pointsG.color.g = 1.0f;
  pointsG.color.a = 1.0;

  // red
  pointsR.color.r = 1.0;
  pointsR.color.a = 1.0;

  // blue 
  pointsB.color.b = 1.0;
  pointsB.color.a = 1.0;

//  cout << "targetPoints.size() is --=------" << targetPoints.size() <<endl;

  for(int i = 0; i < targetPoints.size(); i++){
    if(trackManage[i] == 0) continue;
    geometry_msgs::Point p;
    // p.x = targetPoints[i].x;
    // p.y = targetPoints[i].y;
    p.x = egoTFPoints[i].x;
    p.y = egoTFPoints[i].y;
    p.z = -1.73/2;

//   cout << "is ------------------" << i <<endl;
    // cout << "trackManage[i]  " <<trackManage[i] << endl; // 输出
    if(isStaticVec[i] == true){   // isStaticVec???
      pointsB.points.push_back(p);    // 蓝点
    }
    else if(trackManage[i] < 5 ){  // 小于5为黄点
      pointsY.points.push_back(p);
    }
    else if(trackManage[i] == 5){  // 等于5为绿点
      pointsG.points.push_back(p);
    }
    else if(trackManage[i] > 5){
      pointsR.points.push_back(p);    // 大于5为红点
    }
  }
  vis_pub.publish(pointsY);   // 发布
  // cout << "pointsG" << pointsG.points[0].x << " "<< pointsG.points[0].y << endl;
  vis_pub.publish(pointsG);  // 发布
  vis_pub.publish(pointsR);  // 发布
  vis_pub.publish(pointsB);  // 发布
  // tracking points visualizing end---------------------------------------------



  odom_queue.pop_front();



}

void odom_callback(const nav_msgs::OdometryPtr &odom) {
  odom_queue.push_back(*odom);
  // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! size: " << odom_queue.size() << std::endl;
}


int main(int argc, char **argv) {
    ros::init (argc, argv, "3d_mot_node");

    ros::NodeHandle nh("~");

    // TF
    tf::TransformListener lr(ros::Duration(100));         //(How long to store transform information)
    tran=&lr;

    sub = nh.subscribe ("mot_box", 160, track_box);   //订阅者  track_box -- 话题topic名
    sub_odom = nh.subscribe ("odom", 160, odom_callback);   //订阅者  track_box -- 话题topic名
    // pub_track_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("track_box", 10, true);
    pub_track_box = nh.advertise<sensor_msgs::PointCloud2>("track_box", 10);

    vis_pub = nh.advertise<visualization_msgs::Marker>( "tracking_center", 0 );    //发布者  visualization_marker -- 话题topic名
    vis_pub2 = nh.advertise<visualization_msgs::Marker>( "object_vel_arrows", 0 );  //发布者  visualization_marker2 -- 话题topic名


    ros::spin();
}

























