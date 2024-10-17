/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/

#include "trlo/odom.h"

void controlC(int sig) {

  trlo::OdomNode::abort();

}

int main(int argc, char** argv) {

  ros::init(argc, argv, "trlo_odom_node");
  ros::NodeHandle nh("~");

  signal(SIGTERM, controlC);
  sleep(0.5);

  trlo::OdomNode node(nh);
  ros::AsyncSpinner spinner(0);
  spinner.start();
  node.start();
  ros::waitForShutdown();

  return 0;

}
