/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/

#include "trlo/map.h"

void controlC(int sig) {

  trlo::MapNode::abort();

}

int main(int argc, char** argv) {

  ros::init(argc, argv, "trlo_map_node");
  ros::NodeHandle nh("~");

  signal(SIGTERM, controlC);
  sleep(0.5);

  trlo::MapNode node(nh);
  ros::AsyncSpinner spinner(1);
  spinner.start();
  node.start();
  ros::waitForShutdown();

  return 0;

}
