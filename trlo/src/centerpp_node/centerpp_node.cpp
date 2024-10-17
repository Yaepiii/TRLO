/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/

// PCL specific includes
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include "pcl_ros/impl/transforms.hpp"

#include <ros/ros.h>
#include "center_pointpillars/centerpoint.h"

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <std_msgs/Header.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nano_gicp/nanoflann.hpp>
#include <nav_msgs/Odometry.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <nav_msgs/Odometry.h>

#include "3d_mot/imm_ukf_jpda.h"

#include <algorithm>

#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

std::vector<unsigned char> color;

std::vector<double> avg_centerpoint_time;
std::vector<double> avg_ukf_time;


void GetDeviceInfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

void initDevice(int devNum) {
    int dev = devNum;
    cudaDeviceProp deviceProp;

    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    GPU_CHECK(cudaSetDevice(dev));
}

void SaveToBin(const pcl::PointCloud<pcl::PointXYZI>& cloud, const std::string& path) {
    // std::cout << "bin_path: " << path << std::endl;
    //Create & write .bin file
    std::ofstream out;
    out.open(path, std::ios::out|std::ios::binary);
    if(!out.good()) {
        std::cout<<"Couldn't open "<<path<<std::endl;
        return;
    }
    float zero = 0.0f;
    
    for (size_t i = 0; i < cloud.points.size (); ++i) {
        out.write((char*)&cloud.points[i].x, 3*sizeof(float));
        out.write((char*)&zero, sizeof(float));
        out.write((char*)&zero, sizeof(float));
    }
    
    out.close();
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}

namespace cpp {

class Center_PointPillars_ROS {
  public:
    pcl::PointCloud<pcl::PointXYZ> targetPoints;
    std::vector<std::vector<double>> targetVandYaw;
    std::vector<int> trackManage;
    std::vector<bool> isStaticVec;
    std::vector<bool> isVisVec;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> visBBs;

    Center_PointPillars_ROS(ros::NodeHandle nh);
    ~Center_PointPillars_ROS();

    void Process();
    void extractBBoxPointcloud(std::vector<Bndbox> filter_BBox, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster);
    void mot_3d(std_msgs::Header in_msg_header, std::vector<Bndbox> filter_BBox, std::vector<Bndbox>& dynamic_BBox);

  private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_pointcloud_;
    ros::Subscriber sub_odom_;
    ros::Publisher pub_pointcloud_static_;
    ros::Publisher pub_pointcloud_raw_;
    ros::Publisher pub_bbox_;
    ros::Publisher pub_dynamic_bbox_;
    ros::Publisher pub_pointcloud_cluster_;
    ros::Publisher pub_text_vel_;
    ros::Publisher pub_tracking_center_;
    ros::Publisher pub_arrows_;
    ros::Publisher pub_box_markers_;
    ros::Publisher pub_center_points_;
    
    cudaEvent_t start_, stop_;
    cudaStream_t stream_ = NULL;

    Params params;

    std::string Model_File_Dir_;
    std::string saveBinFile_;

    std::string odom_frame_;
    std::string child_frame_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr original_scan_;

    double MINIMUM_RANGE;
    double MAXMUM_RANGE;
    bool crop_use_;
    double crop_size_;

    bool vf_use_;
    double vf_res_;

    pcl::CropBox<pcl::PointXYZI> crop;
    pcl::VoxelGrid<pcl::PointXYZI> vf;
    std::deque<nav_msgs::Odometry> odom_queue;
    visualization_msgs::MarkerArray center_points_array;

    bool verbose = true;

    std::unique_ptr<CenterPoint> center_pointpillars_ptr_;
    nanoflann::KdTreeFLANN<pcl::PointXYZ>::Ptr objects_kdtree_;

    void PointCloud_Callback(const sensor_msgs::PointCloud2Ptr &msg);
    void Odometry_Callback(const nav_msgs::OdometryPtr &odom);
    void publishCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr);
    void publishObjectBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> filter_BBox);
    void publishDynamicBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> dynamic_BBox);
    void publishClusterCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<pcl::PointIndices> cluster_indices);
    void publishVelArrows(pcl::PointCloud<pcl::PointXYZ> egoTFPoints, ros::Time input_time);
    void publishTrackingCenter(pcl::PointCloud<pcl::PointXYZ> egoTFPoints, ros::Time input_time);
    void publishBoundingBoxMarkers(ros::Time input_time);

    void preprocessPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in, float th1, float th2);
    void removeClosedPointCloud(const pcl::PointCloud<pcl::PointXYZI> &cloud_in, pcl::PointCloud<pcl::PointXYZI> &cloud_out, float th1, float th2);
};

Center_PointPillars_ROS::Center_PointPillars_ROS(ros::NodeHandle nh) : nh_(nh) {
    ros::param::param<std::string>("Model_File_Dir", this->Model_File_Dir_, "/home/jyp/3D_LiDAR_SLAM/trlo_ws/src/trlo/model/center_pointpillars/");
    ros::param::param<std::string>("SavaData_File", this->saveBinFile_, "/home/jyp/3D_LiDAR_SLAM/trlo_ws/src/trlo/data/data.bin");
    ros::param::param<std::string>("~center_pp/frame/odom_frame", this->odom_frame_, "robot/odom");
    ros::param::param<std::string>("~center_pp/frame/child_frame", this->child_frame_, "robot/base_link");

    ros::param::param<double>("~center_pp/preprocessing/threshold/MINIMUM_RANGE", this->MINIMUM_RANGE, 0.5);
    ros::param::param<double>("~center_pp/preprocessing/threshold/MAXMUM_RANGE", this->MAXMUM_RANGE, 80);

    // Crop Box Filter
    ros::param::param<bool>("~center_pp/preprocessing/cropBoxFilter/use", this->crop_use_, false);
    ros::param::param<double>("~center_pp/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

    // Voxel Grid Filter
    ros::param::param<bool>("~center_pp/preprocessing/voxelFilter/use", this->vf_use_, true);
    ros::param::param<double>("~center_pp/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

    checkCudaErrors(cudaEventCreate(&this->start_));
    checkCudaErrors(cudaEventCreate(&this->stop_));
    GPU_CHECK(cudaStreamCreate(&this->stream_));
    this->center_pointpillars_ptr_.reset(new CenterPoint(this->Model_File_Dir_, this->verbose)); // 外部定义调用不了cuda函数
    this->objects_kdtree_.reset(new nanoflann::KdTreeFLANN<pcl::PointXYZ>());
    this->original_scan_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    this->crop.setNegative(true);
    this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
    this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

    this->vf.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);
    center_pointpillars_ptr_->prepare();
    setlocale(LC_ALL,"");
}


Center_PointPillars_ROS::~Center_PointPillars_ROS(){
        checkCudaErrors(cudaEventDestroy(this->start_));
        checkCudaErrors(cudaEventDestroy(this->stop_));
        checkCudaErrors(cudaStreamDestroy(this->stream_));
}


void Center_PointPillars_ROS::Process() {
    std::cout << "Ready to receive point cloud topic!" << std::endl;
    this->sub_pointcloud_ = nh_.subscribe("pointcloud", 1, &Center_PointPillars_ROS::PointCloud_Callback, this);
    this->sub_odom_ = nh_.subscribe ("odom", 160, &Center_PointPillars_ROS::Odometry_Callback, this);
    this->pub_pointcloud_static_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_static", 10);
    this->pub_pointcloud_raw_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_raw", 10);
    this->pub_bbox_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("box", 10, true);
    this->pub_dynamic_bbox_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("dynamic_box", 10, true);
    this->pub_pointcloud_cluster_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_cluster", 10);
    this->pub_text_vel_ = nh_.advertise<visualization_msgs::MarkerArray>("centerpoint_vel", 10);
    this->pub_tracking_center_ = nh_.advertise<visualization_msgs::Marker>("tracking_center", 10);
    this->pub_arrows_ = nh_.advertise<visualization_msgs::Marker>("object_vel_arrows", 10);
    this->pub_box_markers_ = nh_.advertise<visualization_msgs::Marker> ("box_markers", 10);
    this->pub_center_points_ = nh_.advertise<visualization_msgs::MarkerArray> ("center_markers", 10);
    ros::spin();
}



void Center_PointPillars_ROS::PointCloud_Callback(const sensor_msgs::PointCloud2Ptr &msg) {

    pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*msg, *in_cloud_ptr);

    // std::cout << "process before size is : " << in_cloud_ptr->size() << std::endl;
    // this->preprocessPoints(in_cloud_ptr, this->MINIMUM_RANGE, this->MAXMUM_RANGE);
    // std::cout << "process after size is : " << in_cloud_ptr->size() << std::endl;

    SaveToBin(*in_cloud_ptr, this->saveBinFile_);
    // std::cout << "load file: "<< this->saveBinFile_ <<std::endl;
    unsigned int length = 0;
    void *pc_data = NULL;
    loadData(this->saveBinFile_.c_str() , &pc_data, &length);
    size_t points_num = length / (5 * sizeof(float)) ;

    float *d_points = nullptr;
    checkCudaErrors(cudaMallocManaged((void **)&d_points, MAX_POINTS_NUM * 5 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_points, pc_data, length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(this->start_, this->stream_);

    double t1 = ros::Time::now().toSec();
    center_pointpillars_ptr_->doinfer((void *)d_points, points_num, this->stream_);
    double t2 = ros::Time::now().toSec();
    avg_centerpoint_time.push_back((t2 - t1) * 1000);

    float elapsedTime = 0.0f;

    cudaEventRecord(this->stop_, this->stream_);
    cudaEventSynchronize(this->stop_);
    // cudaEventElapsedTime(&elapsedTime, this->start_, this->stop_);

    checkCudaErrors(cudaFree(d_points));


    std::vector<Bndbox> filter_BBox, dynamic_BBox;
    for (auto box : this->center_pointpillars_ptr_->nms_pred_) {
        // car(id=0)/pedestrain(id=8)/cyclists(id=6)/truck(id=3)
        // if ((box.id == 0 && box.score > 0.3) || (box.id == 6 && box.score > 0.5) || (box.id == 8 && box.score > 0.2)) { 
        //     filter_BBox.push_back(box);
        // }
        // if (((box.id == 0 && box.score > 0.5) || (box.id == 6 && box.score > 0.75)) && abs(box.vy) > 0.001) {
        //     dynamic_BBox.push_back(box);
        // }
        if ((box.id == 0 && box.score > 0.5) || (box.id == 6 && box.score > 0.75)) {
            filter_BBox.push_back(box);
        }
    }

    this->mot_3d(msg->header, filter_BBox, dynamic_BBox);
    
    // std::cout << "filter box size: " << filter_BBox.size() << std::endl;
    // std::cout << "dynamic box size: " << dynamic_BBox.size() << std::endl;

    if (filter_BBox.empty()) {
        this->preprocessPoints(in_cloud_ptr, this->MINIMUM_RANGE, this->MAXMUM_RANGE);
        this->publishCloud(msg->header, in_cloud_ptr);
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>());
    this->extractBBoxPointcloud(dynamic_BBox, in_cloud_ptr, out_cloud_ptr, cloud_cluster);

    double avg_centerpoint_totaltime = std::accumulate(avg_centerpoint_time.begin(), avg_centerpoint_time.end(), 0.0) / avg_centerpoint_time.size();
    double avg_ukf_totaltime = std::accumulate(avg_ukf_time.begin(), avg_ukf_time.end(), 0.0) / avg_ukf_time.size();
    std::cout << "CenterPoint Time :: " << std::setfill(' ') << std::setw(6) << avg_centerpoint_time.back() << " ms    // Avg: " << std::setw(5) << avg_centerpoint_totaltime << std::endl;
    std::cout << "UKF Time :: " << std::setfill(' ') << std::setw(6) << avg_ukf_time.back() << " ms    // Avg: " << std::setw(5) << avg_ukf_totaltime << std::endl;


    
    this->preprocessPoints(out_cloud_ptr, this->MINIMUM_RANGE, this->MAXMUM_RANGE);

    this->publishCloud(msg->header, out_cloud_ptr);
    this->publishObjectBoundingBox(msg->header, filter_BBox);
    this->publishDynamicBoundingBox(msg->header, dynamic_BBox);
    // this->publishClusterCloud(msg->header, cloud_cluster, cluster_indices);

}


void Center_PointPillars_ROS::Odometry_Callback(const nav_msgs::OdometryPtr &odom) {
    this->odom_queue.push_back(*odom);
}



void Center_PointPillars_ROS::extractBBoxPointcloud(std::vector<Bndbox> filter_BBox, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster) {
    // KDtree：O((logm)^2) + O(m^(1-1/3)+1) * n
    // Violent methods：O(m*n)
    if (filter_BBox.empty()) {
        *cloud_out = *cloud_in;
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud;
    objects_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto box : filter_BBox) {
        pcl::PointXYZ p;
        p.x = box.x;
        p.y = box.y;
        p.z = box.z;
        objects_cloud->points.push_back(p);
    }
    this->objects_kdtree_->setInputCloud(objects_cloud);
    
    int cloudSize = cloud_in->size();
    for (size_t i = 0; i < cloudSize; i++)
    {
        pcl::PointXYZ p;
        p.x = cloud_in->points[i].x;
        p.y = cloud_in->points[i].y;
        p.z = cloud_in->points[i].z;

        std::vector<int> k_indices(1);
        std::vector<float> k_sqr_distances(1);
        this->objects_kdtree_->nearestKSearch(p, 1, k_indices, k_sqr_distances);
        Bndbox nearest_object = filter_BBox[k_indices[0]];

        if (k_sqr_distances[k_indices[0]] > 2.0) {
            cloud_out->points.push_back(cloud_in->points[i]);
            continue;
        }
        if ((p.x < nearest_object.x - nearest_object.l / 2 || p.x > nearest_object.x + nearest_object.l / 2) || 
            (p.y < nearest_object.y - nearest_object.w / 2 || p.y > nearest_object.y + nearest_object.w / 2) ||
            (p.z < nearest_object.z - nearest_object.h / 2 || p.z > nearest_object.z + nearest_object.h / 2)) {
            cloud_out->points.push_back(cloud_in->points[i]);
            continue;
        }
        cloud_cluster->points.push_back(p);
    }
    // std::cout << "cluster size is: " << cloud_cluster->size() << std::endl;
}


void Center_PointPillars_ROS::publishCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr) {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
    cloud_msg.header = header;
    cloud_msg.header.frame_id = this->child_frame_;
    this->pub_pointcloud_static_.publish(cloud_msg);
}


void Center_PointPillars_ROS::publishObjectBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> filter_BBox) {

    jsk_recognition_msgs::BoundingBoxArray arr_bbox;
    int i = 0;

    for (const auto box : filter_BBox) {
        jsk_recognition_msgs::BoundingBox bbox;

        bbox.header = in_msg_header;
        bbox.header.frame_id = this->child_frame_;
        bbox.pose.position.x =  box.x;
        bbox.pose.position.y =  box.y;
        bbox.pose.position.z = box.z;
        bbox.dimensions.x = box.w;  // width
        bbox.dimensions.y = box.l;  // length
        bbox.dimensions.z = box.h;  // height
        // Using tf::Quaternion for quaternion from roll, pitch, yaw
        tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, -box.rt);
        bbox.pose.orientation.x = q.x();
        bbox.pose.orientation.y = q.y();
        bbox.pose.orientation.z = q.z();
        bbox.pose.orientation.w = q.w();
        bbox.value = box.score;
        bbox.label = box.id;
        arr_bbox.boxes.push_back(bbox);
        // if(box.score>0.5){
        // arr_bbox.boxes.push_back(bbox);
        // }

    }
    // std::cout<<"find bbox Num:"<<arr_bbox.boxes.size()<<std::endl;
    arr_bbox.header = in_msg_header;
    arr_bbox.header.frame_id = this->child_frame_;

    this->pub_bbox_.publish(arr_bbox);
}


void Center_PointPillars_ROS::publishDynamicBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> dynamic_BBox) {

    jsk_recognition_msgs::BoundingBoxArray arr_bbox;
    visualization_msgs::MarkerArray text_vel_array;
    visualization_msgs::Marker text_vel, center_points;
    static int i = 0;
    center_points.lifetime = ros::Duration();
    center_points.header = in_msg_header;
    center_points.header.frame_id = this->child_frame_;
    center_points.ns = "center_points";
    center_points.action = visualization_msgs::Marker::ADD;
    center_points.type = visualization_msgs::Marker::POINTS;
    center_points.scale.x = 0.7;
    center_points.scale.y = 0.7;
    center_points.scale.z = 0.7;

    for (const auto box : dynamic_BBox) {
        jsk_recognition_msgs::BoundingBox bbox;

        bbox.header = in_msg_header;
        bbox.header.frame_id = this->child_frame_;  // Replace with your frame_id
        bbox.pose.position.x =  box.x;
        bbox.pose.position.y =  box.y;
        bbox.pose.position.z = box.z;
        bbox.dimensions.x = box.w;  // width
        bbox.dimensions.y = box.l;  // length
        bbox.dimensions.z = box.h;  // height
        // Using tf::Quaternion for quaternion from roll, pitch, yaw
        tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, -box.rt);
        bbox.pose.orientation.x = q.x();
        bbox.pose.orientation.y = q.y();
        bbox.pose.orientation.z = q.z();
        bbox.pose.orientation.w = q.w();
        bbox.value = box.score;
        bbox.label = box.id;
        arr_bbox.boxes.push_back(bbox);
        // if(box.score>0.5){
        // arr_bbox.boxes.push_back(bbox);
        // }
        
        text_vel.header = in_msg_header;
        text_vel.header.frame_id = this->child_frame_;
        text_vel.ns = "dynamic_vel";
        text_vel.action = visualization_msgs::Marker::ADD;
        text_vel.id = i++;
        text_vel.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_vel.scale.z = 1;
        text_vel.color.r = text_vel.color.b = text_vel.color.g =  1;
        text_vel.color.a = 1;
        float vel = box.vy;
        std::ostringstream oss;
        oss << std::setprecision(2) << vel;
        text_vel.text = oss.str() + "m/s";
        text_vel.pose.orientation.x = q.x();
        text_vel.pose.orientation.y = q.y();
        text_vel.pose.orientation.z = q.z();
        text_vel.pose.orientation.w = q.w();
        text_vel.pose.position.x = box.x;
        text_vel.pose.position.y = box.y;
        text_vel.pose.position.z = box.z + box.h / 2 + 0.1;
        text_vel_array.markers.push_back(text_vel);

        center_points.id = i;
        center_points.type = visualization_msgs::Marker::POINTS;
        center_points.color.r = color[int(3) * i];
        center_points.color.b = color[int(3) * i + 1];
        center_points.color.g = color[int(3) * i + 2];
        center_points.color.a = 0.7;
        center_points.pose.orientation.w = 1;
        geometry_msgs::Point p;
        p.x = box.x;
        p.y = box.y;
        p.z = box.z;
        center_points.points.push_back(p);
        
        this->center_points_array.markers.push_back(center_points);

    }
    // std::cout<<"find bbox Num:"<<arr_bbox.boxes.size()<<std::endl;
    arr_bbox.header = in_msg_header;
    arr_bbox.header.frame_id = this->child_frame_;

    this->pub_dynamic_bbox_.publish(arr_bbox);
    this->pub_text_vel_.publish(text_vel_array);
    this->pub_center_points_.publish(this->center_points_array);
}



void Center_PointPillars_ROS::publishClusterCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<pcl::PointIndices> cluster_indices) {
    int color_index = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_point(new pcl::PointCloud<pcl::PointXYZRGB>());
    int clusterSize = cluster_indices.size();
    for (int i = 0; i < clusterSize; i++) {
        int clusterindixSize = cluster_indices[i].indices.size();
        for (int j = 0; j < clusterindixSize; j++) {
            pcl::PointXYZRGB point;
            point.x = cloud_in->points[cluster_indices[i].indices[j]].x;
            point.y = cloud_in->points[cluster_indices[i].indices[j]].y;
            point.z = cloud_in->points[cluster_indices[i].indices[j]].z;
            point.r = color[int(3) * color_index];
            point.g = color[int(3) * color_index + 1];
            point.b = color[int(3) * color_index + 2];
            color_point->push_back(point);
        }
        color_index++;
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*color_point, cloud_msg);
    cloud_msg.header = header;
    cloud_msg.header.frame_id = this->child_frame_;
    this->pub_pointcloud_cluster_.publish(cloud_msg);
}


void Center_PointPillars_ROS::publishVelArrows(pcl::PointCloud<pcl::PointXYZ> egoTFPoints, ros::Time input_time) {
    
    int targetSize = this->targetPoints.size();
    for(int i = 0; i < targetSize; i++){
        visualization_msgs::Marker arrowsG;
        arrowsG.lifetime = ros::Duration(0.1);
        if(this->trackManage[i] == 0 ) {
        continue;
        }
        if(this->isVisVec[i] == false ) {
        continue;
        }
        if(this->isStaticVec[i] == true){
        continue;
        }
        arrowsG.header.frame_id = this->child_frame_;
        
        arrowsG.header.stamp= input_time;
        arrowsG.ns = "arrows";
        arrowsG.action = visualization_msgs::Marker::ADD;
        arrowsG.type =  visualization_msgs::Marker::ARROW;
        // green
        arrowsG.color.g = 1.0f;
        // arrowsG.color.r = 1.0f;
        arrowsG.color.a = 1.0;  
        arrowsG.id = i;
        geometry_msgs::Point p;
        // assert(targetPoints[i].size()==4);
        p.x = egoTFPoints[i].x;
        p.y = egoTFPoints[i].y;
        p.z = -1.73/2;
        double tv   = this->targetVandYaw[i][0];
        double tyaw = this->targetVandYaw[i][1];

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

        this->pub_arrows_.publish(arrowsG);
    }

}


void Center_PointPillars_ROS::publishTrackingCenter(pcl::PointCloud<pcl::PointXYZ> egoTFPoints, ros::Time input_time) {

    visualization_msgs::Marker pointsY, pointsG, pointsR, pointsB;
    pointsY.header.frame_id = pointsG.header.frame_id = pointsR.header.frame_id = pointsB.header.frame_id = this->child_frame_;
    
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

    // yellow
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

    int targetSize = this->targetPoints.size();
    for(int i = 0; i < targetSize; i++){
        if(this->trackManage[i] == 0) continue;
        geometry_msgs::Point p;
        // p.x = this->targetPoints[i].x;
        // p.y = this->targetPoints[i].y;
        p.x = egoTFPoints[i].x;
        p.y = egoTFPoints[i].y;
        p.z = -1.73/2;

        if(this->isStaticVec[i] == true){
        pointsB.points.push_back(p);
        }
        else if(this->trackManage[i] < 5 ){
        pointsY.points.push_back(p);
        }
        else if(this->trackManage[i] == 5){
        pointsG.points.push_back(p);
        }
        else if(this->trackManage[i] > 5){
        pointsR.points.push_back(p);
        }
    }
    this->pub_tracking_center_.publish(pointsY);
    this->pub_tracking_center_.publish(pointsG);
    this->pub_tracking_center_.publish(pointsR);
    this->pub_tracking_center_.publish(pointsB);
}


void Center_PointPillars_ROS::publishBoundingBoxMarkers(ros::Time input_time) {

    visualization_msgs::Marker line_list;
    line_list.header.frame_id = this->child_frame_;

    line_list.header.stamp = input_time;
    line_list.ns =  "boxes";
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;

    line_list.id = 0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;

    //LINE_LIST markers use only the x component of scale, for the line width
    line_list.scale.x = 0.1;
    // Points are green
    line_list.color.g = 1.0f;
    line_list.color.a = 1.0;

    int id = 0;std::string ids;
    int targetSize = this->visBBs.size();
    for(int objectI = 0; objectI < targetSize; objectI ++){
        for(int pointI = 0; pointI < 4; pointI++){
            assert((pointI+1)%4 < this->visBBs[objectI].size());
            assert((pointI+4) < this->visBBs[objectI].size());
            assert((pointI+1)%4+4 < this->visBBs[objectI].size());
            id ++; ids = to_string(id);
            geometry_msgs::Point p;
            p.x = this->visBBs[objectI][pointI].x;
            p.y = this->visBBs[objectI][pointI].y;
            p.z = this->visBBs[objectI][pointI].z;
            line_list.points.push_back(p);
            p.x = this->visBBs[objectI][(pointI+1)%4].x;
            p.y = this->visBBs[objectI][(pointI+1)%4].y;
            p.z = this->visBBs[objectI][(pointI+1)%4].z;
            line_list.points.push_back(p);

            p.x = this->visBBs[objectI][pointI].x;
            p.y = this->visBBs[objectI][pointI].y;
            p.z = this->visBBs[objectI][pointI].z;
            line_list.points.push_back(p);
            p.x = this->visBBs[objectI][pointI+4].x;
            p.y = this->visBBs[objectI][pointI+4].y;
            p.z = this->visBBs[objectI][pointI+4].z;
            line_list.points.push_back(p);

            p.x = this->visBBs[objectI][pointI+4].x;
            p.y = this->visBBs[objectI][pointI+4].y;
            p.z = this->visBBs[objectI][pointI+4].z;
            line_list.points.push_back(p);
            p.x = this->visBBs[objectI][(pointI+1)%4+4].x;
            p.y = this->visBBs[objectI][(pointI+1)%4+4].y;
            p.z = this->visBBs[objectI][(pointI+1)%4+4].z;
            line_list.points.push_back(p);
        }
    }
    this->pub_box_markers_.publish(line_list);
}


void Center_PointPillars_ROS::preprocessPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in, float th1, float th2) {

    // Original Scan
    *this->original_scan_ = *cloud_in;
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*this->original_scan_, cloud_msg);
    // cloud_msg.header.stamp = cloud_in->header.stamp;
    cloud_msg.header.frame_id = this->child_frame_;
    this->pub_pointcloud_raw_.publish(cloud_msg);

    // Remove NaNs
    std::vector<int> idx;
    cloud_in->is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, idx);
    this->removeClosedPointCloud(*cloud_in, *cloud_in, th1, th2);

    // Crop Box Filter
    if (this->crop_use_) {
        this->crop.setInputCloud(cloud_in);
        this->crop.filter(*cloud_in);
    }

    // Voxel Grid Filter
    if (this->vf_use_) {
        this->vf.setInputCloud(cloud_in);
        this->vf.filter(*cloud_in);
    }

}

void Center_PointPillars_ROS::removeClosedPointCloud(const pcl::PointCloud<pcl::PointXYZI> &cloud_in, pcl::PointCloud<pcl::PointXYZI> &cloud_out, float th1, float th2)
{
  if (&cloud_in != &cloud_out)
  {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  size_t j = 0;

  for (size_t i = 0; i < cloud_in.points.size(); ++i)
  {
    float dis = cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z;
    if (dis < th1 * th1)
      continue;
    if (dis > th2 * th2)
      continue;
    cloud_out.points[j++] = cloud_in.points[i];
  }

  if (j != cloud_in.points.size())
  {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<uint32_t>(j);
  cloud_out.is_dense = true;
}


void Center_PointPillars_ROS::mot_3d(std_msgs::Header in_msg_header, std::vector<Bndbox> filter_BBox, std::vector<Bndbox>& dynamic_BBox)
{
    double timestamp = in_msg_header.stamp.toSec();
    ros::Time input_time = in_msg_header.stamp;

    int box_num = filter_BBox.size();
    // convert local to global-------------------------
    std::vector<pcl::PointCloud<pcl::PointXYZ>> bBoxes;
    pcl::PointCloud<pcl::PointXYZ> oneBbox;

    for(int i = 0; i < box_num; i++)
    {
        pcl::PointXYZ o;
        o.x = filter_BBox[i].x - filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y - filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z - filter_BBox[i].h / 2;
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x + filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y - filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z - filter_BBox[i].h / 2; 
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x + filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y + filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z - filter_BBox[i].h / 2;   
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x - filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y + filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z - filter_BBox[i].h / 2;  
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x - filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y - filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z + filter_BBox[i].h / 2;  
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x + filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y - filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z + filter_BBox[i].h / 2;  
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x + filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y + filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z + filter_BBox[i].h / 2;  
        oneBbox.push_back(o);
        o.x = filter_BBox[i].x - filter_BBox[i].l / 2;
        o.y = filter_BBox[i].y + filter_BBox[i].w / 2;
        o.z = filter_BBox[i].z + filter_BBox[i].h / 2;

        bBoxes.push_back(oneBbox);
        oneBbox.clear();
    }

    

    Eigen::Matrix4f last_T;
    
    if (this->odom_queue.empty())
    {
        last_T.setIdentity();
    }
    else
    {
        Eigen::Quaternionf last_q(this->odom_queue.front().pose.pose.orientation.w, this->odom_queue.front().pose.pose.orientation.x, this->odom_queue.front().pose.pose.orientation.y, this->odom_queue.front().pose.pose.orientation.z);
        last_T.block<3,3>(0,0) = last_q.toRotationMatrix();
        Eigen::Vector3f last_t(this->odom_queue.front().pose.pose.position.x, this->odom_queue.front().pose.pose.position.y, this->odom_queue.front().pose.pose.position.z);
        last_T.block<3,1>(0,3) = last_t;
    }
    // getOriginPoints(last_T.block<3,3>(0,0));

    pcl::PointCloud<pcl::PointXYZ> newBox;
    for(int i = 0; i < box_num; i++ ){
        bBoxes[i].header.frame_id = this->odom_frame_;
        
        pcl::transformPointCloud(bBoxes[i], newBox, last_T);
        bBoxes[i] = newBox;
        bBoxes[i].header.frame_id = this->odom_frame_;
    }
    //end converting----------------------------------------
    this->targetPoints.clear();
    this->targetVandYaw.clear();
    this->trackManage.clear();
    this->isStaticVec.clear();
    this->isVisVec.clear();
    this->visBBs.clear();
    double t1 = ros::Time::now().toSec();
    immUkfJpdaf(bBoxes, timestamp, this->targetPoints, this->targetVandYaw, this->trackManage, this->isStaticVec, this->isVisVec, this->visBBs);
    double t2 = ros::Time::now().toSec();
    avg_ukf_time.push_back((t2 - t1) * 1000);
    // ROS_INFO("UKF cost time:%f ms", (t2 - t1) * 1000);

    assert(targetPoints.size() == trackManage.size());
    assert(targetPoints.size()== targetVandYaw.size());

    //start converting to ego tf-------------------------
    if (this->odom_queue.empty())
    {
        last_T.setIdentity();
    }
    else
    {
        Eigen::Quaternionf last_q(this->odom_queue.front().pose.pose.orientation.w, this->odom_queue.front().pose.pose.orientation.x, this->odom_queue.front().pose.pose.orientation.y, this->odom_queue.front().pose.pose.orientation.z);
        last_T.block<3,3>(0,0) = last_q.toRotationMatrix().inverse();
        Eigen::Vector3f last_t(this->odom_queue.front().pose.pose.position.x, this->odom_queue.front().pose.pose.position.y, this->odom_queue.front().pose.pose.position.z);
        last_t = -last_T.block<3,3>(0,0) * last_t;
        last_T.block<3,1>(0,3) = last_t;
    }

    // converting from global to ego tf for visualization
    // processing targetPoints
    pcl::PointCloud<pcl::PointXYZ> egoTFPoints;
    this->targetPoints.header.frame_id = this->child_frame_;
    pcl::transformPointCloud(targetPoints, egoTFPoints, last_T);

    //processing visBBs
    pcl::PointCloud<pcl::PointXYZ> visEgoBB;
    for(int i = 0; i < visBBs.size(); i++){
        this->visBBs[i].header.frame_id = this->child_frame_;
        pcl::transformPointCloud(this->visBBs[i], visEgoBB, last_T);
        this->visBBs[i] = visEgoBB;
    }
    //end converting to ego tf---------------------------

    pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud;
    objects_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto box : filter_BBox) {
        pcl::PointXYZ p;
        p.x = box.x;
        p.y = box.y;
        p.z = 0;
        objects_cloud->points.push_back(p);
    }
    this->objects_kdtree_->setInputCloud(objects_cloud);

    int targetSize = targetPoints.size();
    for (int i = 0; i < targetSize; i++)
    {
        if(this->trackManage[i] == 0 ) {
        continue;
        }
        if(this->isVisVec[i] == false ) {
        continue;
        }
        if(this->isStaticVec[i] == true){
        continue;
        }
        pcl::PointXYZ p;
        p.x = egoTFPoints[i].x;
        p.y = egoTFPoints[i].y;
        p.z = 0;

        std::vector<int> k_indices(1);
        std::vector<float> k_sqr_distances(1);
        this->objects_kdtree_->nearestKSearch(p, 1, k_indices, k_sqr_distances);
        Bndbox nearest_object = filter_BBox[k_indices[0]];
        dynamic_BBox.push_back(nearest_object);
    }

    if (!this->odom_queue.empty())
        this->odom_queue.pop_front();

    // this->publishVelArrows(egoTFPoints, input_time);
    // this->publishTrackingCenter(egoTFPoints, input_time);
    // this->publishBoundingBoxMarkers(input_time);
}




}




int main(int argc, char **argv) {
    ros::init(argc, argv, "centerpp_node");
    ros::NodeHandle nh("~");

    color.clear();
    for (size_t i_segment = 0; i_segment < 100; i_segment++)
    {
        color.push_back(static_cast<unsigned char>(rand() % 256));
        color.push_back(static_cast<unsigned char>(rand() % 256));
        color.push_back(static_cast<unsigned char>(rand() % 256));
    }

    // GetDeviceInfo();
    initDevice(0);

    cpp::Center_PointPillars_ROS center_pintPillars_ros(nh);
    center_pintPillars_ros.Process();

    return 0;
}
