#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include <GeographicLib/LocalCartesian.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

class GPSToNED : public rclcpp::Node
{
public:
  GPSToNED() : Node("gps_to_ned_node"), ref_lat(-33.72273279219146), ref_lon(150.6739915818924), ref_alt(0.9164452971890569)
  {
    gps_subscription_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "/wamv/sensors/gps/gps/fix", 10, std::bind(&GPSToNED::gps_callback, this, std::placeholders::_1));
    imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/wamv/sensors/imu/imu/data", 10, std::bind(&GPSToNED::imu_callback, this, std::placeholders::_1));
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    ned_publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>("ned_coordinates", 10);
    orientation_publisher_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("imu_orientation_degrees", 10);

    geodetic_converter_.Reset(ref_lat, ref_lon, ref_alt);
  }

private:
  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
  {
    double north, east, down;
    geodetic_converter_.Forward(msg->latitude, msg->longitude, msg->altitude, north, east, down);

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = this->now();
    transform.header.frame_id = "map";
    transform.child_frame_id = "wamv/wamv/base_link";
    transform.transform.translation.x = north;
    transform.transform.translation.y = east;
    transform.transform.translation.z = down;
    transform.transform.rotation = imu_orientation_;

    tf_broadcaster_->sendTransform(transform);

    // Publish NED coordinates
    geometry_msgs::msg::PointStamped ned_msg;
    ned_msg.header.stamp = this->now();
    ned_msg.header.frame_id = "map";
    ned_msg.point.x = north;
    ned_msg.point.y = east;
    ned_msg.point.z = down;
    ned_publisher_->publish(ned_msg);

    // Convert IMU orientation to degrees and publish
    geometry_msgs::msg::Vector3Stamped orientation_msg;
    orientation_msg.header.stamp = this->now();
    orientation_msg.header.frame_id = "map";

    tf2::Quaternion quat(imu_orientation_.x, imu_orientation_.y, imu_orientation_.z, imu_orientation_.w);
    tf2::Matrix3x3 mat(quat);
    double roll, pitch, yaw;
    mat.getRPY(roll, pitch, yaw);

    orientation_msg.vector.x = roll * 180.0 / M_PI;
    orientation_msg.vector.y = pitch * 180.0 / M_PI;
    orientation_msg.vector.z = yaw * 180.0 / M_PI;
    orientation_publisher_->publish(orientation_msg);

    // // Print NED coordinates
    // RCLCPP_INFO(this->get_logger(), "NED Coordinates: [North: %f, East: %f, Down: %f]", north, east, down);

    // // Print orientation in degrees
    // RCLCPP_INFO(this->get_logger(), "Orientation (degrees): [Roll: %f, Pitch: %f, Yaw: %f]",
    //             orientation_msg.vector.x, orientation_msg.vector.y, orientation_msg.vector.z);
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    imu_orientation_ = msg->orientation;
  }

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr ned_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr orientation_publisher_;
  geometry_msgs::msg::Quaternion imu_orientation_;

  const double ref_lat;
  const double ref_lon;
  const double ref_alt;
  GeographicLib::LocalCartesian geodetic_converter_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GPSToNED>());
  rclcpp::shutdown();
  return 0;
}