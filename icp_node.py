#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
from rospy.numpy_msg import numpy_msg
from nav_msgs.msg import Odometry 
import tf.transformations as tft

from kalman_filters import KalmanFilter 

class ICPNode:
  def __init__(self):
    rospy.init_node('icp_node', anonymous=True)
    self.pc_subscriber = rospy.Subscriber("/scan_to_pointcloud", PointCloud2, self.point_cloud_callback)
    self.icp_aligned_pub = rospy.Publisher("/icp_aligned_cloud", PointCloud2, queue_size=10)
    self.kf = KalmanFilter()
    self.odom_subscriber = rospy.Subscriber("/odom", Odometry, self.odom_callback)
    self.latest_odom_pose = None  # Store the latest odometry
    self.prev_odom_time = None
    self.latest_linear_velocity = 0
    self.latest_angular_velocity = 0

  
  @staticmethod
  def quaternion_to_euler(x, y, z, w):
    """
    Converts a quaternion (x, y, z, w) into Euler angles (roll, pitch, yaw)
    """
    euler = tft.euler_from_quaternion([x, y, z, w])
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return roll, pitch, yaw 
  
  def odom_callback(self, msg):
    self.latest_odom_pose = msg.pose.pose
    # Extract linear velocity (v) and angular velocity (omega)
    v = msg.twist.twist.linear.x  # Assuming forward velocity is along the x-axis
    omega = msg.twist.twist.angular.z  # Assuming rotational velocity is around the z-axis

    # Store velocities for use in the predict step
    self.latest_linear_velocity = v
    self.latest_angular_velocity = omega
    
  def point_cloud_callback(self, msg):
    # Convert ROS PointCloud2 to Open3D PointCloud
    pc_array = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.array(pc_array))

    # Setting the target cloud
    # For ICP trial run, we will set the target cloud to be the same as the source cloud
    target = source

    # Prepare the ICP configuration
    threshold = 0.02 # Set this to an appropriate value
    trans_init = np.eye(4) # Assuming initial alignment is identity matrix
    icp_result = o3d.pipelines.registration.registration_icp(
      source, target, threshold, trans_init,
      o3d.pipelines.registration.TransformationEstimationPointToPoint())

       # Kalman Filter Integration
    # 1. Extract ICP Pose 
    icp_pose = np.array([icp_result.transformation[0, 3], 
                         icp_result.transformation[1, 3],
                         np.arctan2(icp_result.transformation[1,0], icp_result.transformation[0,0])])  # Adjust indices if needed

    # 2. Get Odometry Data
    if self.latest_odom_pose is None or self.prev_odom_time is None:
        rospy.logwarn("No odometry received yet. Skipping filter prediction.")
        return

    # Extract Quaternion
    q = self.latest_odom_pose.orientation
    roll, pitch, theta = self.quaternion_to_euler(q.x, q.y, q.z, q.w)

    # Calculate Time Difference (dt)
    current_time = msg.header.stamp.to_sec()  # Use ICP message timestamp
    dt = current_time - self.prev_odom_time if self.prev_odom_time is not None else 0.1
    self.prev_odom_time = current_time

    # Assuming self.latest_linear_velocity and self.latest_angular_velocity have been set in the odom_callback
    v = self.latest_linear_velocity
    omega = self.latest_angular_velocity

    # 4. Kalman Filter
    self.kf.predict([v, omega], dt)  # Now we pass both velocity and angular velocity along with dt
    self.kf.update(icp_pose) 
    refined_pose = self.kf.get_state() 

    # 5. Publish Refined Pose
    x = refined_pose[0]
    y = refined_pose[1]
    theta = refined_pose[2]

    # Construct Rotation Matrix
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, theta))  

    # Construct Translation Vector
    T = np.array([x, y, 0]).reshape(3, 1) 

    # Combine Rotation and Translation
    refined_transformation = np.eye(4)  
    refined_transformation[:3, :3] = R 
    refined_transformation[:3, 3] = T.reshape(3) 

    # Transform the source cloud
    source.transform(refined_transformation) 
    # Transform the source cloud to the target frame
    #source.transform(icp_result.transformation)

    # Convert the aligned Open3D PointCloud back to ROS PointCloud2
    aligned_points = np.asarray(source.points)
    header = msg.header
    aligned_cloud = pc2.create_cloud_xyz32(header, aligned_points)

    # Publish the aligned cloud
    rospy.loginfo("Publishing aligned point cloud")
    self.icp_aligned_pub.publish(aligned_cloud)
    rospy.loginfo("Aligned point cloud published")

if __name__ == '__main__':
  icp_node = ICPNode()
  rospy.spin()