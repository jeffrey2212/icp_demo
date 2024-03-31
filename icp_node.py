#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, LaserScan
from laser_geometry import LaserProjection 
import open3d as o3d
from geometry_msgs.msg import PoseStamped
import numpy as np
from nav_msgs.msg import Odometry 
import tf.transformations as tft

from kalman_filters import KalmanFilter 

class ICPNode:
  def __init__(self):
    rospy.init_node('icp_node', anonymous=True)
    self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
    self.laser_projector = LaserProjection()
    self.odom_subscriber = rospy.Subscriber("/odom", Odometry, self.odom_callback)
    self.kf = KalmanFilter()
    self.refined_pose_publisher = rospy.Publisher('/refined_pose', PoseStamped, queue_size=10)
    self.latest_odom_pose = None  
    self.latest_linear_velocity = None 
    self.latest_angular_velocity = None
    self.prev_odom_time = None 
        
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
    self.latest_linear_velocity = msg.twist.twist.linear.x
    self.latest_angular_velocity = msg.twist.twist.angular.z
    self.prev_odom_time = msg.header.stamp.to_sec()
    
  def laser_callback(self, msg):
    cloud_out = self.laser_projector.projectLaser(msg)  
    pc_array = pc2.read_points_list(cloud_out, field_names=("x", "y", "z"), skip_nans=True)
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
    # Ensure icp_pose is a column vector before passing to update
    icp_pose = np.array([[icp_result.transformation[0, 3]],
                         [icp_result.transformation[1, 3]],
                         [np.arctan2(icp_result.transformation[1, 0], icp_result.transformation[0, 0])]])
    self.kf.update(icp_pose)  # Pass icp_pose to the update method
    
    # 2. Get Odometry Data
    if self.latest_odom_pose is None or self.prev_odom_time is None:
        rospy.logwarn("No odometry received yet. Skipping filter prediction.")
        return

    current_time = msg.header.stamp.to_sec()
    dt = current_time - self.prev_odom_time
    self.prev_odom_time = current_time

    # 4. Kalman Filter
    v = self.latest_linear_velocity
    omega = self.latest_angular_velocity
    self.kf.predict(v, omega, dt)  # Now we pass both velocity and angular velocity along with dt
    self.kf.update(icp_pose) 
    refined_pose = self.kf.get_state() 

    # 5. Publish Refined Pose
    # x = refined_pose[0]
    # y = refined_pose[1]
    # theta = refined_pose[2]

    # # Construct Rotation Matrix
    # R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, theta))  

    # # Construct Translation Vector
    # T = np.array([x, y, 0]).reshape(3, 1) 

    # # Combine Rotation and Translation
    # refined_transformation = np.eye(4)  
    # refined_transformation[:3, :3] = R 
    # refined_transformation[:3, 3] = T.reshape(3) 

    # # Transform the source cloud
    # source.transform(refined_transformation) 

    # # Convert the aligned Open3D PointCloud back to ROS PointCloud2
    # aligned_points = np.asarray(source.points)
    # header = msg.header
    # aligned_cloud = pc2.create_cloud_xyz32(header, aligned_points)

    # # Publish the aligned cloud
    # # rospy.loginfo("Publishing aligned point cloud")
    # # self.icp_aligned_pub.publish(aligned_cloud)
    # # rospy.loginfo("Aligned point cloud published")
    pose_msg = PoseStamped()
    pose_msg.header.stamp = msg.header.stamp  # Synchronize timestamp
    pose_msg.pose.position.x = refined_pose[0]
    pose_msg.pose.position.y = refined_pose[1]
    # ... Set orientation based on refined_pose[2] ... 
    self.refined_pose_publisher.publish(pose_msg)

if __name__ == '__main__':
  icp_node = ICPNode()
  rospy.spin()