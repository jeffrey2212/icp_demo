#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
from rospy.numpy_msg import numpy_msg

class ICPNode:
  def __init__(self):
    rospy.init_node('icp_node', anonymous=True)
    self.pc_subscriber = rospy.Subscriber("/scan_to_pointcloud", PointCloud2, self.point_cloud_callback)
    self.icp_aligned_pub = rospy.Publisher("/icp_aligned_cloud", PointCloud2, queue_size=10)

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

    # Transform the source cloud to the target frame
    source.transform(icp_result.transformation)

    # Convert the aligned Open3D PointCloud back to ROS PointCloud2
    aligned_points = np.asarray(source.points)
    header = msg.header
    aligned_cloud = pc2.create_cloud_xyz32(header, aligned_points)

    # Publish the aligned cloud
    self.icp_aligned_pub.publish(aligned_cloud)

if __name__ == '__main__':
  icp_node = ICPNode()
  rospy.spin()