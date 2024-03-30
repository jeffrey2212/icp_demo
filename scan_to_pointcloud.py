#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from laser_geometry import LaserProjection

class ScanToPointCloud():
    def __init__(self):
        rospy.init_node('scan_to_pointcloud')
        self.laser_projector = LaserProjection() 
        self.pc2_pub = rospy.Publisher('/scan_to_pointcloud', PointCloud2, queue_size=10)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

    def laser_callback(self, msg):
        cloud_out = self.laser_projector.projectLaser(msg)  # Project into PointCloud2
        self.pc2_pub.publish(cloud_out)

if __name__ == '__main__':
    converter = ScanToPointCloud()
    rospy.spin()
