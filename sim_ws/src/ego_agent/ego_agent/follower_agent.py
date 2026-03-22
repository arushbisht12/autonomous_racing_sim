#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from visualization_msgs.msg import Marker

from ament_index_python.packages import get_package_share_directory

import numpy as np

import csv
import os

class FollowerAgent(Node):
    def __init__(self):
        super().__init__('follower_agent')
        
        # PARAMETERS
        self.dt = 0.01
        self.look_ahead = 1.0
        self.target_gap = 2.0
        self.kp = 1.0
        self.wheel_base = 0.20

        # STATE
        self.pos = np.array([0.0, 0.0]) #fix
        self.theta = 0.0
        self.v = 0.0
        self.state_received = False
        
        # OPP STATE
        self.opp_pos = [0.0, 0.0] #fix
        self.opp_theta = 0.0 
        self.opp_mode = 'NICE'
        self.opp_received = False
        
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.opp_sub = self.create_subscription(Odometry, '/ego_racecar/opp_odom', self.opp_callback, 10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        self.opp_prediction_pub = self.create_publisher(Marker, '/debug/opp_prediction', 10)
        
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.opp_prediction_timer = self.create_timer(1, self.publish_opp_prediction)
        
        
        # get track centerline way points
        #pkg_dir = get_package_share_directory('f1tenth_gym_ros')
        #map_path = os.path.join(pkg_dir, 'maps/Spielberg_centerline.csv')
        map_path = '/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_centerline.csv'
        self.waypoints = []
        with open(map_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header if there is one (Check your file first!)
            for row in reader:
                self.waypoints.append([float(row[0]), float(row[1])])
        self.waypoints = np.array(self.waypoints)
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")
        
    def publish_opp_prediction(self):
        pass
        
    def odom_callback(self, msg):
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.theta = self._quaternion_to_euler(q)
        self.state_received = True
    
    def opp_callback(self, msg):
        self.opp_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.opp_v = msg.twist.twist.linear.x
        self.opp_received = True
    
    def _quaternion_to_euler(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def control_loop(self):
        if not (self.state_received and self.opp_received):
            return
        
        
        # pure pursuit controller
        # find path point closest to vehicle
        distances = np.linalg.norm(self.pos - self.waypoints, axis = 1)
        closest_wp = np.argmin(distances)
        # find goal point
        goal_wp = closest_wp
        while distances[goal_wp] < self.look_ahead:
            goal_wp = (goal_wp + 1) % len(self.waypoints) # for overflow
            if goal_wp == closest_wp:
                break
        
        goal_pos = self.waypoints[goal_wp]
        
        dx_map = goal_pos[0] - self.pos[0]
        dy_map = goal_pos[1] - self.pos[1]
        # transformation to ego frame
        y_local = -dx_map * np.sin(self.theta) + dy_map * np.cos(self.theta)
        
        look_ahead_actual = np.linalg.norm([dx_map, dy_map]) # instead of self.waypoints[current_wp][0]
        curvature = 2.0 * y_local / look_ahead_actual**2
        steering_angle = np.arctan(curvature * self.wheel_base)
        
        ########
        # speed p controller
        dist_err = np.linalg.norm(self.pos-self.opp_pos)
        self.v = self.opp_v + self.kp * (dist_err - self.target_gap)
        
        # saturate velocities
        #self.v = np.clip(self.v, v_min, v_max)
        
        # debug
        closest_pos = self.waypoints[closest_wp]

        self.get_logger().info(f'pos: {self.pos}', throttle_duration_sec=0.5)
        self.get_logger().info(f'closest waypoint {closest_wp}: {closest_pos}', throttle_duration_sec=0.5)
        self.get_logger().info(f'goal waypoint {goal_wp}: {goal_pos}', throttle_duration_sec=0.5)
        
        # publish commands
        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = self.v
        self.cmd_pub.publish(cmd)
 
    
def main(args=None):
    rclpy.init(args=args)
    node = FollowerAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
        