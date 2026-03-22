#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from ament_index_python.packages import get_package_share_directory

import numpy as np

import csv
import os

class MeanAgent(Node):
    def __init__(self):
        super().__init__('mean_agent')
        
        # PARAMETERS
        self.dt = 0.01
        self.look_ahead = 1.0
        self.v = 1.0 # m/s    constant speed?
        self.wheel_base = 0.3302
        
        # marker sphere radius
        self.rad = 0.25
        
        self.A = 0.8
        self.w = 0.5 # rad/meter

        # STATE
        self.pos = np.array([0.0, 0.0])
        self.theta = 0.0
        
        self.state_received = False
        
        self.goal_pos = np.array([0, 0])
        self.closest_pos = np.array([0, 0])
        
        self.odom_sub = self.create_subscription(Odometry, '/opp_racecar/odom', self.odom_callback, 10)
        
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/debug/target_point', 10)
        self.closest_marker_pub = self.create_publisher(Marker, '/debug/current_point', 10)
        self.path_pub = self.create_publisher(Marker, 'debug/path', 10)
        
        self.traj_pub = self.create_publisher(Marker, '/debug/trajectory', 10)
        self.N_trajectory_points = 15
        
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        #self.closest_marker_timer = self.create_timer(self.dt, self.publish_closest_waypoint)
        #self.goal_marker_timer = self.create_timer(self.dt, self.publish_goal_waypoint)
        #self.path_timer = self.create_timer(1, self.publish_global_path)
        
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
        #self.waypoints = self.waypoints[::-1]
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")
        
        # adjust waypoints to have sine offset
        s = 0
        self.mean_waypoints = self.waypoints.copy()
        for wp in range(len(self.waypoints)-1, -1, -1):
            curr_wp = self.waypoints[wp]
            next_wp = self.waypoints[(wp - 1) % len(self.waypoints)]
            
            forward_difference = next_wp-curr_wp
            ds = np.linalg.norm(forward_difference)
            
            #track_heading = np.arctan(central_difference[0], central_difference[1])
            tangent_vector = forward_difference / np.linalg.norm(forward_difference)
            normal_vector = np.array([-tangent_vector[1], tangent_vector[0]]) # 90 degree rotation
            self.mean_waypoints[wp] = curr_wp + normal_vector * self.A * np.sin(self.w * s) 
            
            s = s + ds
        self.get_logger().info("Generated Mean Agent Trajectory")
       
    def publish_global_path(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "track_centerline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.1
        
        marker.color.a = 1.0 
        marker.color.b = 0.5
        marker.color.g = 0.0
        marker.color.r = 0.0
        
        # waypoints as points
        for wp in self.mean_waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            marker.points.append(p)
            
        self.path_pub.publish(marker)
  
    def publish_goal_waypoint(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "pure_pursuit"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.goal_pos[0])
        marker.pose.position.y = float(self.goal_pos[1])
        marker.pose.position.z = 0.0 # Lift it off the ground so you can see it
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = self.rad
        marker.scale.y = self.rad
        marker.scale.z = self.rad
        
        marker.color.a = 1.0  # Alpha (Transparency) - MUST be 1.0 to see it!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    
        self.goal_marker_pub.publish(marker)
        
    
    def publish_closest_waypoint(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "pure_pursuit"  
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.closest_pos[0])
        marker.pose.position.y = float(self.closest_pos[1])
        marker.pose.position.z = 0.0 # Lift it off the ground so you can see it
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = self.rad
        marker.scale.y = self.rad
        marker.scale.z = self.rad
        
        marker.color.a = 1.0  # Alpha (Transparency) - MUST be 1.0 to see it!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    
        self.closest_marker_pub.publish(marker)       
        
    def odom_callback(self, msg):
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.theta = self._quaternion_to_euler(q)
        self.state_received = True
        
    def _quaternion_to_euler(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def publish_trajectory(self, closest_wp):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "mean_trajectory"
        marker.id = 2
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        
        # Scale of the individual spheres
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        # Cyan color so it stands out against red/green
        marker.color.a = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        # Collect the next N points 
        # (Using minus because your pure pursuit logic decrements to go forward)
        for i in range(1, self.N_trajectory_points):
            wp_idx = (closest_wp - i) % len(self.mean_waypoints)
            
            p = Point()
            p.x = float(self.mean_waypoints[wp_idx][0])
            p.y = float(self.mean_waypoints[wp_idx][1])
            p.z = 0.0
            marker.points.append(p)
            
        self.traj_pub.publish(marker)
        
    def control_loop(self):
        if not self.state_received:
            return
        
        # pure pursuit (track centerline)
        
        # find path point closest to vehicle
        distances = np.linalg.norm(self.pos - self.mean_waypoints, axis = 1)
        closest_wp = np.argmin(distances)
        # find goal point
        goal_wp = closest_wp
        while distances[goal_wp] < self.look_ahead:
            goal_wp = (goal_wp + 1) % len(self.mean_waypoints) # for overflow
            if goal_wp == closest_wp:
                break
        
        self.goal_pos = self.mean_waypoints[goal_wp]
        
        dx_map = self.goal_pos[0] - self.pos[0]
        dy_map = self.goal_pos[1] - self.pos[1]
        # transformation to ego frame
        y_local = -dx_map * np.sin(self.theta) + dy_map * np.cos(self.theta)
        
        look_ahead_actual = np.linalg.norm([dx_map, dy_map]) # instead of self.waypoints[current_wp][0]
        curvature = 2.0 * y_local / look_ahead_actual**2
        steering_angle = np.arctan(curvature * self.wheel_base)
        
        # saturate velocities
        
        # debug
        self.closest_pos = self.mean_waypoints[closest_wp]
        
        self.publish_trajectory(closest_wp)
        
        # publish cmd
        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = self.v
        self.cmd_pub.publish(cmd)
        
    def update_plot():
        pass
    

def main(args=None):
    rclpy.init(args=args)
    node = MeanAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()