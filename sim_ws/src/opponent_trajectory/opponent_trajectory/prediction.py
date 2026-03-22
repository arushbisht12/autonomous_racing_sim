#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

import csv
import math
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev

# Helper for ROS2 Odometry quaternions
def get_yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class OpponentPrediction(Node):
    def __init__(self):
        super().__init__('opponent_prediction')
        
        # States: [s, n, v, psi]
        self.opp_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.ego_state = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.centerline = None
        self.nmax = 1.1
        self.w_margin = 0.2 # Increased slightly for safety
        self.Lwb = 0.33 # F1TENTH wheelbase
        
        self.s_c_dense = []
        self.psi_c_dense = []
        self.kap_c_dense = []
        
        self.contn_traj = []
        self.yield_traj = []
        self.block_traj = []
        
        # File paths (Adjust as needed)
        map_path = '/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_centerline.csv'
        raceline_path = '/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_raceline.csv'
        self.load_and_convert_raceline_spline(map_path, raceline_path)
        
        
        print(self.raceline[:10])
        
        # Publishers
        self.centerline_pub = self.create_publisher(Marker, 'debug/centerline', 10)
        self.raceline_pub = self.create_publisher(Marker, 'debug/raceline', 10)
        self.traj_pub = self.create_publisher(MarkerArray, 'predict/trajectories', 10)
        
        # Subscribers
        self.create_subscription(Odometry, '/ego_racecar/opp_odom', self.opp_odom_cb, 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.ego_odom_cb, 10)
        
        # Timers
        self.create_timer(1.0, self.publish_centerline)
        self.create_timer(1.0, self.publish_raceline)
        self.create_timer(0.1, self.predict_and_publish) # Run prediction at 10Hz

    # --- TRACK PROCESSING ---
    def load_and_convert_raceline_spline(self, map_path, raceline_path):
        # centerline
        waypoints = []
        with open(map_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                waypoints.append([float(row[0]), float(row[1])])
                
        raw_centerline = np.array(waypoints).T  # Shape: (2, N)
        
        # interpolation (b-spline)
        _, idx = np.unique(raw_centerline, axis=1, return_index=True)
        raw_centerline = raw_centerline[:, np.sort(idx)]
        
        x_raw, y_raw = raw_centerline[0, :], raw_centerline[1, :]
        
        tck, u = splprep([x_raw, y_raw], s=0, per=True) 
        
        # resample
        u_dense = np.linspace(0, 1.0, 1692)  # sample at the length of the raceline points
        x_c_dense, y_c_dense = splev(u_dense, tck)
        
        dx_c_dense, dy_c_dense = splev(u_dense, tck, der=1)
        # heading phi(s)
        self.psi_c_dense = np.arctan2(dy_c_dense, dx_c_dense)
        
        ddx_c_dense, ddy_c_dense = splev(u_dense, tck, der=2)
        # curvature kappa(s)
        self.kap_c_dense = (dx_c_dense * ddy_c_dense - ddx_c_dense * dy_c_dense) / (dx_c_dense**2 + dy_c_dense**2)**(3/2)
        
        ds = np.hypot(np.diff(x_c_dense), np.diff(y_c_dense))
        self.s_c_dense = np.insert(np.cumsum(ds), 0, 0.0)
        
        self.centerline = np.stack((x_c_dense, y_c_dense), axis=0)

        self.get_logger().info(f"Spline-fitted centerline generated: {self.centerline.shape}")


        # raceline
        raceline_data = np.loadtxt(raceline_path, delimiter=';', skiprows=3)
        x_r = raceline_data[:, 1]
        y_r = raceline_data[:, 2]
        psi_r = raceline_data[:, 3] 
        v_r = raceline_data[:, 5]
           
        self.raceline = np.stack((x_r, y_r), axis=0)

        # find lateral n using closest centerline point
        tree = KDTree(np.column_stack((x_c_dense, y_c_dense)))
        _, closest_idx = tree.query(np.column_stack((x_r, y_r)))

        s_r = np.zeros_like(x_r)
        n_r = np.zeros_like(x_r)

        for i in range(len(x_r)):
            idx = closest_idx[i]
            
            s_r[i] = self.s_c_dense[idx]
            
            normal_vec = np.array([-np.sin(self.psi_c_dense[idx]), np.cos(self.psi_c_dense[idx])]) 
            diff_vec = np.array([x_r[i] - x_c_dense[idx], y_r[i] - y_c_dense[idx]])
            n_r[i] = np.dot(diff_vec, normal_vec)

        self.raceline_frenet = np.column_stack((s_r, n_r, v_r, psi_r))
        self.raceline_s = s_r
        self.raceline_n = n_r
        self.raceline_v = v_r
        self.get_logger().info(f"Spline-fitted Frenet raceline generated: {self.raceline.shape}")
        
    # --- ODOMETRY & STATE CONVERSION ---
    def global_to_frenet(self, x, y, theta, v):
        if self.centerline is None: return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Find closest point on dense centerline
        dists = (self.centerline[0, :] - x)**2 + (self.centerline[1, :] - y)**2
        idx = np.argmin(dists)
        
        s = self.s_c_dense[idx]
        
        # Lateral deviation (dot product with normal vector)
        dx = x - self.centerline[0, idx]
        dy = y - self.centerline[1, idx]
        nx = -np.sin(self.psi_c_dense[idx])
        ny = np.cos(self.psi_c_dense[idx])
        n = dx * nx + dy * ny
        
        # Heading error
        psi = theta - self.psi_c_dense[idx]
        psi = (psi + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]
        
        return np.array([s, n, v, psi])

    def opp_odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = get_yaw_from_quaternion(msg.pose.pose.orientation)
        v = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.opp_state = self.global_to_frenet(x, y, theta, v)

    def ego_odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = get_yaw_from_quaternion(msg.pose.pose.orientation)
        v = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        self.ego_state = self.global_to_frenet(x, y, theta, v)

    # --- PREDICTION HORIZON ---
    def predict_and_publish(self):
        if self.centerline is None: return
        self.get_traj()
        self.publish_trajectories()

    def get_traj(self):
        N = 40      # Horizon steps
        dt = 0.02   # Time step
        
        # Reset lists on each call to prevent infinite memory growth
        self.contn_traj = [self.opp_state]
        self.yield_traj = [self.opp_state]
        self.block_traj = [self.opp_state]
        
        for i in range(N):
            # Continue
            xi_c = self.contn_traj[-1]
            self.contn_traj.append(self.step(xi_c, self.continue_line(xi_c), dt))
            
            # Yield
            xi_y = self.yield_traj[-1]
            self.yield_traj.append(self.step(xi_y, self.yield_line(xi_y), dt))
            
            # Block
            xi_b = self.block_traj[-1]
            self.block_traj.append(self.step(xi_b, self.block_line(xi_b), dt))

    # --- BEHAVIOR MODELS ---
    def continue_line(self, state):
        s_curr = state[0]
        # Make sure to wrap 's' around track length if it exceeds max distance
        max_s = self.s_c_dense[-1]
        s_curr = s_curr % max_s 
        
        n_target = np.interp(s_curr, self.s_c_dense, self.raceline_n)
        v_target = np.interp(s_curr, self.s_c_dense, self.raceline_v)
        return np.array([n_target, v_target])
    
    def yield_line(self, state):
        s_curr, n_curr = state[0], state[1]
        _, ne, _, _ = self.ego_state # Assume ego holds lateral position
        
        max_s = self.s_c_dense[-1]
        s_curr = s_curr % max_s 

        n_target = np.sign(n_curr - ne + 1e-3) * (self.nmax - self.w_margin)
        v_target = np.interp(s_curr, self.s_c_dense, self.raceline_v) * 0.5 
        return np.array([n_target, v_target])
    
    def block_line(self, state):
        s_curr, n_curr = state[0], state[1]
        se, ne, ve, _ = self.ego_state
        
        max_s = self.s_c_dense[-1]
        s_curr = s_curr % max_s 

        n_target = np.clip(ne, -self.nmax + self.w_margin, self.nmax - self.w_margin)
        
        # Handle wraparound for distance check
        dist = (s_curr - se) % max_s
        if dist > max_s / 2: dist -= max_s # If ego is ahead, distance is negative
        
        if dist > 0 and dist < 1.0: # Ego is close behind
            v_target = ve
        else:
            v_target = np.interp(s_curr, self.s_c_dense, self.raceline_v)
            
        return np.array([n_target, v_target])

    # --- KINEMATICS ---
    def controller(self, state, target):
        nt, vt = target
        s, n, v, psi = state
        
        Kpv = 1.5
        a = Kpv * (vt - v)
        
        Kpn = 0.8
        Kdn = 0.5
        n_dot = v * np.sin(psi)
        
        # Positive steering moves car to the left (increasing n)
        steer = Kpn * (nt - n) - Kdn * n_dot 
        steer = min(max(steer, -np.pi/2), np.pi/2)
        return np.array([a, steer])
    
    def predict(self, state, u):
        s, n, v, psi = state
        a, steer = u
        
        s_mod = s % self.s_c_dense[-1]
        kappa = np.interp(s_mod, self.s_c_dense, self.kap_c_dense)
        
        denominator = 1 - n * kappa
        if denominator == 0: denominator = 0.001 # Prevent division by zero
        
        s_dot = v * np.cos(psi) / denominator
        n_dot = v * np.sin(psi)
        v_dot = a
        psi_dot = (v / self.Lwb) * np.tan(steer) - kappa * s_dot
        
        return np.array([s_dot, n_dot, v_dot, psi_dot])
          
    def step(self, state, target, dt):
        u = self.controller(state, target)
        f = self.predict(state, u)
        return state + f * dt

    # --- VISUALIZATION ---
    def frenet_to_global(self, s, n):
        s_mod = s % self.s_c_dense[-1]
        idx = np.searchsorted(self.s_c_dense, s_mod)
        idx = np.clip(idx, 0, len(self.s_c_dense)-1)
        
        xc = self.centerline[0, idx]
        yc = self.centerline[1, idx]
        psi_c = self.psi_c_dense[idx]
        
        # Convert back using normal vector
        x = xc - n * np.sin(psi_c)
        y = yc + n * np.cos(psi_c)
        return x, y

    def publish_trajectories(self):
        marker_array = MarkerArray()
        
        trajs = [
            (self.contn_traj, 0, [0.0, 1.0, 0.0]),    # Green
            (self.yield_traj, 1, [0.0, 0.0, 1.0]),    # Blue
            (self.block_traj, 2, [1.0, 0.0, 0.0])     # Red
        ]
        
        for traj, i, color in trajs:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'prediction'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05 # line width
            
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            for state in traj:
                x, y = self.frenet_to_global(state[0], state[1])
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                marker.points.append(p)
                
            marker_array.markers.append(marker)
            
        self.traj_pub.publish(marker_array)

    def publish_centerline(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "centerline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.1
        
        marker.color.a = 1.0 
        marker.color.b = 0.5
        marker.color.g = 0.0
        marker.color.r = 0.0
        
        # waypoints as points
        for wp in self.centerline.T:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = -0.1
            marker.points.append(p)
            
        self.centerline_pub.publish(marker)
    
    def publish_raceline(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "raceline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.1
        
        marker.color.a = 1.0 
        marker.color.b = 0.0
        marker.color.g = 0.0
        marker.color.r = 0.5
        
        # waypoints as points
        for wp in self.raceline.T:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = -0.1
            marker.points.append(p)
            
        self.raceline_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = OpponentPrediction()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()