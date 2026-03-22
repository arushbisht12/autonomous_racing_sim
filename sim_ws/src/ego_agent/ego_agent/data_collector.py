#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from rclpy.serialization import serialize_message
from nav_msgs.msg import Odometry

from message_filters import Subscriber, ApproximateTimeSynchronizer
import rosbag2_py


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        self.writer = rosbag2_py.SequentialWriter()

        storage_options = rosbag2_py.StorageOptions(
            uri='nice_odom_bag',
            storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)
        
        ego_topic_info = rosbag2_py.TopicMetadata(
            name='/ego_racecar/odom', 
            type='nav_msgs/msg/Odometry', 
            serialization_format='cdr')
        self.writer.create_topic(ego_topic_info)
        
        opp_topic_info = rosbag2_py.TopicMetadata(
            name='/opp_racecar/odom', 
            type='nav_msgs/msg/Odometry', 
            serialization_format='cdr')
        self.writer.create_topic(opp_topic_info)
        
        
        self.ego_odom_sub = Subscriber(self, Odometry, '/ego_racecar/odom')
        self.opp_odom_sub = Subscriber(self, Odometry, '/opp_racecar/odom')
        
        
        self.time_sync = ApproximateTimeSynchronizer(
            [self.ego_odom_sub, self.opp_odom_sub], 
            queue_size=10, 
            slop=0.002)
        self.time_sync.registerCallback(self.sync_callback)
        
        self.get_logger().info("Data Collector Node started. Waiting for synchronized messages...")
        
    def sync_callback(self, ego_msg, opp_msg):
        # Grab a single timestamp for the bag entry
        timestamp = self.get_clock().now().nanoseconds
        
        # Write both serialized messages to the bag
        self.writer.write('/ego_racecar/odom', serialize_message(ego_msg), timestamp)
        self.writer.write('/opp_racecar/odom', serialize_message(opp_msg), timestamp)

    def destroy_node(self):
        # Explicitly destroy the writer to ensure the bag closes and flushes properly
        del self.writer
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    data_collector = DataCollector()
    
    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        pass
    finally:
        data_collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()