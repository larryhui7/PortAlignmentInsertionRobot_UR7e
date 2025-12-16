import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import serial
import serial.tools.list_ports
import time


class LoadcellNode(Node):
    """
    ROS2 node that interfaces with ESP32 loadcell via serial port.
    Reads loadcell data in grams and publishes it to a ROS2 topic.
    """
    
    def __init__(self):
        super().__init__('loadcell_node')
        
        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('timeout', 1.0)
        self.declare_parameter('publish_rate', 10.0)  # Hz
        
        # Get parameters
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        timeout = self.get_parameter('timeout').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Initialize serial connection
        self.serial_conn = None
        try:
            self.serial_conn = serial.Serial(
                port=serial_port,
                baudrate=baud_rate,
                timeout=timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.get_logger().info(f'Connected to serial port: {serial_port} at {baud_rate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {serial_port}: {e}')
            self.get_logger().info('Available serial ports:')
            for port in serial.tools.list_ports.comports():
                self.get_logger().info(f'  - {port.device}: {port.description}')
            raise
        
        # Publisher for loadcell data
        self.publisher = self.create_publisher(Float32, '/loadcell/weight', 10)
        
        # Timer for reading and publishing data
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.read_and_publish)
        
        self.get_logger().info('Loadcell node started')
        self.get_logger().info(f'Publishing to /loadcell/weight at {publish_rate} Hz')
    
    def read_and_publish(self):
        """Read data from serial port and publish to ROS2 topic."""
        if self.serial_conn is None or not self.serial_conn.is_open:
            self.get_logger().warn('Serial port is not open')
            return
        
        try:
            # Read all available lines and use the most recent valid weight reading
            latest_weight = None
            while self.serial_conn.in_waiting > 0:
                try:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    if line:
                        # Try to parse as float (weight value)
                        # Debug lines from ESP32 will fail to parse and be ignored
                        try:
                            weight = float(line)
                            latest_weight = weight
                        except ValueError:
                            # Not a numeric line, likely debug output - ignore
                            pass
                except UnicodeDecodeError:
                    # Skip lines that can't be decoded
                    continue
            
            # Publish the most recent valid weight reading
            if latest_weight is not None:
                msg = Float32()
                msg.data = latest_weight
                self.publisher.publish(msg)
                self.get_logger().debug(f'Published weight: {latest_weight:.2f} g')
                
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')
    
    def destroy_node(self):
        """Clean up serial connection on shutdown."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LoadcellNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

