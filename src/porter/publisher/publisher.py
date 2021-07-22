from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import rospy
from std_msgs.msg import Header 


class Publisher:
    def __init__(self):
        # Initiate ROS variables
        rospy.init_node("sensor_oakd")
        self.pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)
        self.rate = rospy.Rate(10)
        self.points = []
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        # PointField('rgb', 12, PointField.UINT32, 1),
                        #PointField('rgba', 12, PointField.UINT32, 1),
                    ]
        self.header = Header()
        self.header.frame_id = "oakd_frame"
    
    def publish(self, points):
        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.pub.publish(pc2)

