#!/usr/bin/env python3

import rospy
import std_msgs.msg
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

def talker():
    pub = rospy.Publisher('pybullet_poses', PoseArray, queue_size=1)
    rospy.init_node('pybullet_talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    while not rospy.is_shutdown():
        new_x = float(input("new x"))
        posearray = PoseArray()
        posearray.header.stamp = rospy.Time.now()  # timestamp of creation of the message
        posearray.header.frame_id = "panda_link0"  # frame id in which the array is published

        # start pose
        pose_start = Pose()

        pose_start.orientation.x = 1.0
        pose_start.orientation.y = 0.0
        pose_start.orientation.z = 0.0
        pose_start.orientation.w = 0.0007

        pose_start.position.x = 0.2
        pose_start.position.y = 0.0
        pose_start.position.z = 0.5

        # goal pose
        pose_goal = Pose()

        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        pose_goal.orientation.z = 0.0
        pose_goal.orientation.w = 0.0007

        pose_goal.position.x = new_x
        pose_goal.position.y = 0.0
        pose_goal.position.z = 0.8

        posearray.poses.append(pose_start)
        posearray.poses.append(pose_goal)

        rospy.loginfo(posearray)
        pub.publish(posearray)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

