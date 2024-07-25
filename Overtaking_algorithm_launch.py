import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import numpy as np
import pandas as pd
from std_msgs.msg import Float64
from planning import overtake_traj_planner
from tf.transformations import euler_from_quaternion
import time
from msg import VehicleState
import math

class Overtaking_algorithm():
    def __init__(self):
        rospy.init_node("Overtaking_algorithm")
        self.target_pose_sub=rospy.Subscriber('/target_pose', Pose, self.target_pose_callback)
        self.target_velocity_sub=rospy.Subscriber('/target_velocity',Float64, self.target_velocity_callback)
        self.current_pose_sub=rospy.Subscriber('/current_pose',PoseStamped, self.current_pose_callback)
        self.current_velocity=rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_callback)
        self.target_angular=rospy.Subscriber('/target_angular',Float64, self.target_angular_callback)
        file_path = '~/Desktop/car-racing/local_coordinates_tags.csv' 
    #  target course
        df = pd.read_csv(file_path)
        lat_list = list(zip(df['local_x']))
        lon_list=list(zip(df['local_y']))
        cx=[]
        cy=[]
        self.old_ey=None
        self.old_direction_flag=None
        self.time=0.0
        self.x=[0]*6
        self.vehicles_interest={"ego":VehicleState(), "vehicle_1":VehicleState()}
        rate = rospy.Rate(1000)
        self.waypoint=[]
        for i in lat_list:
            cx.append(i[0])
        for j in lon_list:
            cy.append(j[0])
        for k in range(len(cx)):
            self.waypoint.append([cx[k],cy[k]])
        while not rospy.is_shutdown():
            rate.sleep()
            # (overtake_traj_xcurv,
            #         overtake_traj_xglob,
            #         direction_flag,
            #         sorted_vehicles,
            #         bezier_xglob,
            #         solve_time,
            #         all_bezier_xglob,
            #         all_traj_xglob)=overtake_traj_planner.OvertakeTrajPlanner.get_local_traj(
            #         self.x,
            #         self.time,
            #         self.vehicles_interest,
            #         self.matrix_Atv,
            #         self.matrix_Btv,
            #         self.matrix_Ctv,
            #         self.old_ey,
            #         self.old_direction_flag,)
            # self.old_ey = overtake_traj_xcurv[-1, 5]
            # self.old_direction_flag = direction_flag
    def target_angular_callback(self,data):
        self.target_angular_z=data
        self.vehicles_interest['target_1'].xglob[2]=self.target_angular_z
        self.vehicles_interest['target_1'].xcurv[2]=self.target_angular_z
        
            
    def current_velocity_callback(self,data):
        self.current_velocity_x=data.twist.linear.x
        self.current_velocity_y=data.twist.linear.y
        self.current_angular_z=data.twist.angular.z
        self.vehicles_interest['ego'].xglob[0]=self.current_velocity_x
        self.vehicles_interest['ego'].xglob[1]=self.current_velocity_y
        self.vehicles_interest['ego'].xglob[2]=self.current_angular_z
        self.x[0]=self.current_velocity_x
        self.x[1]=self.current_velocity_y
        self.x[2]=self.current_angular_z
        self.vehicles_interest['ego'].xcurv[0]=self.x[0]
        self.vehicles_interest['ego'].xcurv[1]=self.x[1]
        self.vehicles_interest['ego'].xcurv[2]=self.x[2]
        
        
        self.current_velocity=(self.current_velocity_x**2+self.current_velocity_y**2)**(1/2)
        
    def target_velocity_callback(self, data):
        self.target_velocity=data 
        self.target_velocity_x=data*math.cos(self.target_yaw)
        self.target_velocity_y=data*math.sin(self.target_yaw)
        self.vehicles_interest['target_1'].xcurv[0]=self.target_velocity_x
        self.vehicles_interest['target_1'].xcurv[1]=self.target_velocity_y
        self.vehicles_interest['target_1'].xglob[0]=self.current_velocity_x
        self.vehicles_interest['target_1'].xglob[1]=self.current_velocity_y

    def target_pose_callback(self, data):
        self.target_x=data.position.x
        self.target_y=data.position.y
        self.target_z=data.position.z
        self.target_orientation_x=data.orientation.x 
        self.target_orientation_y=data.orientation.y 
        self.target_orientation_z=data.orientation.z 
        self.target_orientation_w=data.orientation.w 
        self.target_s,self.target_d = self.get_frenet(self.target_x, self.target_y,self.waypoint)
        self.target_yaw = self.get_yaw_from_orientation(self.target_orientation_x, self.target_orientation_y, self.target_orientation_z, self.target_orientation_w)
        path_yaw = self.get_path_yaw(self.waypoint)
        epsi = self.target_yaw - path_yaw
        self.vehicles_interest['target_1'].xglob[3]=self.target_yaw
        self.vehicles_interest['target_1'].xglob[4]=self.target_x
        self.vehicles_interest['target_1'].xglob[5]=self.target_y
        self.vehicles_interest['target_1'].xcurv[3]=epsi
        self.vehicles_interest['target_1'].xcurv[4]=self.target_s
        self.vehicles_interest['target_1'].xcurv[5]=self.target_d
        
        
    def current_pose_callback(self, data):
        self.current_x=data.pose.position.x
        self.current_y=data.pose.position.y
        self.current_z=data.pose.position.z
        self.current_orientation_x=data.pose.orientation.x 
        self.current_orientation_y=data.pose.orientation.y 
        self.current_orientation_z=data.pose.orientation.z 
        self.current_orientation_w=data.pose.orientation.w
        self.current_s,self.current_d = self.get_frenet(self.current_x, self.current_y,self.waypoint)
        self.x[4]=self.current_s
        self.x[5]=self.current_d
        
        yaw = self.get_yaw_from_orientation(self.current_orientation_x, self.current_orientation_y, self.current_orientation_z, self.current_orientation_w)
        path_yaw = self.get_path_yaw(self.waypoint)
        epsi = yaw - path_yaw
        self.x[3] = epsi 
        self.vehicles_interest['ego'].xglob[3]=yaw
        self.vehicles_interest['ego'].xglob[4]=self.current_x
        self.vehicles_interest['ego'].xglob[5]=self.current_y
        self.vehicles_interest['ego'].xcurv[3]=self.x[3]
        self.vehicles_interest['ego'].xcurv[4]=self.x[4]
        self.vehicles_interest['ego'].xcurv[5]=self.x[5]
        
    
    def get_path_yaw(self, waypoints):
        closest_wp_index = self.get_closest_waypoint(self.current_x, self.current_y, waypoints)
        next_wp_index = (closest_wp_index + 1) % len(waypoints)
        dx = waypoints[next_wp_index][0] - waypoints[closest_wp_index][0]
        dy = waypoints[next_wp_index][1] - waypoints[closest_wp_index][1]
        return np.arctan2(dy, dx)
        

    def calculate_distance(self,point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_closest_waypoint(self,x, y, waypoints):
        closest_len = float('inf')
        closest_index = 0
        for i in range(len(waypoints)):
            dist = self.calculate_distance((x, y), waypoints[i])
            if dist < closest_len:
                closest_len = dist
                closest_index = i
        return closest_index
    
    def get_yaw_from_orientation(self, x, y, z, w):
        euler = euler_from_quaternion([x, y, z, w])
        return euler[2] 

    def get_frenet(self,x, y, waypoints):
        closest_wp_index = self.get_closest_waypoint(x, y, waypoints)
        next_wp_index = (closest_wp_index + 1) % len(waypoints)

        map_x = waypoints[closest_wp_index][0]
        map_y = waypoints[closest_wp_index][1]
        next_map_x = waypoints[next_wp_index][0]
        next_map_y = waypoints[next_wp_index][1]

        n_x = next_map_x - map_x
        n_y = next_map_y - map_y
        x_x = x - map_x
        x_y = y - map_y

    # Calculate frenet d coordinate
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x**2 + n_y**2)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y
        frenet_d = self.calculate_distance((x_x, x_y), (proj_x, proj_y))

    # Calculate frenet s coordinate
        frenet_s = 0
        for i in range(closest_wp_index):
            frenet_s += self.calculate_distance(waypoints[i], waypoints[i + 1])
            frenet_s += self.calculate_distance((0, 0), (proj_x, proj_y))

    # Ensure d is positive if the point is to the left of the reference path
        ref_point = [map_x + proj_x, map_y + proj_y]
        if np.cross(np.array([x - map_x, y - map_y]), np.array([next_map_x - map_x, next_map_y - map_y])) > 0:
            frenet_d = -frenet_d

        return frenet_s, frenet_d




if __name__=='__main__':
    try:
        Overtaking_algorithm()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()