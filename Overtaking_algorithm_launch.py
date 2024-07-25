import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import numpy as np
import pandas as pd
from std_msgs.msg import Float64
from planning import overtake_traj_planner
from utils import racing_env, base
from tf.transformations import euler_from_quaternion
import time
from racing import offboard
from car_racing.msg import VehicleState
from pathos.multiprocessing import ProcessingPool as Pool
import math
from cvxopt.solvers import qp
from control import control, lmpc_helper
from cvxopt import spmatrix, matrix, solvers

class Overtaking_algorithm():
    def __init__(self):
        rospy.init_node("Overtaking_algorithm")
        self.target_pose_sub=rospy.Subscriber('/target_pose', Pose, self.target_pose_callback)
        self.target_velocity_sub=rospy.Subscriber('/target_velocity',Float64, self.target_velocity_callback)
        self.current_pose_sub=rospy.Subscriber('/current_pose',PoseStamped, self.current_pose_callback)
        self.current_velocity=rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_callback)
        self.target_angular=rospy.Subscriber('/target_angular',Float64, self.target_angular_callback)
        file_path = '~/car-racing/local_coordinates_tags.csv' 
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
        self.timestep=0.1
        self.iter=0
        self.p=Pool(4)
        self.lmpc_param=10
        self.u_ss=None
        self.ss_xcurv=None
        self.lin_input=None
        self.lin_points=None
        self.time_ss=np.ones(1).astype(int)
        timestep=0.1
        X_DIM=6
        U_DIM=2
        ego = offboard.DynamicBicycleModel(name="ego", param=base.CarParam(edgecolor="black"), system_param = base.SystemParam())
        ego.set_timestep(timestep)
    # run the pid controller for the first lap to collect data
        pid_controller = offboard.PIDTracking(vt=0.7, eyt=0.0)
        pid_controller.set_timestep(timestep)
        ego.set_ctrl_policy(pid_controller)
        pid_controller.set_track(track)
        ego.set_state_curvilinear(np.zeros((X_DIM,)))
        ego.set_state_global(np.zeros((X_DIM,)))
        ego.start_logging()
        ego.set_track(track)
    # run mpc-lti controller for the second lap to collect data
        mpc_lti_param = base.MPCTrackingParam(vt=0.7, eyt=0.0)
        mpc_lti_controller = offboard.MPCTracking(mpc_lti_param, ego.system_param)
        mpc_lti_controller.set_timestep(timestep)
        mpc_lti_controller.set_track(track)
        self.vehicles_interest={"ego":VehicleState(), "vehicle_1":VehicleState()}
        rate = rospy.Rate(1000)
        self.waypoint=[]
        track_spec = np.genfromtxt(file_path, delimiter=',', names=True)
        track = racing_env.ClosedTrack(track_spec, track_width=0.8)
        self.set_track(track)
        for i in lat_list:
            cx.append(i[0])
        for j in lon_list:
            cy.append(j[0])
        for k in range(len(cx)):
            self.waypoint.append([cx[k],cy[k]])
        self.matrix_Atv, self.matrix_Btv, self.matrix_Ctv, _ = self.estimate_ABC()
        while not rospy.is_shutdown():
            rate.sleep()
            (overtake_traj_xcurv,
                    overtake_traj_xglob,
                    direction_flag,
                    sorted_vehicles,
                    bezier_xglob,
                    solve_time,
                    all_bezier_xglob,
                    all_traj_xglob)=overtake_traj_planner.OvertakeTrajPlanner.get_local_traj(
                    self.x,
                    self.time,
                    self.vehicles_interest,
                    self.matrix_Atv,
                    self.matrix_Btv,
                    self.matrix_Ctv,
                    self.old_ey,
                    self.old_direction_flag,)
            self.old_ey = overtake_traj_xcurv[-1, 5]
            self.old_direction_flag = direction_flag
            ego.ctrl_policy.add_trajectory(
                            ego,
                            2,
                        )
            print(overtake_traj_xcurv)
    def add_trajectory(self, ego, lap_number):

        iter = self.iter
        end_iter = int(round((ego.times[lap_number][-1] - ego.times[lap_number][0]) / ego.timestep))
        times = np.stack(ego.times[lap_number], axis=0)
        self.time_ss[iter] = end_iter
        xcurvs = np.stack(ego.xcurvs[lap_number], axis=0)
        self.ss_xcurv[0 : (end_iter + 1), :, iter] = xcurvs[0 : (end_iter + 1), :]
        xglobs = np.stack(ego.xglobs[lap_number], axis=0)
        self.ss_glob[0 : (end_iter + 1), :, iter] = xglobs[0 : (end_iter + 1), :]
        inputs = np.stack(ego.inputs[lap_number], axis=0)
        self.u_ss[0:end_iter, :, iter] = inputs[0:end_iter, :]
        self.Qfun[0 : (end_iter + 1), iter] = lmpc_helper.compute_cost(
            xcurvs[0 : (end_iter + 1), :],
            inputs[0:(end_iter), :],
            self.lap_length,
        )
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, iter] == 0:
                self.Qfun[i, iter] = self.Qfun[i - 1, iter] - 1
        if self.iter == 0:
            self.lin_points = self.ss_xcurv[1 : self.lmpc_param.num_horizon + 2, :, iter]
            self.lin_input = self.u_ss[1 : self.lmpc_param.num_horizon + 1, :, iter]
        self.iter = self.iter + 1
        self.time_in_iter = 0
            
    def estimate_ABC(self):
        lin_points = self.lin_points
        lin_input = self.lin_input
        num_horizon = self.lmpc_param
        ss_xcurv = self.ss_xcurv
        u_ss = self.u_ss
        time_ss = self.time_ss
        point_and_tangent = self.point_and_tangent
        timestep = self.timestep
        iter = self.iter
        p = self.p
        Atv = []
        Btv = []
        Ctv = []
        index_used_list = []
        lap_used_for_linearization = 2
        used_iter = range(iter - lap_used_for_linearization, iter)
        max_num_point = 40
        for i in range(0, num_horizon):
            (Ai, Bi, Ci, index_selected,) = lmpc_helper.regression_and_linearization(
                lin_points,
                lin_input,
                used_iter,
                ss_xcurv,
                u_ss,
                time_ss,
                max_num_point,
                qp,
                matrix,
                point_and_tangent,
                timestep,
                i,
            )
            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)
            index_used_list.append(index_selected)
        return Atv, Btv, Ctv, index_used_list
    
    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width
    
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
        path_yaw = self.get_path_yaw(self.waypoint,self.target_x,self.target_y)
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
        path_yaw = self.get_path_yaw(self.waypoint,self.current_x,self.current_y)
        epsi = yaw - path_yaw
        self.x[3] = epsi 
        self.vehicles_interest['ego'].xglob[3]=yaw
        self.vehicles_interest['ego'].xglob[4]=self.current_x
        self.vehicles_interest['ego'].xglob[5]=self.current_y
        self.vehicles_interest['ego'].xcurv[3]=self.x[3]
        self.vehicles_interest['ego'].xcurv[4]=self.x[4]
        self.vehicles_interest['ego'].xcurv[5]=self.x[5]
        
    
    def get_path_yaw(self, waypoints,x,y):
        closest_wp_index = self.get_closest_waypoint(self.current_x, x,y)
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