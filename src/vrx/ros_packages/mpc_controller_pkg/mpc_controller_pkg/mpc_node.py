# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float64

# class ThrusterPublisher(Node):
#     def __init__(self):
#         super().__init__('thruster_publisher')
#         self.publisher_ = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)

#         self.timer_ = self.create_timer(1.0, self.publish_thrust)
#         self.thrust_values = [150.0, 0.0]
#         self.current_index = 0

#     def publish_thrust(self):
#         msg = Float64()
#         msg.data = self.thrust_values[self.current_index]
#         self.publisher_.publish(msg)
#         self.get_logger().info(f'Publishing: {msg.data}')
#         self.current_index = (self.current_index + 1) % len(self.thrust_values)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ThrusterPublisher()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()





##########################

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray
# import numpy as np
# import cvxpy as cp

# class MPCNode(Node):
#     def __init__(self):
#         super().__init__('mpc_node')
#         self.subscription = self.create_subscription(
#             Float32MultiArray,
#             'new_state_topic',
#             self.listener_callback,
#             10)
#         self.publisher_ = self.create_publisher(Float32MultiArray, 'new_control_topic', 10)
#         self.get_logger().info('MPCNode has been started.')

#     def dynamics(self, x, u):
#         dx = np.zeros(12)
#         dx[0] = - (4 * x[0]) / 15 - x[1] * (x[1] / 1970 - x[5] / 197) - x[2] * (x[2] / 4470 + x[4] / 447)
#         dx[1] = (30 * x[5]) / 197 - (200 * x[1]) / 197 + (x[2] * x[3]) / 400 + x[0] * (x[1] / 1970 - x[5] / 197)
#         dx[2] = x[0] * (x[2] / 4470 + x[4] / 447) - x[4] / 447 - (x[1] * x[3]) / 400 - (5 * x[2]) / 149
#         dx[3] = x[1] * (x[2] / 298 + x[4] / 4470) - x[4] * (x[1] / 1970 - x[5] / 197) - x[2] * ((2 * x[1]) / 591 - x[5] / 1970) - x[3] / 8 - x[5] * (x[2] / 4470 + x[4] / 447)
#         dx[4] = (x[0] * x[2]) / 300 + (x[3] * x[5]) / 400 + x[3] * (x[1] / 1970 - x[5] / 197) - x[0] * (x[2] / 298 + x[4] / 4470)
#         dx[5] = (20 * x[1]) / 197 - (200 * x[5]) / 197 - (x[0] * x[1]) / 300 - (x[3] * x[4]) / 400 + x[0] * ((2 * x[1]) / 591 - x[5] / 1970) + x[3] * (x[2] / 4470 + x[4] / 447)
#         dx[6] = (x[2] / 298 + x[4] / 4470) * (np.sin(x[9]) * np.sin(x[11]) + np.cos(x[9]) * np.cos(x[11]) * np.sin(x[10])) - ((2 * x[1]) / 591 - x[5] / 1970) * (np.cos(x[9]) * np.sin(x[11]) - np.cos(x[11]) * np.sin(x[9]) * np.sin(x[10])) + (x[0] * np.cos(x[10]) * np.cos(x[11])) / 300
#         dx[7] = ((2 * x[1]) / 591 - x[5] / 1970) * (np.cos(x[9]) * np.cos(x[11]) + np.sin(x[9]) * np.sin(x[10]) * np.sin(x[11])) - (x[2] / 298 + x[4] / 4470) * (np.cos(x[11]) * np.sin(x[9]) - np.cos(x[9]) * np.sin(x[10]) * np.sin(x[11])) + (x[0] * np.cos(x[10]) * np.sin(x[11])) / 300
#         dx[8] = np.cos(x[9]) * np.cos(x[10]) * (x[2] / 298 + x[4] / 4470) - (x[0] * np.sin(x[10])) / 300 + np.cos(x[10]) * np.sin(x[9]) * ((2 * x[1]) / 591 - x[5] / 1970)
#         dx[9] = x[3] / 400 - np.cos(x[9]) * np.tan(x[10]) * (x[1] / 1970 - x[5] / 197) + np.sin(x[9]) * np.tan(x[10]) * (x[2] / 4470 + x[4] / 447)
#         dx[10] = np.cos(x[9]) * (x[2] / 4470 + x[4] / 447) + np.sin(x[9]) * (x[1] / 1970 - x[5] / 197)
#         dx[11] = (np.sin(x[9]) * (x[2] / 4470 + x[4] / 447)) / np.cos(x[10]) - (np.cos(x[9]) * (x[1] / 1970 - x[5] / 197)) / np.cos(x[10])
#         return dx

#     def mpc_control(self, x_init_val, x_ref_val):
#         n_states = 12
#         n_controls = 1
#         N = 20
#         dt = 0.1

#         x = cp.Variable((n_states, N+1))
#         u = cp.Variable((n_controls, N))

#         x_init = cp.Parameter(n_states)
#         x_ref = cp.Parameter((n_states, N+1))

#         x_init.value = x_init_val
#         x_ref.value = x_ref_val

#         Q = np.eye(n_states)
#         R = np.eye(n_controls)

#         cost = 0
#         constraints = []

#         for k in range(N):
#             cost += cp.quad_form(x[:, k] - x_ref[:, k], Q) + cp.quad_form(u[:, k], R)
#             x_next = x[:, k] + dt * self.dynamics(x[:, k].value, u[:, k].value)
#             constraints += [x[:, k+1] == x_next]
#             constraints += [u[:, k] >= -1, u[:, k] <= 1]

#         constraints += [x[:, 0] == x_init]

#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve()

#         if problem.status not in ["optimal", "optimal_inaccurate"]:
#             self.get_logger().error(f'Problem could not be solved. Status: {problem.status}')
#             return None

#         return u.value

#     def listener_callback(self, msg):
#         try:
#             self.get_logger().info('Received message on new_state_topic.')
#             if len(msg.data) < 12:
#                 self.get_logger().error('Received data does not contain enough elements.')
#                 return

#             x_init_val = np.array(msg.data[:12])
#             x_ref_val = np.zeros((12, 21))  # Example reference value, should be updated accordingly

#             self.get_logger().info(f'x_init_val: {x_init_val}')
#             self.get_logger().info(f'x_ref_val: {x_ref_val}')

#             u_optimal = self.mpc_control(x_init_val, x_ref_val)
#             if u_optimal is None:
#                 self.get_logger().error('MPC control failed to find a solution.')
#                 return

#             self.get_logger().info(f'u_optimal: {u_optimal}')
            
#             pwm_values = self.convert_to_pwm(u_optimal.flatten())

#             self.get_logger().info(f'Publishing PWM values: {pwm_values}')
            
#             self.publisher_.publish(Float32MultiArray(data=pwm_values))
#         except Exception as e:
#             self.get_logger().error(f'Error in listener_callback: {e}')

#     def convert_to_pwm(self, u):
#         try:
#             pwm_min = 1100
#             pwm_max = 1900
#             pwm_values = ((u + 1) / 2) * (pwm_max - pwm_min) + pwm_min
#             return pwm_values
#         except Exception as e:
#             self.get_logger().error(f'Error in convert_to_pwm: {e}')
#             return np.array([1500] * len(u))  # Default PWM value in case of error

# def main(args=None):
#     rclpy.init(args=args)
#     mpc_node = MPCNode()
#     rclpy.spin(mpc_node)
#     mpc_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()




# ##################################


# import sys
# import os
# import time

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Pose, PoseArray, Twist
# from sensor_msgs.msg import NavSatFix, Imu
# from std_msgs.msg import Float64, Float32MultiArray
# import casadi as ca
# import numpy as np
# from scipy.spatial.transform import Rotation
# from pyproj import Proj, Transformer
# from qpsolvers import Problem, solve_problem

# class WAMV_NMPC_Controller(Node):

#     def __init__(self):
#         super().__init__('wamv_nmpc_controller')
        
#         # Model parameters
#         self.m11, self.m22, self.m33 = 345.0, 256.0, 944.0
#         self.d11, self.d22, self.d33 = 137.0, 79.0, 934.0
#         self.d11_2, self.d22_2, self.d33_2 = 225.0, 55.0, 1057.0
#         self.d_m, self.L_m = 1.0, 2.3
        
#         # NMPC parameters
#         self.Np = 50  # Prediction horizon
#         self.Nu = 30        
#         self.dt = 0.5  # Time step
#         self.nu = 2
#         self.nx = 6
        
#         # Initialize state and reference
#         self.current_state = np.zeros((self.nx,1))
       
#         # Last known actuator inputs
#         self.last_inputs = np.zeros((self.nu,1))

#         # EKF parameters
#         self.P = np.diag([1000.0, 1000.0, 10.0, 0.0, 0.0, 0.0]) # Initial state covariance
#         self.Q = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.01])  # Process noise covariance
#         self.R_gps = np.diag([1.0, 1.0])  # GPS measurement noise covariance
#         self.R_imu = np.diag([0.01, 0.01])  # IMU measurement noise covariance
        
#         # Control parameters
#         self.Qctrl = np.diag([100, 100, 200, 0.00001, 0.00001, 0.1])
#         self.Rctrl = 2
#         self.thrust_lower_bound = -10  # Adjusted to match the graph's lower thrust bound
#         self.thrust_upper_bound = 16  # Adjusted to match the graph's upper thrust bound

#         self.U = np.zeros((self.nu, self.Nu))
#         self.Xref = np.zeros((self.nx, self.Np+1))
#         self.input_scale = 1
#         self.waypoint = np.array([[-350], [750]])
#         self.waypoints = self.generate_end_trajectory()

#         # Sydney Regatta Centre coordinates (approximate center)
#         self.datum_lat = -33.7285
#         self.datum_lon = 150.6789

#         # Initialize projections
#         self.proj_wgs84 = Proj(proj='latlong', datum='WGS84')
#         self.proj_utm = Proj(proj='utm', zone=56, datum='WGS84', south=True)
#         self.transformer = Transformer.from_proj(self.proj_wgs84, self.proj_utm)

#         # Calculate datum in UTM coordinates
#         self.datum_x, self.datum_y = self.transformer.transform(self.datum_lon, self.datum_lat)
        
#         # Counters for number of gps and imu updates
#         self.gpsUpdates = 0
#         self.imuUpdates = 0
#         self.prev_yaw = None
#         self.yaw_offset = 0.0
#         self.headingOld = 0.0
#         self.currentwaypoint = 0

#         # Timestamp of last update
#         self.last_update_time = self.get_clock().now().nanoseconds / 1e9
        
#         # ROS2 publishers and subscribers
#         self.cmd_L_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
#         self.cmd_R_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
#         self.gps_sub = self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
#         self.imu_sub = self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
#         self.ref_sub = self.create_subscription(PoseArray, '/vrx/wayfinding/waypoints', self.reference_callback, 10)

#         ###
        
#         # Timer for control loop
#         self.create_timer(self.dt, self.control_loop)

#     def generate_end_trajectory(self):
#         waypoints = np.zeros((3, 11))  # 10 waypoints + initial position
#         for i in range(11):
#             waypoints[0, i] = i  # x coordinate (0 to 10 meters)
#             waypoints[1, i] = 0  # y coordinate (straight line along x-axis)
#             waypoints[2, i] = 0  # orientation (yaw angle)
#         return waypoints
    
#     def gps_callback(self, msg):
#         current_time = self.get_clock().now().nanoseconds / 1e9
#         dt = current_time - self.last_update_time
#         self.last_update_time = current_time

#         # Convert GPS to local x-y coordinates
#         x, y = self.gps_to_local_xy(msg.longitude, msg.latitude)
        
#         self.update_ekf(dt, gps_measurement=np.array([x, y]))

#         if self.gpsUpdates < 300:
#             self.gpsUpdates += 1

#     def imu_callback(self, msg):
#         current_time = self.get_clock().now().nanoseconds / 1e9
#         dt = current_time - self.last_update_time
#         self.last_update_time = current_time

#         quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
#         euler = Rotation.from_quat(quat).as_euler('xyz')
#         yaw = euler[2]
#         angular_velocity = msg.angular_velocity.z
        
#         if self.prev_yaw is not None:
#             # Check for wrap-around
#             diff = yaw - self.prev_yaw
#             if diff > np.pi:
#                 self.yaw_offset -= 2.0 * np.pi
#             elif diff < -np.pi:
#                 self.yaw_offset += 2.0 * np.pi

#         unwrapped_yaw = yaw + self.yaw_offset
#         self.prev_yaw = yaw

#         self.update_ekf(dt, imu_measurement=np.array([unwrapped_yaw, angular_velocity]))

#         if self.imuUpdates < 300:
#             self.imuUpdates += 1

#     def update_ekf(self, dt, gps_measurement=None, imu_measurement=None):
#         # Prediction step
#         self.current_state, F = self.state_transition_A(self.current_state, dt, 0 * self.last_inputs)
#         self.P = F @ self.P @ F.T + self.Q * dt  # Scale process noise with dt

#         # GPS Update
#         if gps_measurement is not None:
#             H_gps = np.array([[1, 0, 0, 0, 0, 0],
#                               [0, 1, 0, 0, 0, 0]])
#             self.kalman_update(H_gps, gps_measurement, self.R_gps)

#         # IMU Update
#         if imu_measurement is not None:
#             H_imu = np.array([[0, 0, 1, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 1]])
#             self.kalman_update(H_imu, imu_measurement, self.R_imu)

#     def kalman_update(self, H, measurement, R):
#         y = measurement.reshape(-1, 1) - H @ self.current_state
#         S = H @ self.P @ H.T + R
#         K = self.P @ H.T @ np.linalg.inv(S)
#         self.current_state = self.current_state + K @ y
#         self.P = (np.eye(6) - K @ H) @ self.P

#     def reference_callback(self, msg):
#         # Extract yaw from quaternion for the reference pose
#         self.waypoints = np.zeros((3,len(msg.poses)))
#         for i in range(len(msg.poses)):
#             quat = [msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w]
#             euler = Rotation.from_quat(quat).as_euler('xyz')
#             x, y = self.gps_to_local_xy(msg.poses[i].position.y, msg.poses[i].position.x)
#             self.waypoints[:3,i] = [x, y, euler[2]]  # yaw is euler[2]

#     def listener_callback(self, msg):
#         if len(msg.data) < 12:
#             self.get_logger().error('Received data does not contain enough elements.')
#             return

#         x_init_val = np.array(msg.data[:12]).reshape((12, 1))
#         x_ref_val = np.zeros((12, 21))
#         u_optimal = self.mpc_control(x_init_val, x_ref_val)
#         if u_optimal is None:
#             self.get_logger().error('MPC control failed to find a solution.')
#             return

#         self.publish_control_inputs(u_optimal)

#     def control_loop(self):
#         print(self.current_state)
#         if self.waypoints is not None and self.gpsUpdates > 100 and self.imuUpdates > 100:
#             # Apply previous control action and compute predicted state at k+1
#             self.last_inputs = self.U[:, 0:1]
#             x = self.state_transition_xonly(self.current_state, self.last_inputs, self.dt)

#             print(self.last_inputs)

#             # Ensure control inputs are within bounds
#             self.last_inputs = np.clip(self.last_inputs, self.thrust_lower_bound, self.thrust_upper_bound)

#             # Publish control commands
#             self.publish_control_inputs(self.last_inputs)

#             # Start timer
#             start_time = time.time()

#             # Solve NMPC problem
#             U = np.zeros((self.nu, self.Nu))
#             U[:, :-1] = self.U[:, 1:]
#             U[:, -1] = self.U[:, -1]

#             # Check if we are close to waypoint and move to the next
#             if np.linalg.norm(self.waypoints[:, [self.currentwaypoint]] - self.current_state[:3, [0]]) < 0.5:
#                 self.currentwaypoint += 1
#                 if self.currentwaypoint >= self.waypoints.shape[1]:
#                     self.currentwaypoint = self.waypoints.shape[1] - 1

#             # Pick waypoint from the list
#             self.waypoint = self.waypoints[:, [self.currentwaypoint]]

#             # Determine trajectory
#             self.computeTrajectory(x)

#             # Run main optimisation loop
#             for i in range(10):
#                 # Compute error and jacobian
#                 e, J = self.error_and_Jacobian(x, U, self.Xref)

#                 # Compute search direction
#                 H = J.T @ J
#                 f = J.T @ e
#                 lb = -U.reshape(-1, 1, order='f') + self.thrust_lower_bound * np.ones((self.Nu * self.nu, 1))
#                 ub = -U.reshape(-1, 1, order='f') + self.thrust_upper_bound * np.ones((self.Nu * self.nu, 1))
#                 prob = Problem(H, f, None, None, None, None, lb, ub)
#                 sol = solve_problem(prob, solver='proxqp')

#                 p = sol.x
#                 lam = sol.z_box

#                 # Compute the Newton Decrement and return if less than some threshold
#                 if np.linalg.norm(f + lam.reshape(-1, 1, order='f')) < 1e-2:
#                     break

#                 # Compute cost (sum of squared errors)
#                 co = e.T @ e

#                 # Use backtracking line search
#                 alp = 1.0
#                 for j in range(30):
#                     # Try out new input sequence
#                     Un = U + alp * p.reshape(self.nu, -1, order='f')
#                     e = self.error_only(x, Un, self.Xref)
#                     cn = e.T @ e
#                     # If we have reduced the cost then accept the new input sequence and return
#                     if np.isfinite(cn) and cn < co:
#                         U = Un
#                         break
#                     # Otherwise halve the step length
#                     alp = alp / 2.0

#             # Record the optimal input sequence
#             self.U = U

#             # End timer
#             end_time = time.time()

#             # Calculate elapsed time
#             elapsed_time = end_time - start_time
#             print("Elapsed time: ", elapsed_time)

#     def publish_control_inputs(self, u_optimal):
#         pwm_values = self.convert_to_pwm(u_optimal.flatten())

#         # Clip the control inputs to the allowed range
#         pwm_values = np.clip(pwm_values, 1100, 1900)

#         msg_left = Float64()
#         msg_right = Float64()
#         msg_left.data = float(pwm_values[0])
#         msg_right.data = float(pwm_values[1])
#         self.cmd_L_pub.publish(msg_left)
#         self.cmd_R_pub.publish(msg_right)

#     def convert_to_pwm(self, thrust_values):
#         # Convert thrust to PWM using linear interpolation
#         min_pwm, max_pwm = 1100, 1900
#         min_thrust, max_thrust = -10, 16
        
#         pwm_values = min_pwm + (thrust_values - min_thrust) * (max_pwm - min_pwm) / (max_thrust - min_thrust)
#         return pwm_values

#     def computeTrajectory(self, x):
#         # Compute unit direction vector from x to next waypoint
#         uv = self.waypoint[0:2,:] - x[0:2,:]
#         distance_to_waypoint = np.linalg.norm(uv)
#         uv = uv / distance_to_waypoint

#         maxSpeed = 2.0 # m/s
#         maxDistance = self.Np * self.dt * maxSpeed
#         if maxDistance < distance_to_waypoint:
#             wpX = x[0,0] + maxDistance* uv[0,0]
#             wpY = x[1,0] + maxDistance* uv[1,0]
#         else:
#             wpX = self.waypoint[0,0]
#             wpY = self.waypoint[1,0]

#         self.Xref[0,:] = self.waypoint[0,0] # np.linspace(x[0,0], wpX, self.Np+1)
#         self.Xref[1,:] = self.waypoint[1,0] # np.linspace(x[1,0], wpY, self.Np+1)
        
#         maxAngSpeed = 0.5 # rads/s
#         if distance_to_waypoint < 10.0:
#             heading = self.waypoint[2,0] #self.headingOld
#         else:
#             heading = np.arctan2(uv[1,0],uv[0,0])
#         self.Xref[2,:] = self.waypoint[2,0] # np.linspace(x[2,0], heading, self.Np+1)
#         self.headingOld = heading

#     def gps_to_local_xy(self, lon, lat):
#         # Convert GPS coordinates to UTM
#         x, y = self.transformer.transform(lon, lat)
        
#         # Calculate local x-y relative to the datum
#         local_x = x - self.datum_x
#         local_y = y - self.datum_y
        
#         return local_x, local_y

#     def error_and_Jacobian(self, x, U, Xref):
#         dXdU = np.zeros((self.nx , self.nu * self.Nu))
#         ex = np.zeros((self.nx, self.Np+1))
#         Jx = np.zeros((self.nx * (self.Np+1), self.nu * self.Nu))

#         for k in range(self.Np+1):
#             if k < self.Nu:
#                 ucur = U[:, [k]]
#             else:
#                 ucur = U[:, [self.Nu - 1]]

#             ex[:, [k]] = self.Qctrl @ (x - Xref[:, [k]])
#             Jx[k * self.nx:(k + 1) * self.nx, :] = self.Qctrl @ dXdU
#             x, dXdU = self.state_transition_U(x, ucur, self.dt, dXdU, k)

#         Ju = self.Rctrl * np.eye(self.nu * self.Nu)
#         e = np.vstack((ex.reshape(-1, 1, order='f'), self.Rctrl * U.reshape(-1, 1, order='f')))
#         J = np.vstack((Jx, Ju))

#         return e, J
    
#     def error_only(self, x, U, Xref):
#         ex = np.zeros((self.nx, self.Np+1))
#         for k in range(self.Np+1):
#             if k < self.Nu:
#                 ucur = U[:, [k]]
#             else:
#                 ucur = U[:, [self.Nu - 1]]
            
#             ex[:, [k]] = self.Qctrl @ (x - Xref[:, [k]])
#             x = self.state_transition_xonly(x, ucur, self.dt)
        
#         e = np.vstack((ex.reshape(-1, 1, order='f'), self.Rctrl * U.reshape(-1, 1, order='f')))
#         return e

#     def CTStateModel(self, x, u):
#         # Unpack state and inputs
#         N, E, psi, u_vel, v, r = x[:,0]
#         F_l = self.input_scale * u[0,0]
#         F_r = self.input_scale * u[1,0]
        
#         # Compute state derivatives
#         dN = u_vel * np.cos(psi) - v * np.sin(psi)
#         dE = u_vel * np.sin(psi) + v * np.cos(psi)
#         dpsi = r
#         du = (self.m22 * v * r - (self.d11 + self.d11_2 * np.fabs(u_vel)) * u_vel + F_l + F_r) / self.m11
#         dv = (-self.m11 * u_vel * r - (self.d22 + self.d22_2 * np.fabs(v)) * v) / self.m22
#         dr = ((F_r - F_l) * self.d_m - (self.d33 + self.d33_2 * np.fabs(r)) * r) / self.m33

#         return np.array([[dN],[dE],[dpsi],[du],[dv],[dr]])

#     def CTStateModelDx(self, x):
#         # Unpack state
#         N, E, psi, u_vel, v, r = x[:,0]

#         # Compute Jacobian
#         J = np.array([
#             [0, 0, (-u_vel * np.sin(psi) - v * np.cos(psi)), np.cos(psi), -np.sin(psi), 0],
#             [0, 0, (u_vel * np.cos(psi) - v * np.sin(psi)), np.sin(psi), np.cos(psi), 0],
#             [0, 0, 0, 0, 0, 1.0],
#             [0, 0, 0, 0 - ((self.d11 + 2.0 * self.d11_2 * np.fabs(u_vel)) / self.m11), (self.m22 / self.m11) * r, (self.m22 / self.m11) * v],
#             [0, 0, 0, -(self.m11 / self.m22) * r, -((self.d22 + 2.0 * self.d22_2 * np.fabs(v)) / self.m22), -(self.m11 / self.m22) * u_vel],
#             [0, 0, 0, 0, 0, -((self.d33 + 2.0 * self.d33_2 * np.fabs(r)) / self.m33)]
#         ])
        
#         return J

#     def state_transition_A(self, x, dt, u):
#         k1 = self.CTStateModel(x,u)
#         k2 = self.CTStateModel(x+(dt/2.0)*k1,u)
#         k3 = self.CTStateModel(x+(dt/2.0)*k2,u)
#         k4 = self.CTStateModel(x+dt*k3,u)

#         xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
#         xn = xn.reshape(-1,1,order='f')

#         dk1dx = self.CTStateModelDx(x)
#         dk2dx = self.CTStateModelDx(x+(dt/2.0)*k1) @ (np.eye(6)+(dt/2.0)*dk1dx)
#         dk3dx = self.CTStateModelDx(x+(dt/2.0)*k2) @ (np.eye(6)+(dt/2.0)*dk2dx)
#         dk4dx = self.CTStateModelDx(x+(dt/1.0)*k3) @ (np.eye(6)+(dt/1.0)*dk3dx)
#         A = np.eye(6) + (dt/6.0) * (dk1dx + 2.0*dk2dx + 2.0*dk3dx + dk4dx)
        
#         return xn, A
    
#     def state_transition_xonly(self, x, u, dt):
#         k1 = self.CTStateModel(x,u)
#         k2 = self.CTStateModel(x+(dt/2.0)*k1,u)
#         k3 = self.CTStateModel(x+(dt/2.0)*k2,u)
#         k4 = self.CTStateModel(x+dt*k3,u)

#         xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
#         return xn.reshape(-1,1,order='f')

#     def state_transition_U(self, x, u, dt, dxdu,k):
#         k1 = self.CTStateModel(x,u)
#         k2 = self.CTStateModel(x+(dt/2.0)*k1,u)
#         k3 = self.CTStateModel(x+(dt/2.0)*k2,u)
#         k4 = self.CTStateModel(x+dt*k3,u)

#         xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
#         xn = xn.reshape(-1,1,order='f')

#         B = np.array([
#             [0, 0],
#             [0, 0],
#             [0, 0],
#             [1/self.m11, 1/self.m11],
#             [0, 0],
#             [-self.d_m/self.m33, self.d_m/self.m33]
#         ])

#         dk1du = self.CTStateModelDx(x) @ dxdu
#         if k < self.Nu:
#             dk1du[:,k*self.nu:(k+1)*self.nu] += B
#         else:
#             dk1du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
#         dk2du = self.CTStateModelDx(x+(dt/2.0)*k1) @ (dxdu+(dt/2.0)*dk1du)
#         if k < self.Nu:
#             dk2du[:,k*self.nu:(k+1)*self.nu] += B
#         else:
#             dk2du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
#         dk3du = self.CTStateModelDx(x+(dt/2.0)*k2) @ (dxdu+(dt/2.0)*dk2du)
#         if k < self.Nu:
#             dk3du[:,k*self.nu:(k+1)*self.nu] += B
#         else:
#             dk3du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
#         dk4du = self.CTStateModelDx(x+(dt/1.0)*k3) @ (dxdu+(dt/1.0)*dk3du)
#         if k < self.Nu:
#             dk4du[:,k*self.nu:(k+1)*self.nu] += B
#         else:
#             dk4du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
#         dxdun = dxdu + (dt/6.0) * (dk1du + 2.0*dk2du + 2.0*dk3du + dk4du)
        
#         return xn, dxdun

# def main(args=None):
#     rclpy.init(args=args)
#     controller = WAMV_NMPC_Controller()
#     rclpy.spin(controller)
#     controller.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



####################################


#########################


import sys
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, Twist
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64, Float64MultiArray
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
from pyproj import Proj
from pyproj import Proj, Transformer
from qpsolvers import Problem, solve_problem

class WAMV_NMPC_Controller(Node):

    def __init__(self):
        super().__init__('wamv_nmpc_controller')
        
        # Model parameters
        self.m11, self.m22, self.m33 = 345.0, 256.0, 944.0
        self.d11, self.d22, self.d33 = 137.0, 79.0, 934.0
        self.d11_2, self.d22_2, self.d33_2 = 225.0, 55.0, 1057.0
        self.d_m, self.L_m = 1.0, 2.3
        # self.d_b, self.L_b = 0.8, 1.5
        
        # NMPC parameters
        self.Np = 50  # Prediction horizon
        self.Nu = 30        
        self.dt = 0.5  # Time step
        self.nu = 2
        self.nx = 6
        
        # Initialize state and reference
        self.current_state = np.zeros((self.nx,1))
       
        # Last known actuator inputs
        self.last_inputs = np.zeros((self.nu,1))

        # EKF parameters
        self.P = np.diag([1000.0, 1000.0, 10.0, 0.0, 0.0, 0.0]) # Initial state covariance
        self.Q = np.diag([0.1, 0.1, 0.01, 0.01, 0.01, 0.01])  # Process noise covariance
        self.R_gps = np.diag([1.0, 1.0])  # GPS measurement noise covariance
        self.R_imu = np.diag([0.01, 0.01])  # IMU measurement noise covariance
        
        #Control parameters
        self.Qctrl = np.diag([100, 100, 200, 0.00001, 0.00001, 0.1])
        self.Rctrl = 2
        self.thrust_lower_bound = -400
        self.thrust_upper_bound = 400

        self.U     = np.zeros((self.nu, self.Nu))
        self.Xref  = np.zeros((self.nx, self.Np+1))
        self.input_scale = 1
        self.waypoint = np.array([[-350], [750]])
        self.waypoints = np.array([[-400],[700],[np.pi/2]])

        # Sydney Regatta Centre coordinates (approximate center)
        self.datum_lat = -33.7285
        self.datum_lon = 150.6789

        # Initialize projections
        self.proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        self.proj_utm = Proj(proj='utm', zone=56, datum='WGS84', south=True)
        self.transformer = Transformer.from_proj(self.proj_wgs84, self.proj_utm)

        # Calculate datum in UTM coordinates
        self.datum_x, self.datum_y = self.transformer.transform(self.datum_lon, self.datum_lat)
        
        # Counters for number of gps and imu updates
        self.gpsUpdates = 0
        self.imuUpdates = 0
        self.prev_yaw = None
        self.yaw_offset = 0.0
        self.headingOld = 0.0
        self.currentwaypoint = 0

        # Timestamp of last update
        self.last_update_time = self.get_clock().now().nanoseconds / 1e9
        
        # ROS2 publishers and subscribers
        self.cmd_L_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.cmd_R_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        # self.ref_sub = self.create_subscription(Float64MultiArray, '/wamv/reference', self.reference_callback, 10)
        self.ref_sub = self.create_subscription(PoseArray, '/vrx/wayfinding/waypoints', self.reference_callback, 10)
        
        # Timer for control loop
        self.create_timer(self.dt, self.control_loop)
    
    def error_and_Jacobian(self, x, U, Xref):
        dXdU = np.zeros((self.nx , self.nu * self.Nu))
        ex = np.zeros((self.nx, self.Np+1))
        Jx = np.zeros((self.nx * (self.Np+1), self.nu * self.Nu))

        for k in range(self.Np+1):
            if k<self.Nu:
                ucur = U[:,[k]]
            else:
                ucur = U[:,[self.Nu-1]]

            ex[:,[k]] = self.Qctrl @ (x - Xref[:,[k]])
            Jx[k*self.nx:(k+1)*self.nx,:] = self.Qctrl @ dXdU
            x, dXdU = self.state_transition_U(x, ucur, self.dt, dXdU, k)

        Ju = self.Rctrl * np.eye(self.nu * self.Nu)
        e  = np.vstack((ex.reshape(-1,1,order='f'), self.Rctrl * U.reshape(-1,1,order='f')))
        J  = np.vstack((Jx, Ju))

        return e, J
    
    def error_only(self, x, U, Xref):
        ex = np.zeros((self.nx, self.Np+1))
        for k in range(self.Np+1):
            if k<self.Nu:
                ucur = U[:,[k]]
            else:
                ucur = U[:,[self.Nu-1]]
            
            ex[:,[k]] = self.Qctrl @ (x - Xref[:,[k]])
            x = self.state_transition_xonly(x, ucur, self.dt)
        
        e  = np.vstack((ex.reshape(-1,1,order='f'), self.Rctrl * U.reshape(-1,1,order='f')))
        return e


    def computeTrajectory(self, x):
        #Compute unit direction vector from x to next waypoint
        uv = self.waypoint[0:2,:] - x[0:2,:]
        distance_to_waypoint = np.linalg.norm(uv)
        uv = uv / distance_to_waypoint

        maxSpeed = 2.0 #m/s
        maxDistance = self.Np * self.dt * maxSpeed
        if maxDistance < distance_to_waypoint:
            wpX = x[0,0] + maxDistance* uv[0,0]
            wpY = x[1,0] + maxDistance* uv[1,0]
        else:
            wpX = self.waypoint[0,0]
            wpY = self.waypoint[1,0]

        self.Xref[0,:] = self.waypoint[0,0] # np.linspace(x[0,0], wpX, self.Np+1)
        self.Xref[1,:] = self.waypoint[1,0] # np.linspace(x[1,0], wpY, self.Np+1)
        
        maxAngSpeed = 0.5 #rads/s
        if distance_to_waypoint < 10.0:
            heading = self.waypoint[2,0] #self.headingOld
        else:
            heading = np.arctan2(uv[1,0],uv[0,0])
        self.Xref[2,:] = self.waypoint[2,0] # np.linspace(x[2,0], heading, self.Np+1)
        self.headingOld = heading

        # print(self.Xref[0:3,:])

    def control_loop(self):
        print(self.current_state)
        if self.waypoints is not None and self.gpsUpdates > 100 and self.imuUpdates > 100:
            # Apply previous control action and compute predicted state at k+1
            self.last_inputs = self.U[:,0:1]
            x = self.state_transition_xonly(self.current_state, self.last_inputs, self.dt)

            print(self.last_inputs)

            # Publish control commands
            msg = Float64()
            msg.data = float(self.last_inputs[0] * self.input_scale)
            self.cmd_L_pub.publish(msg)
            msg.data = float(self.last_inputs[1] * self.input_scale)
            self.cmd_R_pub.publish(msg)

            # Start timer
            start_time = time.time()

            # Solve NMPC problem
            U = np.zeros((self.nu, self.Nu))
            U[:,:-1] = self.U[:,1:]
            U[:,-1]  = self.U[:,-1]

            #Check if we are close to waypoint and move to the next
            if np.linalg.norm(self.waypoints[:,[self.currentwaypoint]] - self.current_state[:3,[0]]) < 0.5:
                self.currentwaypoint += 1
                if self.currentwaypoint >= self.waypoints.shape[1]:
                    self.currentwaypoint = self.waypoints.shape[1]-1

            #Pick waypoint from the list
            self.waypoint = self.waypoints[:,[self.currentwaypoint]]

            #Determine trajectory
            self.computeTrajectory(x)

            # Run main optimisation loop
            for i in range(10):
                #compute error and jacobian
                e, J = self.error_and_Jacobian(x, U, self.Xref)

                # Compute search direction
                # p, res, tmp, sv = np.linalg.lstsq(J,e, rcond=None)
                H = J.T @ J
                f = J.T @ e
                lb = -U.reshape(-1,1,order='f') + self.thrust_lower_bound*np.ones((self.Nu * self.nu,1))
                ub = -U.reshape(-1,1,order='f') + self.thrust_upper_bound*np.ones((self.Nu * self.nu,1))
                prob = Problem(H,f,None,None,None,None,lb,ub)
                sol  = solve_problem(prob,solver='proxqp')

                p = sol.x
                lam = sol.z_box

                #Compute the Newton Decrement and return if less than some threshold
                if np.linalg.norm(f+lam.reshape(-1,1,order='f')) < 1e-2:
                    break

                #Compute cost (sum of squared errors)
                co = e.T @ e

                # print("Cost = {:f}, FONC = {:f}".format(co[0,0], np.linalg.norm(f+lam.reshape(-1,1,order='f'))))

                #Use backtracking line search
                alp = 1.0
                for j in range(30):
                    #Try out new input sequence 
                    Un = U + alp*p.reshape(self.nu,-1,order='f')
                    e = self.error_only(x, Un, self.Xref)
                    cn = e.T @ e
                    #If we have reduced the cost then accept the new input sequence and return
                    if np.isfinite(cn) and cn < co:
                        U = Un
                        break
                    #Otherwise halve the step length
                    alp = alp / 2.0

            #record the optimal input sequence
            self.U = U

            # End timer
            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)
            # print('end')

    def reference_callback(self, msg):
        # Extract yaw from quaternion for the reference pose
        self.waypoints = np.zeros((3,len(msg.poses)))
        for i in range(len(msg.poses)):
            quat = [msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w]
            euler = Rotation.from_quat(quat).as_euler('xyz')
            x, y = self.gps_to_local_xy(msg.poses[i].position.y, msg.poses[i].position.x)
            self.waypoints[:3,i] = [x, y, euler[2]]  # yaw is euler[2]

    def CTStateModel(self, x, u):
        # Unpack state and inputs
        N, E, psi, u_vel, v, r = x[:,0]
        F_l = self.input_scale * u[0,0]
        F_r = self.input_scale * u[1,0]
        
        # Compute state derivatives
        dN = u_vel * np.cos(psi) - v * np.sin(psi)
        dE = u_vel * np.sin(psi) + v * np.cos(psi)
        dpsi = r
        du = (self.m22 * v * r - (self.d11 + self.d11_2*np.fabs(u_vel)) * u_vel + F_l + F_r) / self.m11
        dv = (-self.m11 * u_vel * r - (self.d22 + self.d22_2*np.fabs(v)) * v) / self.m22
        dr = ((F_r - F_l) * self.d_m - (self.d33 + self.d33_2*np.fabs(r)) * r) / self.m33

        return np.array([[dN],[dE],[dpsi],[du],[dv],[dr]])

    def CTStateModelDx(self, x):
        # Unpack state
        N, E, psi, u_vel, v, r = x[:,0]

        # Compute Jacobian
        J = np.array([
            [0, 0, (-u_vel * np.sin(psi) - v * np.cos(psi)) , np.cos(psi) , -np.sin(psi) , 0],
            [0, 0, (u_vel * np.cos(psi) - v * np.sin(psi)) , np.sin(psi) , np.cos(psi) , 0],
            [0, 0, 0, 0, 0, 1.0],
            [0, 0, 0, 0 - ((self.d11 + 2.0*self.d11_2*np.fabs(u_vel)) / self.m11) , (self.m22 / self.m11) * r , (self.m22 / self.m11) * v ],
            [0, 0, 0, -(self.m11 / self.m22) * r , - ((self.d22 + 2.0*self.d22_2*np.fabs(v)) / self.m22) , -(self.m11 / self.m22) * u_vel ],
            [0, 0, 0, 0, 0, - ((self.d33 + 2.0*self.d33_2*np.fabs(r)) / self.m33)]
        ])
        
        return J

    def state_transition_A(self, x, dt, u):
        k1 = self.CTStateModel(x,u)
        k2 = self.CTStateModel(x+(dt/2.0)*k1,u)
        k3 = self.CTStateModel(x+(dt/2.0)*k2,u)
        k4 = self.CTStateModel(x+dt*k3,u)

        xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        xn = xn.reshape(-1,1,order='f')

        dk1dx = self.CTStateModelDx(x)
        dk2dx = self.CTStateModelDx(x+(dt/2.0)*k1) @ (np.eye(6)+(dt/2.0)*dk1dx)
        dk3dx = self.CTStateModelDx(x+(dt/2.0)*k2) @ (np.eye(6)+(dt/2.0)*dk2dx)
        dk4dx = self.CTStateModelDx(x+(dt/1.0)*k3) @ (np.eye(6)+(dt/1.0)*dk3dx)
        A = np.eye(6) + (dt/6.0) * (dk1dx + 2.0*dk2dx + 2.0*dk3dx + dk4dx)
        
        return xn, A
    
    def state_transition_xonly(self, x, u, dt):
        k1 = self.CTStateModel(x,u)
        k2 = self.CTStateModel(x+(dt/2.0)*k1,u)
        k3 = self.CTStateModel(x+(dt/2.0)*k2,u)
        k4 = self.CTStateModel(x+dt*k3,u)

        xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return xn.reshape(-1,1,order='f')

    def state_transition_U(self, x, u, dt, dxdu,k):
        k1 = self.CTStateModel(x,u)
        k2 = self.CTStateModel(x+(dt/2.0)*k1,u)
        k3 = self.CTStateModel(x+(dt/2.0)*k2,u)
        k4 = self.CTStateModel(x+dt*k3,u)

        xn = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        xn = xn.reshape(-1,1,order='f')

        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1/self.m11, 1/self.m11],
            [0, 0],
            [-self.d_m/self.m33, self.d_m/self.m33]
        ])

        dk1du = self.CTStateModelDx(x) @ dxdu
        if k<self.Nu:
            dk1du[:,k*self.nu:(k+1)*self.nu] += B
        else:
            dk1du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
        dk2du = self.CTStateModelDx(x+(dt/2.0)*k1) @ (dxdu+(dt/2.0)*dk1du)
        if k<self.Nu:
            dk2du[:,k*self.nu:(k+1)*self.nu] += B
        else:
            dk2du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
        dk3du = self.CTStateModelDx(x+(dt/2.0)*k2) @ (dxdu+(dt/2.0)*dk2du)
        if k<self.Nu:
            dk3du[:,k*self.nu:(k+1)*self.nu] += B
        else:
            dk3du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
        dk4du = self.CTStateModelDx(x+(dt/1.0)*k3) @ (dxdu+(dt/1.0)*dk3du)
        if k<self.Nu:
            dk4du[:,k*self.nu:(k+1)*self.nu] += B
        else:
            dk4du[:,(self.Nu-1)*self.nu:(self.Nu)*self.nu] += B
        dxdun = dxdu + (dt/6.0) * (dk1du + 2.0*dk2du + 2.0*dk3du + dk4du)
        
        return xn, dxdun

    def gps_to_local_xy(self, lon, lat):
        # Convert GPS coordinates to UTM
        x, y = self.transformer.transform(lon, lat)
        
        # Calculate local x-y relative to the datum
        local_x = x - self.datum_x
        local_y = y - self.datum_y
        
        return local_x, local_y

    def gps_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Convert GPS to local x-y coordinates
        x, y = self.gps_to_local_xy(msg.longitude, msg.latitude)
        
        self.update_ekf(dt, gps_measurement=np.array([x, y]))

        if self.gpsUpdates < 300:
            self.gpsUpdates += 1

    def imu_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        euler = Rotation.from_quat(quat).as_euler('xyz')
        yaw = euler[2]
        angular_velocity = msg.angular_velocity.z
        
        if self.prev_yaw is not None:
            # Check for wrap-around
            diff = yaw - self.prev_yaw
            if diff > np.pi:
                self.yaw_offset -= 2.0 * np.pi
            elif diff < -np.pi:
                self.yaw_offset += 2.0 * np.pi

        unwrapped_yaw = yaw + self.yaw_offset
        self.prev_yaw = yaw


        self.update_ekf(dt, imu_measurement=np.array([unwrapped_yaw, angular_velocity]))

        if self.imuUpdates < 300:
            self.imuUpdates += 1

    def update_ekf(self, dt, gps_measurement=None, imu_measurement=None):
        # Prediction step
        self.current_state, F = self.state_transition_A(self.current_state, dt, 0*self.last_inputs)
        self.P = F @ self.P @ F.T + self.Q * dt  # Scale process noise with dt

        # Update step
        if gps_measurement is not None:
            H_gps = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0]])
            y = gps_measurement.reshape(-1,1,order='f') - H_gps @ self.current_state
            S = H_gps @ self.P @ H_gps.T + self.R_gps
            K = self.P @ H_gps.T @ np.linalg.inv(S)
            self.current_state = self.current_state + K @ y
            self.P = (np.eye(6) - K @ H_gps) @ self.P

        if imu_measurement is not None:
            H_imu = np.array([[0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1]])
            y = imu_measurement.reshape(-1,1,order='f') - H_imu @ self.current_state
            S = H_imu @ self.P @ H_imu.T + self.R_imu
            K = self.P @ H_imu.T @ np.linalg.inv(S)
            self.current_state = self.current_state + K @ y
            self.P = (np.eye(6) - K @ H_imu) @ self.P

def main(args=None):
    rclpy.init(args=args)
    controller = WAMV_NMPC_Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# Jn = 0*J
# for j in range(self.Nu * self.nu):
#     Ut = U.copy()
#     Ut = Ut.reshape(-1,1,order='f')
#     Ut[j] += 1e-8
#     Ut = Ut.reshape(self.nu,-1,order='f')
#     ep, Jp = self.error_and_Jacobian(x, Ut, self.Xref)
#     Ut = U.copy()
#     Ut = Ut.reshape(-1,1,order='f')
#     Ut[j] -= 1e-8
#     Ut = Ut.reshape(self.nu,-1,order='f')
#     em, Jm = self.error_and_Jacobian(x, Ut, self.Xref)
#     Jn[:,[j]] = (ep-em)/2e-8

# print('hello')