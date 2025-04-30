"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
import numpy as np
from scipy.interpolate import BSpline, splprep, splrep, splev
from collections import deque
from scipy.ndimage import gaussian_filter1d

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import path_planner as h
except ImportError:
    # PyTest import.
    from . import path_planner as h

#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        ## visualization
        # Plot trajectory in each dimension and 3D.
        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


    def planning(self, use_firmware, initial_info):
        """Trajectory planning algorithm"""
        ## generate waypoints for planning

        self.banking = False
        roll_angle = 30  # degrees
        self.ascend_with_angle = True
        ascend_speed = 1/3
        
        avg_desired_velocity = 1.00  # m/s

        if self.ascend_with_angle:
            self.post_takeoff_height = 0.2 # Make sure this matches height input for takeoff in cmdFirmware
            self.takeoff_time = self.post_takeoff_height / 0.33  # Time to reach the post-takeoff height
        else:
            self.post_takeoff_height = 1 # Make sure this matches height input for takeoff in cmdFirmware
            self.takeoff_time = self.post_takeoff_height / ascend_speed  # Time to reach the post-takeoff height

        # initial waypoint
        if use_firmware:
            gate_height = initial_info["gate_dimensions"]["tall"]["height"] # Height is hardcoded scenario knowledge.
        else:
            gate_height = self.initial_obs[4]

        start = (self.initial_obs[0], self.initial_obs[2])
        end = (-0.5,2.0)
        gate_rad = 0.2 #0.04 default
        obs_rad = 0.4 #0.06 default
        # np.random.shuffle(self.NOMINAL_GATES)
        #gates = self.NOMINAL_GATES
        gates = [self.NOMINAL_GATES[0], self.NOMINAL_GATES[2], self.NOMINAL_GATES[3], self.NOMINAL_GATES[1], self.NOMINAL_GATES[0], self.NOMINAL_GATES[3]]
        print(gates)
        #np.random.shuffle(gates)
        # print(self.NOMINAL_GATES)

        t = h.get_path(start, end, gate_rad, obs_rad, gates, self.NOMINAL_OBSTACLES)

        t = [(t[i]+(gate_height,)) for i in range(len(t))]  #Add 1.0 as the z coordinate of the waypoints
        if calculate_total_distance(t[:2])>0.8:
            t_1 = add_intermediate_points(t[:2], max_distance=0.8)  # Add intermediate points if the distance between two points exceeds 1 so the curve has straighter sections
            t = t_1 + t[2:]  # Concatenate the intermediate points with the rest of the waypoints
        t[0] = (t[0][0], t[0][1], self.post_takeoff_height)      # Set the starting z to be the height post takeoff
        # remove duplicates in the list of waypoints
        t = [t[i] for i in range(len(t)) if i == 0 or t[i] != t[i - 1]]
        if self.ascend_with_angle:
            ascend_time = (gate_height - self.post_takeoff_height) / ascend_speed  # Time to reach the gate height
            # Interpolate from start to the first waypoint
            t, interpolated_points = path_with_ascend(t, ascend_time, self.CTRL_FREQ)

        t = add_intermediate_points(t, max_distance=0.5)  # Add intermediate points if the distance between two points exceeds 1 so the curve has straighter sections
        
        self.waypoints = np.array(t)
            
        # Extract x, y, z coordinates
        waypoints_x = self.waypoints[:, 0]
        waypoints_y = self.waypoints[:, 1]
        waypoints_z = self.waypoints[:, 2]

        # Calculate total distance of waypoints
        total_distance = calculate_total_distance(self.waypoints)
        print(f"Total distance of waypoints: {total_distance:.2f} meters")

        duration = total_distance / avg_desired_velocity

        self.traj_dur = duration
        resolution = int(self.CTRL_FREQ*duration)
        t_scaled = np.linspace(0, duration, resolution)
        t_waypoints = np.linspace(0, duration, len(waypoints_x))
        #tck, u = splprep([waypoints_x, waypoints_y, waypoints_z], s=0.05)   # s=0 ensures the curve passes through all points
        smooth_pos = []
        smooth_vel = []
        smooth_acc = []
        waypoints = [waypoints_x, waypoints_y, waypoints_z]
        for dimension in range(3):
            if self.banking:
                tck = splrep(x=t_waypoints, y=waypoints[dimension], s=0.01)
            else:
                tck = splrep(x=t_waypoints, y=waypoints[dimension], s=0.05)   # s=0 ensures the curve passes through all points
            t, c, k = tck
            c0 = np.asarray(c)
            spline = BSpline(t, c0.T, k, extrapolate=False)
            smooth_pos.append(spline(t_scaled, nu=0))
            smooth_vel.append(spline(t_scaled, nu=1))
            smooth_acc.append(spline(t_scaled, nu=2))
        
        # Ensure smooth_vel does not exceed 1 m/s in any direction
        max_velocity = 1  # Maximum velocity in m/s
        smooth_vel[0] = np.clip(smooth_vel[0], -max_velocity, max_velocity)  # Clip x-velocity
        smooth_vel[1] = np.clip(smooth_vel[1], -max_velocity, max_velocity)  # Clip y-velocity
        smooth_vel[2] = np.clip(smooth_vel[2], -max_velocity, max_velocity)  # Clip z-velocity

        # Recalculate accelerations based on the adjusted velocities
        smooth_acc[0] = np.gradient(smooth_vel[0], t_scaled)  # Recalculate x-acceleration
        smooth_acc[1] = np.gradient(smooth_vel[1], t_scaled)  # Recalculate y-acceleration
        smooth_acc[2] = np.gradient(smooth_vel[2], t_scaled)  # Recalculate z-acceleration

        # Get target positions
        self.ref_x = smooth_pos[0]
        self.ref_y = smooth_pos[1]
        self.ref_z = smooth_pos[2]            
        

        if self.banking:

            # Adjust trajectory
            self.ref_x, self.ref_y, self.ref_z = adjust_for_bank_angle(self.ref_x, self.ref_y, self.ref_z, target_roll_angle=np.radians(roll_angle))

            # Use the recompute_t_scaled function to update t_scaled
            t_scaled = recompute_t_scaled(self.ref_x, self.ref_y, self.ref_z, velocity=avg_desired_velocity)

            # Adjust velocity
            self.ref_velocity = adjust_velocity_for_roll_rate(
                self.ref_x, self.ref_y, self.ref_z, target_roll_angle=np.radians(roll_angle), g=9.81
            )

            # Smooth the velocity profile
            self.ref_velocity = gaussian_filter1d(self.ref_velocity, sigma=2)
            

            # # Scale velocities to a maximum of 2 m/s
            max_velocity = 1  # Maximum velocity in m/s
            self.ref_velocity = np.clip(self.ref_velocity, 0, max_velocity)

            # Update velocities based on the adjusted velocity profile
            gradient_x = np.gradient(self.ref_x)
            gradient_y = np.gradient(self.ref_y)
            norm_gradients = np.linalg.norm([gradient_x, gradient_y], axis=0)

            # Avoid division by zero
            norm_gradients[norm_gradients == 0] = 1e-6

            self.ref_vx = self.ref_velocity * gradient_x / norm_gradients
            self.ref_vy = self.ref_velocity * gradient_y / norm_gradients
            self.ref_vz = np.gradient(self.ref_z, np.linspace(0, self.traj_dur, len(self.ref_z)))  # Keep z velocity as is

            # Recompute accelerations
            self.ref_ax = np.gradient(self.ref_vx, np.linspace(0, self.traj_dur, len(self.ref_vx)))
            self.ref_ay = np.gradient(self.ref_vy, np.linspace(0, self.traj_dur, len(self.ref_vy)))
            self.ref_az = np.gradient(self.ref_vz, np.linspace(0, self.traj_dur, len(self.ref_vz)))

            self.roll_rates = calculate_roll_rate(
                self.ref_x, self.ref_y, self.ref_z,
                self.ref_vx, self.ref_vy, self.ref_vz,
                self.CTRL_TIMESTEP, target_roll_angle=np.radians(roll_angle),
                lookahead=int(1.5*self.CTRL_FREQ)
            )
            # Smooth roll rates
            self.roll_rates[:, 1] = gaussian_filter1d(self.roll_rates[:, 1], sigma=2)
            # # print(roll_rates)
        else:
                # Get target velocities
                self.ref_vx = smooth_vel[0]
                self.ref_vy = smooth_vel[1]
                self.ref_vz = smooth_vel[2]

                # Get target accelerations
                self.ref_ax = smooth_acc[0]
                self.ref_ay = smooth_acc[1]
                self.ref_az = smooth_acc[2]
        if self.ascend_with_angle:
            # Add interpolated points back to the smoothed trajectory
            self.ref_x = np.concatenate((np.array([p[0] for p in interpolated_points]), self.ref_x))
            self.ref_y = np.concatenate((np.array([p[1] for p in interpolated_points]), self.ref_y))
            self.ref_z = np.concatenate((np.array([p[2] for p in interpolated_points]), self.ref_z))
            t_scaled = recompute_t_scaled_with_ascend(
                self.ref_x, self.ref_y, self.ref_z, ascend_time, len(interpolated_points), velocity=avg_desired_velocity
            )

            # Compute velocities and accelerations for the ascend part
            ascend_vel = []
            ascend_acc = []

            for i in range(1, len(interpolated_points)):
                dx = interpolated_points[i][0] - interpolated_points[i - 1][0]
                dy = interpolated_points[i][1] - interpolated_points[i - 1][1]
                dz = interpolated_points[i][2] - interpolated_points[i - 1][2]
                dt = ascend_time / len(interpolated_points)  # Time step for the ascend phase

                # Velocity for the current segment
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt
                ascend_vel.append((vx, vy, vz))

                # Acceleration for the current segment
                if i > 1:
                    prev_vx, prev_vy, prev_vz = ascend_vel[i - 2]
                    ax = (vx - prev_vx) / dt
                    ay = (vy - prev_vy) / dt
                    az = (vz - prev_vz) / dt
                    ascend_acc.append((ax, ay, az))

            # Add zero acceleration for the first point
            ascend_acc.insert(0, (0.0, 0.0, 0.0))

            # Convert to numpy arrays
            ascend_vel = np.array(ascend_vel).T  # Transpose to match smooth_vel format
            ascend_acc = np.array(ascend_acc).T  # Transpose to match smooth_acc format

            # Concatenate ascend velocities and accelerations with the smoothed trajectory
            self.ref_vx = np.concatenate((ascend_vel[0], smooth_vel[0]))
            self.ref_vy = np.concatenate((ascend_vel[1], smooth_vel[1]))
            self.ref_vz = np.concatenate((ascend_vel[2], smooth_vel[2]))

            self.ref_ax = np.concatenate((ascend_acc[0], smooth_acc[0]))
            self.ref_ay = np.concatenate((ascend_acc[1], smooth_acc[1]))
            self.ref_az = np.concatenate((ascend_acc[2], smooth_acc[2]))
            



        # Adjust velocity for straights with lookahead
        # self.ref_velocity = adjust_velocity_for_straights(
        #     self.ref_x, self.ref_y, self.ref_z, base_velocity=1, straight_velocity=1.5, curvature_threshold=0.1, lookahead=t_scaled.shape[0]//10
        # )
        
        # Prepare to record actual trajectory for error analysis post-flight
        self.act_x = np.array([])
        self.act_y = np.array([])
        self.act_z = np.array([])

        self.traj_dur = len(self.ref_x)/self.CTRL_FREQ
        print(f"Duration: {self.traj_dur}")

        self.landed = False
        self.stopped = False

        return t_scaled

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        # [INSTRUCTIONS] 
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)

        traj_start = self.takeoff_time
        #traj_dur = len(self.ref_x)/self.CTRL_FREQ
        traj_dur = self.traj_dur
        land_dur = 2

        # Take-off.
        if iteration == 0:
            height = self.post_takeoff_height
            duration = traj_start

            command_type = Command(2)  
            args = [height, duration]

        
        # [INSTRUCTIONS] Example code for using cmdFullState interface   
        elif iteration >= int(traj_start*self.CTRL_FREQ) and iteration < (traj_start+traj_dur)*self.CTRL_FREQ:
            step = min(iteration-int(traj_start*self.CTRL_FREQ), len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            if step>=len(self.ref_vx) or step>=len(self.ref_ax):
                    target_vel = np.array([0.0, 0.0, 0.0])
                    target_acc = np.array([0.0, 0.0, 0.0])
            else:
                target_vel = np.array([self.ref_vx[step], self.ref_vy[step], self.ref_vz[step]])
                target_acc = np.array([self.ref_ax[step], self.ref_ay[step], self.ref_az[step]])
            
            if self.banking:
                if np.linalg.norm(target_vel[:2]) > 1e-3:  # Avoid division by zero
                    target_yaw = np.arctan2(target_vel[1], target_vel[0])
                else:
                    target_yaw = 0.0  # Default yaw when velocity is near zero

                # Calculate yaw rate
                if step > 0:
                    prev_yaw = np.arctan2(self.ref_vy[step - 1], self.ref_vx[step - 1])
                    yaw_rate = (target_yaw - prev_yaw) * self.CTRL_FREQ  # Yaw rate (rad/s)
                else:
                    yaw_rate = 0.0

                # Limit yaw rate
                max_yaw_rate = np.radians(45)  # Example: 90 degrees per second
                yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)

                if step>=len(self.roll_rates):
                    target_rpy_rates = np.array([0.0, 0.0, 0.0])
                else:
                    # print(np.array([self.roll_rates[step, 1]]))
                    target_roll_rate = self.roll_rates[step, 1]
                    target_rpy_rates = np.array([target_roll_rate, 0.0, yaw_rate])  # Roll rate, pitch rate, yaw rate
            else:
                target_rpy_rates = np.array([0.0, 0.0, 0.0])
                target_yaw = 0.

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

            print(f"Iteration: {iteration}, Pos: {target_pos}, Vel: {target_vel}, Acc: {target_acc}")
            # print(f"Observed yaw: {obs[8]}")
            # Take note of the actual reported state
            self.act_x = np.append(self.act_x, obs[0])
            self.act_y = np.append(self.act_y, obs[2])
            self.act_z = np.append(self.act_z, obs[4])

        elif iteration >= (traj_start+traj_dur)*self.CTRL_FREQ and self.stopped == False:
            self.stopped = True
            command_type = Command(6)  # Notify setpoint stop.
            args = []

        elif iteration >= (traj_start+traj_dur)*self.CTRL_FREQ + 1 and self.landed == False:
            self.landed = True
            height = 0.
            duration = land_dur

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration >= (traj_start+traj_dur+land_dur+1)*self.CTRL_FREQ:
            command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0


def calculate_roll_rate(ref_x, ref_y, ref_z, ref_vx, ref_vy, ref_vz, dt, g=9.81, target_roll_angle=np.radians(45), lookahead=50):
    """
    Calculate the roll rate required for the drone to bank during turns.

    Args:
        ref_x, ref_y, ref_z: Arrays of the drone's reference positions.
        ref_vx, ref_vy, ref_vz: Arrays of the drone's reference velocities.
        dt: Time step between waypoints.
        g: Acceleration due to gravity (default: 9.81 m/s^2).
        target_roll_angle: Desired roll angle during turns (default: 45 degrees in radians).
        lookahead: Number of waypoints to look ahead for curvature calculation.

    Returns:
        roll_rates: Array of roll rates (rad/s).
    """
    roll_rates = []
    for i in range(1, len(ref_x) - 1):
        # Look ahead to calculate curvature
        lookahead_idx = min(i + lookahead, len(ref_x) - 1)
        dx1 = ref_x[lookahead_idx] - ref_x[i]
        dy1 = ref_y[lookahead_idx] - ref_y[i]
        dx2 = ref_x[i] - ref_x[i - 1]
        dy2 = ref_y[i] - ref_y[i - 1]
        curvature = np.abs(dx1 * dy2 - dy1 * dx2) / (np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2))
        r = 1 / curvature if curvature != 0 else np.inf

        # Compute velocity magnitude
        v = np.sqrt(ref_vx[i]**2 + ref_vy[i]**2 + ref_vz[i]**2)

        # Compute lateral acceleration
        a_y = v**2 / r if r != np.inf else 0

        # Compute roll angle
        roll_angle = np.arctan(a_y / g)

        # Adjust roll angle to target 45 degrees during turns
        if roll_angle > target_roll_angle:
            roll_angle = target_roll_angle

        # Compute roll rate
        if i > 1:
            prev_roll_angle = roll_rates[-1][0]
            roll_rate = (roll_angle - prev_roll_angle) / dt
        else:
            roll_rate = 0

        roll_rates.append((roll_angle, roll_rate))

    return np.array(roll_rates)

def adjust_for_bank_angle(ref_x, ref_y, ref_z, target_roll_angle, g=9.81):
    """Adjust trajectory to ensure the desired bank angle is reached."""
    adjusted_x, adjusted_y, adjusted_z = [ref_x[0]], [ref_y[0]], [ref_z[0]]
    for i in range(1, len(ref_x) - 1):
        dx1 = ref_x[i] - ref_x[i - 1]
        dy1 = ref_y[i] - ref_y[i - 1]
        dx2 = ref_x[i + 1] - ref_x[i]
        dy2 = ref_y[i + 1] - ref_y[i]
        curvature = np.abs(dx1 * dy2 - dy1 * dx2) / (np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2))
        if curvature > 0:
            # Calculate required velocity for target roll angle
            v = np.sqrt(target_roll_angle * g * (1 / curvature))
            adjusted_x.append(ref_x[i])
            adjusted_y.append(ref_y[i])
            adjusted_z.append(ref_z[i])
    adjusted_x.append(ref_x[-1])
    adjusted_y.append(ref_y[-1])
    adjusted_z.append(ref_z[-1])
    return np.array(adjusted_x), np.array(adjusted_y), np.array(adjusted_z)

def adjust_velocity_for_roll_rate(ref_x, ref_y, ref_z, target_roll_angle, g=9.81):
    """Adjust velocity to achieve the desired roll rate."""
    velocities = []
    for i in range(1, len(ref_x) - 1):
        dx1 = ref_x[i] - ref_x[i - 1]
        dy1 = ref_y[i] - ref_y[i - 1]
        dx2 = ref_x[i + 1] - ref_x[i]
        dy2 = ref_y[i + 1] - ref_y[i]
        curvature = np.abs(dx1 * dy2 - dy1 * dx2) / (np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2))
        if curvature > 0:
            # Calculate required velocity for target roll angle
            v = np.sqrt(target_roll_angle * g * (1 / curvature))
            velocities.append(v)
        else:
            velocities.append(1.0)  # Default velocity for straight sections
    velocities = [velocities[0]] + velocities + [velocities[-1]]  # Extend to match trajectory length
    return np.array(velocities)
def adjust_velocity_for_straights(ref_x, ref_y, ref_z, base_velocity=1.0, straight_velocity=2.0, curvature_threshold=0.05, lookahead=3):
    """Adjust velocity to speed up in straight sections based on curvature with lookahead."""
    velocities = []
    for i in range(1, len(ref_x) - 1):
        # Look ahead to calculate curvature
        lookahead_idx = min(i + lookahead, len(ref_x) - 1)
        dx1 = ref_x[lookahead_idx] - ref_x[i]
        dy1 = ref_y[lookahead_idx] - ref_y[i]
        dx2 = ref_x[i] - ref_x[i - 1]
        dy2 = ref_y[i] - ref_y[i - 1]
        curvature = np.abs(dx1 * dy2 - dy1 * dx2) / (np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2))
        
        # Increase velocity in straight sections (low curvature)
        if curvature < curvature_threshold:
            velocities.append(straight_velocity)
        else:
            velocities.append(base_velocity)
    
    # Extend velocities to match trajectory length
    velocities = [velocities[0]] + velocities + [velocities[-1]]
    return np.array(velocities)

def calculate_total_distance(waypoints):
    """Calculate the total distance of a series of waypoints."""
    total_distance = 0.0
    for i in range(1, len(waypoints)):
        p1 = waypoints[i - 1]
        p2 = waypoints[i]
        distance = np.linalg.norm(np.array(p2) - np.array(p1))  # Euclidean distance
        total_distance += distance
    return total_distance

# Add intermediate points if the distance between two points exceeds 1 so the curve has straighter sections
def add_intermediate_points(points, max_distance=0.5):
    """Add intermediate points if the distance between two points exceeds max_distance."""
    new_points = [points[0]]  # Start with the first point
    for i in range(1, len(points)):
        p1 = np.array(points[i - 1])
        p2 = np.array(points[i])
        dist = np.linalg.norm(p2 - p1)
        if dist > max_distance:
            # Calculate the number of intermediate points needed
            num_intermediate = int(np.ceil(dist / max_distance))
            for j in range(1, num_intermediate):
                # Linearly interpolate between p1 and p2
                intermediate_point = p1 + (p2 - p1) * (j / num_intermediate)
                new_points.append(tuple(intermediate_point))
        new_points.append(tuple(p2))
    return new_points

def path_with_ascend(t, ascend_time, ctrl_freq=30):
    # Interpolate from start to the first waypoint
    first_waypoint = t[0]
    first_gate = t[1]
    num_interpolation_points = int(ctrl_freq * ascend_time)  # Interpolate over ascend time
    interpolated_points = [
        (
            first_waypoint[0] + (first_gate[0] - first_waypoint[0]) * i / num_interpolation_points,
            first_waypoint[1] + (first_gate[1] - first_waypoint[1]) * i / num_interpolation_points,
            first_waypoint[2] + (first_gate[2] - first_waypoint[2]) * i / num_interpolation_points,
        )
        for i in range(num_interpolation_points)
    ]

    # only non ascend points are kept in the trajectory
    t = t[1:]
    return t, interpolated_points

def recompute_t_scaled_with_ascend(ref_x, ref_y, ref_z, ascend_time, ascend_points, velocity=1.0):
    """Recompute t_scaled to include ascend time and adjust for varying velocity."""
    distances = [0.0]  # Start with zero distance
    for i in range(1, len(ref_x)):
        dx = ref_x[i] - ref_x[i - 1]
        dy = ref_y[i] - ref_y[i - 1]
        dz = ref_z[i] - ref_z[i - 1]
        segment_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(distances[-1] + segment_distance)

    # Ensure ascend_points is valid
    ascend_points = min(ascend_points, len(ref_x))

    # Split the trajectory into ascend and the rest
    ascend_distances = distances[:ascend_points]
    rest_distances = distances[ascend_points:]

    # Time for the ascend phase
    if len(ascend_distances) > 1:
        ascend_t_scaled = np.linspace(0, ascend_time, len(ascend_distances))
    else:
        ascend_t_scaled = np.array([0])  # No ascend phase if there are no ascend points

    # Time for the rest of the trajectory
    if len(rest_distances) > 1:
        total_rest_distance = rest_distances[-1] - ascend_distances[-1] if len(ascend_distances) > 0 else rest_distances[-1]
        rest_time = total_rest_distance / velocity
        rest_t_scaled = np.linspace(ascend_time, ascend_time + rest_time, len(rest_distances))
    else:
        rest_t_scaled = np.array([])  # No rest phase if there are no rest points

    # Combine ascend and rest times
    t_scaled = np.concatenate((ascend_t_scaled, rest_t_scaled))
    return t_scaled

# Recompute t_scaled based on the adjusted trajectory
def recompute_t_scaled(ref_x, ref_y, ref_z, velocity=1.0):
    """Recompute t_scaled based on the adjusted trajectory and a constant velocity."""
    distances = [0.0]  # Start with zero distance
    for i in range(1, len(ref_x)):
        dx = ref_x[i] - ref_x[i - 1]
        dy = ref_y[i] - ref_y[i - 1]
        dz = ref_z[i] - ref_z[i - 1]
        segment_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(distances[-1] + segment_distance)

    # Total time is proportional to total distance divided by velocity
    total_time = distances[-1] / velocity
    t_scaled = np.linspace(0, total_time, len(ref_x))
    return t_scaled