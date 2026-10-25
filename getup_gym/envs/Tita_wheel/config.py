# Get-up training config for 8-DOF wheeled bipedal robot (JROwheel)
from getup_gym.HhdRslRl.basement.base_env.base_config import BaseConfig
from getup_gym.HhdRslRl.basement.base_config.encoder_config import *
class TitaWheelGetupCfg(BaseConfig):
    
    class terrain:
        """Terrain friction and restitution parameters"""
        border_size = 20  # [m]25
        curriculum = True
        static_friction = 1.0   
        dynamic_friction = 0.7  
        restitution = 0.4           
        # rough terrain only:
        selected = False  # (select a unique terrain type and pass all arguments)
        terrain_kwargs = ""  # Dict of arguments for selected terrain
        max_init_terrain_level = 6  # starting curriculum state5
        
        # terrain types: 
        #1. stairs(up) with step width = 0.32, height = stairs_height_range
        #2. slopes whth max slope = slope_max
        #3. stairs(up) with step width = stairs_width_range, height = stairs_height_range
        #4. stairs(down) with step width = stairs_width_range, height = stairs_height_range
        #5. discrete obstacles with height = discrete_obstacles_height_range
        #6. random uniform terrain with max height = random_uniform_terrain_max_height
        #7. stepping stones with stone size = stepping_stones_size, stone distance = stone_distance
        #8. gaps with gap size = gap_size
        #9. pits with depth = pit_depth
        #else: plane
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = (
            0.8  # slopes above this threshold will be corrected to vertical surfaces
        )
        measure_heights = True

        #myself settings:
        mesh_type = "plane"   # "heightfield" # none, plane, heightfield or trimesh）
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        terrain_length = 6.0
        terrain_width = 6.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)

        measured_points_x = [
            -0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,
             0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
        ]#25
        measured_points_y = [
            -0.5,-0.4,-0.3,-0.2,-0.1,0.0,
             0.1,0.2,0.3,0.4,0.5
        ]

        #terrain params setting
        slope_max = 0.25
        stairs_height_range = [0.01, 0.1]
        stairs_width_range = [0.3, 0.42]
        discrete_obstacles_height_range = [0.01, 0.06]
        random_uniform_terrain_max_height = 0.03
    
    class env():
        num_envs = 4000
        num_envs_teacher = 3000
        num_actions = 8        
        layer_size = 12             # encoder's layer size    
        env_spacing = 3.0           # not used with heightfields/trimeshes
        send_timeouts = True        # (send time out information to the algorithm)
        episode_length_s = 10       # (episode length in seconds)
        unactuated_timesteps = 30

        class myself_setting:
            add_lots_last_time = True
            add_time_number = 5
        
    class commands():
        curriculum = False
        max_curriculum = 1.0
        num_commands = 6  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        base_height_command = False
        class ranges():
            lin_vel_x_only = 1500
            without_lin_vel_y = 500 # not add lin_vel_x_only
            lin_vel_x = [-0, 0]  # min max [m/s] 0.4
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]0.5
            angle_pitch = [-0.2, 0.2]
            angle_roll = [-0.2, 0.2]
            rew_base_height = [0.24, 0.4]
    
    class init_state:
        pos = [0.0, 0.0, 0.4]       # (x, y, z [m])
        rot = [0.0, 0.0, 0.0, 1.0] # [0.0, -1.0, 0.0, 1.0]  # (x, y, z, w [quat])
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]    
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_left_leg_1": 0.0,
            "joint_right_leg_1": -0.0,
            "joint_left_leg_2": 0.8, 
            "joint_right_leg_2": 0.8, 
            "joint_left_leg_3": -1.5,
            "joint_right_leg_3": -1.5,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        }  
        init_joint_angles = {  
            "joint_left_leg_1": 0.55,
            "joint_right_leg_1": -0.55,
            "joint_left_leg_2": 1.03, 
            "joint_right_leg_2": 1.03, 
            "joint_left_leg_3": -2.6,
            "joint_right_leg_3": -2.6,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        }
    
    class control:
        control_type ={"leg_1":"P","leg_2":"P","leg_3":"P","leg_4":"V"}   # P: position, V: velocity, T: torques
        stiffness = {"leg_1": 40,"leg_2": 40,"leg_3": 40,"leg_4":0.0}
        damping = {"leg_1": 1,"leg_2": 1,"leg_3":1,"leg_4":1.000}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25, 0.25, 0.25, 8.0, 0.25, 0.25, 0.25, 8.0]  # action scaling factors
        #if need change the action scale value in one training, use this mode
        using_actions_scale_changing = False
        action_scale_final = [0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 1.0]
        action_scale_changing_start_epochs = 3000
        action_scale_changing_end_epochs = 2000
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4   

    class asset:
        file = "{GETUP_GYM_ROOT_DIR}/getup_gym/robots/Tita/urdf/tita_description.urdf"
        name = "Tita"
        foot_name = ["left_leg_4",'right_leg_4']
        calf_name = ["left_leg_3",'right_leg_3']
        collapse_fixed_joints = True
        flip_visual_attachments = (
            False
        )
        
        terminate_after_contacts_on = []
        
        # -- rarely need to change below --
        disable_gravity = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        enable_gyroscopic_forces = False
        penalize_contacts_on = [""]

        density = 0.0                 # density
        angular_damping = 0.01           # angular damping coefficient
        linear_damping = 0.01           # linear damping coefficient
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001                # armature          
        thickness = 0.01               # thickness
        
        left_foot_indices = "left_leg_4"
        right_foot_indices = "right_leg_4"
        
    
    class domain_rand:

        randomize_friction = True  
        friction_range = [0.1, 1.2]
        randomize_base_mass = True
        added_mass_range = [-0.1, 1.2]  

        #PD domain rand
        randomize_pd = False
        kp_range = {"hiprange": [58.0, 62.0], "femrange": [39.0, 42.0], "tibrange": [39.0, 42.0], "wheelrange": [1.01, 0.99]}
        kd_range = {"hiprange": [3.4, 3.6], "femrange": [1.9, 2.1], "tibrange": [1.9, 2.1]}

        push_robots = True
        push_start_epoch = 2000
        push_interval_s = 5.0 
        max_push_vel_x = 1.0
        max_push_vel_y = 1.0

        #force push(not velocity push)
        randomize_force_push = False
        push_rigid_name = ["torso"]
        push_direction = [
            ["x", "y"],
        ]
        force_push_start_epoch = 3000
        #push the robot every force_push_interval_s seconds
        force_push_interval_s = 10.0 
        push_force_x_range = [-100.0, 100.0]
        push_force_y_range = [-100.0, 100.0]  
        push_force_z_range = [-0.0, 0.0]

        #rand restitution
        randomize_restitution = True
        restitution_range = [0.0, 0.7]

        #rand motor strength
        randomize_motor_strength = False
        joint_name_need_strength_rand = ["lfempitch", "rfempitch"]
        motor_strength_range = [0.5, 0.8]

        add_mass_ranges = True
        add_mass_range_x = {"upper": 0.02, "lower": -0.02}
        add_mass_range_y = {"upper": 0.01, "lower": -0.01}
        add_mass_range_z = {"upper": 0.01, "lower":  -0.01}

        #pose randm
        randomize_joint_pos = False
        rand_coeff = 0.4 #range = [-joint_limit* rand_coeff, joint_limit* rand_coeff]

        #torso size randm
        randomize_torso_size = False
        torso_size_coeff = 0.05 # torso size = size + [-size* torso_size_coeff, size* torso_size_coeff]
        #this is only used when the torso is box

    class rewards:
        
        class reward_group_1:
            pen_action_rate_l2 = -0.06
            pen_base_orientation_l2 = -0.8
            pen_dof_acc_l2 =-2.5e-08
            pen_femur_pos_bias_l2 = -1.0
            pen_dof_pos_bias_l2 =-0.01
            pen_no_fly_l2 =-2.0
            track_base_height_exp =2.0
            pen_termination =-100
            pen_torques_l2 =-1.0e-06
            pen_two_leg_bias_l2 =-0.01
            rew_wheel_contact_force = 4.0
        
        class reward_group_2:
            pen_termination = -200
            pen_base_orientation_l2 = -2.6
            pen_base_orientation_z_l2 = -0.6
            track_base_height_exp = 4.0
            pen_dof_acc_l2 = -2.5e-8
            pen_action_rate_l2 = -0.06
            pen_dof_pos_bias_l2 = -0.06
            pen_two_leg_bias_l2 = -0.02
            pen_no_fly_l2 = -2.0
            rew_wheel_contact_force = 4.0

        class reward_group_3:
            pen_termination = -100
            pen_base_orientation_l2 = -2.0
            pen_base_orientation_z_l2 = -1.0
            pen_torques_l2 = -1.0e-6
            pen_dof_acc_l2 = -2.5e-6
            pen_action_rate_l2 = -0.02
            pen_two_leg_bias_l2 = -0.02
            pen_dof_pos_bias_l2 = -0.14
            track_base_height_exp = 6.0
            wheel_adjustment = 0.5
            rew_wheel_contact_force = 4.0

        class reward_group_4:
            pen_termination = -500.0           #
            track_lin_vel_xy_exp = 7.0
            track_ang_vel_yaw_exp = 4.0
            pen_base_orientation_l2 = -15.0              #
            pen_torques_l2 = -1.0e-5
            pen_dof_acc_l2 = -2.5e-6
            pen_dof_vel_l2 = -4.0e-3
            pen_action_rate_l2 = -0.1
            pen_dof_pos_bias_l2 = -0.12
            pen_no_fly_l2 = -0.2
            track_base_height_exp = 8.0
            feet_distance = -1.0
            pen_max_velocity_l2 = 1.0
            pen_dof_pos_limits = -5.0

            pen_torque_limits = -0.05
            pen_action_smoothness_l2 = -0.02
            pen_joint_power_l2 = -1.0e-4
            wheel_adjustment = 1.0
            rew_wheel_contact_force = 4.0
            collision = -0.1

        min_feet_distance = 0.55
        max_feet_distance = 0.64
            
            

        class curriculum:
            """Stage-wise curriculum: reward groups, target heights, pull-up assistance."""
            reward_group = ["reward_group_1", "reward_group_2", "reward_group_3", "reward_group_4"]
            target_height = [0.1, 0.2, 0.3, 0.4]
            using_pull_up = True
            base_pull_up_max = 600.0
            using_pull_up_end = True
            pull_up_end_epochs = 3000

        class terminations:
            """Extra termination conditions beyond the default time-out."""
            using_time_overstep_terminate = False
            time_overstep_max = 10.0

        class delays:
            """Action delay schedule."""
            using_delay_action = True
            delay_start_epochs = 500

        only_positive_rewards = False
        tracking_sigma = 0.12  # tracking reward = exp(-error^2/sigma)0.0085
        soft_dof_pos_limit = (
                0.9  # percentage of URDF limits, values above this limit are penalized
            )
        soft_dof_vel_limit = 0.5  # penalizes joint velocity near limits
        soft_torque_limit = 0.8
        base_height_target = 0.4
        max_contact_force = 200.0  # forces above this value are penalized  200
        
    class normalization:
        
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            heading = 1.0
            base_height = 2.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        
        clip_observations = 100.0   # observation clipping range
        clip_actions = 50.0         # action clipping range
    
    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.04
            dof_vel = 0.06
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.1


    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]

    class sim:
        """Physics engine parameters (typically left unchanged)"""
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            """Physics engine parameters (keep defaults)"""
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 4
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )
    
class TitaWheelGetUpPPO(BaseConfig):
    """PPO training configuration"""
    seed = 900                                   # training random seed
    runner_class_name = "TSOnPolicyRunner"        # algorithm class name

    class runner:
        
        policy_class_name = "ActorCritic" #"ActorCritic" or "Actor_MutiCritic" "MoeActorCritic" "ResNetActorCritic"
        policy_config = PolicyConfig(
            init_noise_std = 1.0,                    
            actor_hidden_dims = [512, 256, 128],     
            critic_hidden_dims = [512, 256, 128],    
            activation = "elu",  
        )
        algorithm_class_name = "TSPPO"
        algorithm_config = AlgorithmsPPOConfig(
            value_loss_coef = 1.0,
            use_clipped_value_loss = True,
            clip_param = 0.2,
            entropy_coef = 0.01,
            num_learning_epochs = 5,
            num_mini_batches = 4,  # mini batch size = num_envs*nsteps / nminibatches
            learning_rate = 1.0e-3,  # 5.e-4
            schedule = "adaptive",  # could be adaptive, fixed
            gamma = 0.99,
            lam = 0.95,
            desired_kl = 0.01,
            max_grad_norm = 1.0,
        )
        
        #encoder setting
        #TODO: the teacher encoder will be updated use encoder-decoder alg when the teacher encoder is set to MLP-Encoder-Decoder
        #if algorithms = None, the policy will delete all the encoder and only use the basement reinforcement learning
        encoder_training_setting = TrainingAlgConfig(
            algorithms = "CTS", #"RMA-Teacher", "RMA-TeacherStudent", "CTS", None
            teacher_encoder_name = "MLP-Encoder", #"MLP-Encoder", "MLP-Encoder-Decoder" "Identity" "UNet" "MLP-UNet"            
            student_encoder_name = "MLP-Encoder", #"MLP-Encoder", "MLP-UNet"
            student_resume_check_point = "",
            teacher_encoder_alg_type = "KL-Entropy", #None
            student_encoder_alg_class_name = "MSE", #MSE
            student_encoder_inference_decimation = 10,
            teacher_encoder_config = MlpEncoderConfig(
                encoder_hidden_dims = [256, 128],
                activation = "elu",
            ),
            student_encoder_config = MlpEncoderConfig(
                encoder_hidden_dims = [256, 128],
                activation = "elu",
            ),
            student_encoder_alg_config = EncoderAlgConfig(
                learning_rate = 1.0e-3,
                num_mini_batches = 4,
                num_learning_epochs = 5, #test good is 3,maybe 4 is better
                max_grad_norm = 0.8,
                mse_loss_coef= 1.0,
                student_kl_coef= 0.1,
                alg_type = "MSE",
            )
        )
        
        force_guidance_config = ForceGuidanceConfig(
            using_force_guidance = True,
        )
    
        symmetrical_loss_config = SymmetricalConfig(
            using_symmetrical_loss = False,
            symmetrical_loss_coef = 0.02,
        )
        
        num_steps_per_env = 24      
        max_iterations = 8000        

        # logging
        save_interval = 500          # check for potential saves every this many iterations
        experiment_name = "Getup"     
        run_name = ""
        # load and resume
        resume = False
        load_run = -1               # -1 = last run
        checkpoint = -1             # -1 = last saved model
        resume_path = None          # updated from load_run and chkpt
# ============================================================================
# Force-guided fall-recovery framework (FTSR x FR-Net) for Tita_wheel.
# - encoder estimates per-link contact forces (no teacher-student)
# - external guidance force driven by relu(g_proj + 0.9), mixed into the
#   advantage as an optimizable constraint (FTSR penalty-function form)
# - staged rewards flagged by (g_proj < -0.9) AND (estimated base contact < thr)
# ============================================================================
class TitaWheelFFCfg(BaseConfig):
    """Environment config for the force-guided fall-recovery task."""

    class terrain:
        """Terrain friction and restitution parameters"""
        border_size = 20  # [m]25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 0.7
        restitution = 0.4
        selected = False
        terrain_kwargs = ""
        max_init_terrain_level = 6
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.8
        measure_heights = True

        mesh_type = "plane"  # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        terrain_length = 6.0
        terrain_width = 6.0
        num_rows = 10
        num_cols = 20

        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]
        measured_points_y = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5
        ]

        slope_max = 0.25
        stairs_height_range = [0.01, 0.1]
        stairs_width_range = [0.3, 0.42]
        discrete_obstacles_height_range = [0.01, 0.06]
        random_uniform_terrain_max_height = 0.03

    class env():
        num_envs = 4000
        num_envs_teacher = 3000
        num_actions = 8
        layer_size = 12             # encoder's layer size
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        unactuated_timesteps = 30
        num_contact_links = 9       # encoder predicts contact forces of all rigid-body links

        class myself_setting:
            add_lots_last_time = True
            add_time_number = 5     # encoder observation history length H

    class commands():
        curriculum = False
        max_curriculum = 1.0
        num_commands = 6
        resampling_time = 10.0
        heading_command = False
        base_height_command = False

        class ranges():
            lin_vel_x_only = 1500
            without_lin_vel_y = 500
            lin_vel_x = [-0, 0]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            angle_pitch = [-0.2, 0.2]
            angle_roll = [-0.2, 0.2]
            rew_base_height = [0.24, 0.4]

    class init_state:
        pos = [0.0, 0.0, 0.3]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "joint_left_leg_1": 0.0,
            "joint_right_leg_1": -0.0,
            "joint_left_leg_2": 0.8,
            "joint_right_leg_2": 0.8,
            "joint_left_leg_3": -1.5,
            "joint_right_leg_3": -1.5,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        }
        init_joint_angles = {
            "joint_left_leg_1": 0.55,
            "joint_right_leg_1": -0.55,
            "joint_left_leg_2": 1.03,
            "joint_right_leg_2": 1.03,
            "joint_left_leg_3": -2.6,
            "joint_right_leg_3": -2.6,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        }

    class control:
        control_type = {"leg_1": "P", "leg_2": "P", "leg_3": "P", "leg_4": "V"}
        stiffness = {"leg_1": 40, "leg_2": 40, "leg_3": 40, "leg_4": 0.0}
        damping = {"leg_1": 1, "leg_2": 1, "leg_3": 1, "leg_4": 1.000}
        action_scale = [0.25, 0.25, 0.25, 8.0, 0.25, 0.25, 0.25, 8.0]
        using_actions_scale_changing = False
        action_scale_final = [0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 1.0]
        action_scale_changing_start_epochs = 3000
        action_scale_changing_end_epochs = 2000
        decimation = 4

    class asset:
        file = "{GETUP_GYM_ROOT_DIR}/getup_gym/robots/Tita/urdf/tita_description.urdf"
        name = "Tita"
        foot_name = ["left_leg_4", 'right_leg_4']
        calf_name = ["left_leg_3", 'right_leg_3']
        collapse_fixed_joints = True
        flip_visual_attachments = False
        terminate_after_contacts_on = []
        disable_gravity = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        enable_gyroscopic_forces = False
        penalize_contacts_on = [""]
        density = 0.0
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001
        thickness = 0.01
        left_foot_indices = "left_leg_4"
        right_foot_indices = "right_leg_4"

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 1.2]
        randomize_base_mass = True
        added_mass_range = [-0.1, 1.2]

        randomize_pd = False
        kp_range = {"hiprange": [58.0, 62.0], "femrange": [39.0, 42.0], "tibrange": [39.0, 42.0], "wheelrange": [1.01, 0.99]}
        kd_range = {"hiprange": [3.4, 3.6], "femrange": [1.9, 2.1], "tibrange": [1.9, 2.1]}

        push_robots = True
        push_start_epoch = 2000
        push_interval_s = 5.0
        max_push_vel_x = 1.0
        max_push_vel_y = 1.0

        randomize_force_push = False
        push_rigid_name = ["torso"]
        push_direction = [["x", "y"]]
        force_push_start_epoch = 3000
        force_push_interval_s = 10.0
        push_force_x_range = [-100.0, 100.0]
        push_force_y_range = [-100.0, 100.0]
        push_force_z_range = [-0.0, 0.0]

        randomize_restitution = True
        restitution_range = [0.0, 0.7]

        randomize_motor_strength = False
        joint_name_need_strength_rand = ["lfempitch", "rfempitch"]
        motor_strength_range = [0.5, 0.8]

        add_mass_ranges = True
        add_mass_range_x = {"upper": 0.02, "lower": -0.02}
        add_mass_range_y = {"upper": 0.01, "lower": -0.01}
        add_mass_range_z = {"upper": 0.01, "lower": -0.01}

        randomize_joint_pos = False
        rand_coeff = 0.4

        randomize_torso_size = False
        torso_size_coeff = 0.05

    class rewards:
        # Single reward group: staging is done per-env inside the reward
        # functions via env.stage_flip_mask / env.stage_stand_mask.
        class reward_group_1:
            # always-on terms (reused from the base framework)
            pen_action_rate_l2 = -0.06
            pen_base_orientation_l2 = -2.0
            pen_base_orientation_z_l2 = -0.6
            pen_dof_acc_l2 = -2.5e-08
            pen_dof_pos_bias_l2 = -0.06
            pen_termination = -200
            pen_torques_l2 = -1.0e-06
            pen_two_leg_bias_l2 = -0.02
            track_base_height_exp = 4.0
            rew_wheel_contact_force = 2.0
            # stage 1 (flip recovery): FR-Net upright gaussian
            rew_proj_gravity_gaussian = 6.0
            # stage 2 (stable standing): posture convergence, base off ground, wheel support
            rew_stand_posture_exp = 4.0
            pen_base_contact_force_l2 = -2.0
            rew_wheel_support_exp = 2.0
            # feet_pose_track = 5.0
            # wheel_adjustment = 0.5

        # external guidance force (FTSR, driven by relu(g_proj + offset))
        class force_guidance:
            using_force_guidance = True
            force_body_name = "base_link"
            force_max = 600.0           # absolute cap [N]
            force_weight_ratio = 0.9    # cap = ratio * total_weight; must stay < 1 to avoid lift-off
            mu = 10.0
            proj_gravity_offset = 0.9
            force_end_epochs = 3000
            # constraint penalty factor mixing the force margin into the advantage
            constraint_beta = 0.001
            constraint_gamma = 0.99
            constraint_lam = 0.95

        # staged-reward flag: stand = (g_proj < thr) AND (estimated base contact < thr)
        class stage:
            proj_gravity_threshold = -0.9
            base_contact_force_threshold = 0.05  # normalized by total weight

        upright_sigma = 0.25
        posture_sigma = 0.4
        wheel_support_sigma = 0.5

        min_feet_distance = 0.55
        max_feet_distance = 0.64

        class curriculum:
            """Stage-wise curriculum: single reward group, old pull-up disabled."""
            reward_group = ["reward_group_1"]
            target_height = [0.1, 0.2, 0.3, 0.4]
            using_pull_up = False
            base_pull_up_max = 600.0
            using_pull_up_end = True
            pull_up_end_epochs = 3000

        class terminations:
            using_time_overstep_terminate = False
            time_overstep_max = 10.0

        class delays:
            using_delay_action = True
            delay_start_epochs = 500

        only_positive_rewards = False
        tracking_sigma = 0.12
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.5
        soft_torque_limit = 0.8
        base_height_target = 0.4
        max_contact_force = 200.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            heading = 1.0
            base_height = 2.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 50.0

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.04
            dof_vel = 0.06
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11.0, 5, 3.0]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 4
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class TitaWheelFFPPO(BaseConfig):
    """Training config for the force-guided PPO with contact-force encoder."""
    seed = 900
    runner_class_name = "FFOnPolicyRunner"

    class runner:
        policy_class_name = "ActorCriticFF"
        policy_config = dict(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            encoder_hidden_dims=[256, 128],
            activation="elu",
        )
        algorithm_class_name = "FFPPO"
        algorithm_config = dict(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            encoder_mse_coef=1.0,
            constraint_beta=0.001,
            constraint_gamma=0.99,
            constraint_lam=0.95,
        )
        # kept for compatibility with task_registry resume logic (algorithms=None -> no encoder files)
        class encoder_training_setting:
            algorithms = None

        force_guidance_config = ForceGuidanceConfig(using_force_guidance=True)

        num_steps_per_env = 24
        max_iterations = 8000

        save_interval = 500
        experiment_name = "Getup_FF"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
