
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet
import pybullet_data

import numpy as np
import math as m

from pybulletgym_rocus.envs.roboschool.robots.robot_bases import URDFBasedRobot

def rand_target_loc(np_random):
    '''
    generate random target location
    '''
    x = np_random.uniform(low=0.05, high=0.5)
    if np_random.randint(0, 2) == 0:
        x = -x
    y = np_random.uniform(low=-0.3, high=0.2)
    z = np_random.uniform(low=0.65, high=1.0)
    return x, y, z

class PandaReacher(URDFBasedRobot):

    initial_positions = {
        'panda_joint1': 0, 'panda_joint2': -0.3, 'panda_joint3': 0.0,
        'panda_joint4': -2, 'panda_joint5': 0, 'panda_joint6': 2.0,
        'panda_joint7': m.pi / 4
    }

    # target range
    tx_ranges = (-0.3, 0.3)
    ty_ranges = (-0.4, 0.1)
    tz_ranges = (0.8, 1.0)

    @staticmethod
    def asset_path():
        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")

    def __init__(self, base_position=(0, -0.65, 0.625), use_IK=False, control_orientation=True, shelf=False):
        plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
        plane_kwargs = {'useFixedBase': True, 'basePosition': [0, 0, 0], 'baseOrientation': [0, 0, 0, 1]}
        if not shelf:
            table_path = os.path.join(self.asset_path(), "things/table/table.urdf")
        else:
            table_path = os.path.join(self.asset_path(), "things/table/table_shelf.urdf")
        table_kwargs = {'useFixedBase': True, 'basePosition': [0, 0, 0], 'baseOrientation': [0, 0, 0, 1]}
        target_path = os.path.join(self.asset_path(), "things/sphere_small_red.urdf")
        target_kwargs = {'useFixedBase': True, 'basePosition': [0, 0, 0.8], 'baseOrientation': [0, 0, 0, 1]}
        URDFBasedRobot.__init__(self, model_urdf='franka_panda/panda_model.urdf', robot_name='panda', action_dim=7, obs_dim=22,
                basePosition=base_position, baseOrientation=[0, 0, 0.7071068, 0.7071068], fixed_base=True, self_collision=True,
                additional_urdfs=[(plane_path, plane_kwargs), (table_path, table_kwargs), (target_path, target_kwargs)])
        self._joint_name_to_ids = {}
        self.robot_id = 0
        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self.end_eff_idx = 9

        self._workspace_lim = [[0.3, 0.65], [-0.3, 0.3], [0.65, 1.5]]
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]
        self.action_dim = 7
       
    def robot_specific_reset(self, bullet_client, **kwargs):
        assert bullet_client == self._p
        # reset joints to home position
        num_joints = bullet_client.getNumJoints(0)
        self.control_joints = []
        for i in range(num_joints):
            joint_info = bullet_client.getJointInfo(0, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type in [pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC]:
                assert joint_name in self.initial_positions.keys()
                self._joint_name_to_ids[joint_name] = i
                j = self.jdict[joint_name]
                j.reset_current_position(self.initial_positions[joint_name], 0)
                self.control_joints.append(j)

                bullet_client.setJointMotorControl2(self.robot_id, i, pybullet.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0)

        self.ll = np.array([j.lowerLimit for j in self.control_joints])
        self.ul = np.array([j.upperLimit for j in self.control_joints])

        if self._use_IK:
            self._home_hand_pose = [0.2, 0.0, 0.8,
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, 0)),
                                    min(m.pi, max(-m.pi, 0))]
            self.apply_action(self._home_hand_pose)
            self._p.stepSimulation()

        # target reset
        if 'target_loc' in kwargs:
            tx, ty, tz = kwargs['target_loc']
        else:
            tx, ty, tz = rand_target_loc(self.np_random)
        self.tx, self.ty, self.tz = tx, ty, tz
        self._p.resetBasePositionAndOrientation(3, [tx, ty, tz], [0, 0, 0, 1])

    def calc_state(self):
        # [j1-j7, e_xyz, e_rpy, ve_xyz, t_xyz, d_xyz]
        # Create observation state
        observation = []

        # --- Joint poses --- # 7 entries
        jointStates = self._p.getJointStates(self.robot_id, self._joint_name_to_ids.values())
        jointPoses = [x[0] for x in jointStates]
        observation.extend(jointPoses)

        # --- End effector xyz rpy --- # 6 entries
        state = self._p.getLinkState(self.robot_id, self.end_eff_idx, computeLinkVelocity=1,
                               computeForwardKinematics=1)
        ex, ey, ez = state[0]
        observation.extend([ex, ey, ez])
        orn = state[1]
        euler = pybullet.getEulerFromQuaternion(orn)
        observation.extend(list(euler))  # roll, pitch, yaw

        # --- Cartesian linear velocity --- # 3 entries
        # standardize by subtracting the mean and dividing by the std
        vel_std = [0.04, 0.07, 0.03]
        vel_mean = [0.0, 0.01, 0.0]
        vel_l = list((np.array(state[6]) - vel_mean) / vel_std)
        observation.extend(vel_l)

        # absolute target position # 3 entries
        tx, ty, tz = self._p.getBasePositionAndOrientation(3)[0]
        observation.extend([tx, ty, tz])

        # relative target position to end effector # 3 entries
        observation.extend([tx - ex, ty - ey, tz - ez])

        return np.array(observation)

    def apply_action(self, action, max_vel=-1):
        if self._use_IK:
            self.apply_ik_action(action, max_vel)
        else:
            self.apply_joint_position_action(action)

    def apply_joint_position_action(self, action):
        assert len(action) == self.action_dim
        action = np.clip(np.array(action), -1, 1) * 0.05
        num_joints = len(self._joint_name_to_ids)
        cur_joint_values = self.calc_state()[:7]
        target = action + cur_joint_values
        target = np.clip(target, self.ll, self.ul)
        for i, t in enumerate(target):
            self._p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=i, controlMode=pybullet.POSITION_CONTROL,
                targetPosition=t, positionGain=0.5, velocityGain=1.0)

    def calc_potential(self):
        state = self.calc_state()
        return - np.linalg.norm(state[-3:])


    # def apply_ik_action(self, action, max_vel=-1):
    #     if not (len(action) == 3 or len(action) == 6 or len(action) == 7):
    #         raise AssertionError('number of action commands must be \n- 3: (dx,dy,dz)'
    #                              '\n- 6: (dx,dy,dz,droll,dpitch,dyaw)'
    #                              '\n- 7: (dx,dy,dz,qx,qy,qz,w)'
    #                              '\ninstead it is: ', len(action))
    #     # --- Constraint end-effector pose inside the workspace --- #
    #     dx, dy, dz = action[:3]
    #     new_pos = [dx, dy,
    #                min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

    #     # if orientation is not under control, keep it fixed
    #     if not self._control_orientation:
    #         new_quat_orn = pybullet.getQuaternionFromEuler(self._home_hand_pose[3:6])

    #     # otherwise, if it is defined as euler angles
    #     elif len(action) == 6:
    #         droll, dpitch, dyaw = action[3:]

    #         eu_orn = [min(m.pi, max(-m.pi, droll)),
    #                   min(m.pi, max(-m.pi, dpitch)),
    #                   min(m.pi, max(-m.pi, dyaw))]

    #         new_quat_orn = pybullet.getQuaternionFromEuler(eu_orn)

    #     # otherwise, if it is define as quaternion
    #     elif len(action) == 7:
    #         new_quat_orn = action[3:7]

    #     # otherwise, use current orientation
    #     else:
    #         new_quat_orn = self._p.getLinkState(self.robot_id, self.end_eff_idx)[5]

    #     # --- compute joint positions with IK --- #
    #     jointPoses = self._p.calculateInverseKinematics(self.robot_id, self.end_eff_idx, new_pos, new_quat_orn,
    #                                               maxNumIterations=100,
    #                                               residualThreshold=.001)

    #     # --- set joint control --- #
    #     if max_vel == -1:
    #         self._p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
    #                                     jointIndices=self._joint_name_to_ids.values(),
    #                                     controlMode=pybullet.POSITION_CONTROL,
    #                                     targetPositions=jointPoses,
    #                                     positionGains=[0.2] * len(jointPoses),
    #                                     velocityGains=[1.0] * len(jointPoses))

    #     else:
    #         for i in range(self._num_dof):
    #             self._p.setJointMotorControl2(bodyUniqueId=self.robot_id,
    #                                     jointIndex=i,
    #                                     controlMode=pybullet.POSITION_CONTROL,
    #                                     targetPosition=jointPoses[i],
    #                                     maxVelocity=max_vel)

    # def debug_gui(self):
    #     ws = self._workspace_lim
    #     p1 = [ws[0][0], ws[1][0], ws[2][0]]  # xmin,ymin
    #     p2 = [ws[0][1], ws[1][0], ws[2][0]]  # xmax,ymin
    #     p3 = [ws[0][1], ws[1][1], ws[2][0]]  # xmax,ymax
    #     p4 = [ws[0][0], ws[1][1], ws[2][0]]  # xmin,ymax

    #     self._p.addUserDebugLine(p1, p2, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
    #     self._p.addUserDebugLine(p2, p3, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
    #     self._p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
    #     self._p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)

    #     self._p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
    #                        parentLinkIndex=-1)
    #     self._p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
    #                        parentLinkIndex=-1)
    #     self._p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
    #                        parentLinkIndex=-1)

    #     self._p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
    #                        parentLinkIndex=self.end_eff_idx)
    #     self._p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
    #                        parentLinkIndex=self.end_eff_idx)
    #     self._p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
    #                        parentLinkIndex=self.end_eff_idx)

    # def get_action_dim(self):
    #     if self._use_IK:
    #         return 6
    #     else:
    #         return self.action_dim

    # def get_observation_dim(self):
    #     return len(self.get_observation())

    # def get_workspace(self):
    #     return [i[:] for i in self._workspace_lim]

    # def set_workspace(self, ws):
    #     self._workspace_lim = [i[:] for i in ws]

    # def get_rotation_lim(self):
    #     return [i[:] for i in self._eu_lim]

    # def set_rotation_lim(self, eu):
    #     self._eu_lim = [i[:] for i in eu]

    ################ grasp utilities ################
    # def pre_grasp(self):
    #     self.apply_action_fingers([0.04, 0.04])

    # def grasp(self, obj_id=None):
    #     self.apply_action_fingers([0.0, 0.0], obj_id)

    # def apply_action_fingers(self, action, obj_id=None):
    #     # move finger joints in position control
    #     assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

    #     idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

    #     # use object id to check contact force and eventually stop the finger motion
    #     if obj_id is not None:
    #         _, forces = self.check_contact_fingertips(obj_id)
    #         # print("contact forces {}".format(forces))

    #         if forces[0] >= 20.0:
    #             action[0] = self._p.getJointState(self.robot_id, idx_fingers[0])[0]

    #         if forces[1] >= 20.0:
    #             action[1] = self._p.getJointState(self.robot_id, idx_fingers[1])[0]

    #     for i, idx in enumerate(idx_fingers):
    #         self._p.setJointMotorControl2(self.robot_id,
    #                                 idx,
    #                                 pybullet.POSITION_CONTROL,
    #                                 targetPosition=action[i],
    #                                 force=10,
    #                                 maxVelocity=1)

    # def check_collision(self, obj_id):
    #     # check if there is any collision with an object

    #     contact_pts = self._p.getContactPoints(obj_id, self.robot_id)

    #     # check if the contact is on the fingertip(s)
    #     n_fingertips_contact, _ = self.check_contact_fingertips(obj_id)

    #     return (len(contact_pts) - n_fingertips_contact) > 0

    # def check_contact_fingertips(self, obj_id):
    #     # check if there is any contact on the internal part of the fingers, to control if they are correctly touching an object

    #     idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

    #     p0 = self._p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[0])
    #     p1 = self._p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[1])

    #     p0_contact = 0
    #     p0_f = [0]
    #     if len(p0) > 0:
    #         # get cartesian position of the finger link frame in world coordinates
    #         w_pos_f0 = self._p.getLinkState(self.robot_id, idx_fingers[0])[4:6]
    #         f0_pos_w = pybullet.invertTransform(w_pos_f0[0], w_pos_f0[1])

    #         for pp in p0:
    #             # compute relative position of the contact point wrt the finger link frame
    #             f0_pos_pp = pybullet.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

    #             # check if contact in the internal part of finger
    #             if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
    #                 p0_contact += 1
    #                 p0_f.append(pp[9])

    #     p0_f_mean = np.mean(p0_f)

    #     p1_contact = 0
    #     p1_f = [0]
    #     if len(p1) > 0:
    #         w_pos_f1 = self._p.getLinkState(self.robot_id, idx_fingers[1])[4:6]
    #         f1_pos_w = pybullet.invertTransform(w_pos_f1[0], w_pos_f1[1])

    #         for pp in p1:
    #             # compute relative position of the contact point wrt the finger link frame
    #             f1_pos_pp = pybullet.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

    #             # check if contact in the internal part of finger
    #             if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
    #                 p1_contact += 1
    #                 p1_f.append(pp[9])

    #     p1_f_mean = np.mean(p0_f)

    #     return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)
    ################ grasp utilities ################
