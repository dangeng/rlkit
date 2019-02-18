import gym
from gym import Env, spaces
import sys, os
import random
import numpy as np
import six
import click
import h5py
from grasper import log

import mujoco_py
from . import grasper_maker
from .target import MeshObject
from .mock_object import MockObject

import pdb
import matplotlib.pyplot as plt

# simulation parameters
grasp_torque = 2
height_difference = 0.18
up_height = -0.06
up_height = 0
contact_difference = 0.07
default_im_size = 500
get_out_of_the_picture_height=1


class GraspingWorld(Env):
    def __init__(self):
        self.is_gripping = False
        self.action_space = spaces.Discrete(512*512*4)
        self.observation_space = spaces.Box(0,255,[3,500,500],dtype=np.uint8)

    def reset(self):
        vhacd=False
        render=False
        seed=1337
        debug=False

        objs = []
        objs.append(MockObject('rectangle'))
        objs.append(MockObject('sphere'))
        objs.append(MockObject('cylinder'))
        objs.append(MockObject('ellipsoid'))
        #objs.append(MeshObject('whistle', vhacd, target_size=0.03, target_mass=0.05))
        self.obj_num = 0

        # create xml file for environment
        grasper_maker.grasper(target_objs=objs, friction=[5, 0.1, 0.1]).save('grasper/assets/grasper.xml')

        # start simulation
        self.model = Grasper(model_path="grasper/assets/grasper.xml", objs=objs, render=render, seed=seed, debug=debug)
        self.model.idle()
        count = 0

        target_xs, target_ys, target_angles = self.model.reset_targets()

        other_loc = []
        for n in range(len(self.model.objs)):
            target_x, target_y, target_angle = list(zip(target_xs, target_ys, target_angles))[n]
            gripper_goal, gripper_angle = self.model.reset_gripper(target_x, target_y, target_angle, n)
            other_loc.append(gripper_goal)

        del other_loc[self.obj_num]

        self.obj = self.model.objs[self.obj_num]
        target_x, target_y, target_angle = list(zip(target_xs, target_ys, target_angles))[self.obj_num]

        gripper_goal, gripper_angle = self.model.reset_gripper(target_x, target_y, target_angle, self.obj_num)

        self.z0 = self.model.obj_z()

        image = self.render()

        return image.copy()

    def img_to_mj(self, p):
        A = np.array([1570.87903103, -1570.54846599])
        b = np.array([248.25918547, 249.76304022])

        # Correction, translation needs to be origin centered
        # This is an empirical correction... (but so is the above ¯\_(ツ)_/¯)
        correction = .8

        return correction * (p-b)/A

    def actionTranslator(self, action):
        idx = np.unravel_index(action, (4,512,512))

        angles = [-3*np.pi/8, -np.pi/8, np.pi/8, 3*np.pi/8]
        p = idx[1:3]
        mj_coor = self.img_to_mj(p)
        angle = angles[idx[0]]
        
        return [mj_coor[0], mj_coor[1], angle]

    def step(self, action):
        action = self.actionTranslator(action)

        if not self.is_gripping:
            gripper_goal = action[:2]
            gripper_angle = action[2]

            # execute grasp attempt
            self.model.set_gripper_angle(gripper_angle) # Set angle
            self.model.action(num_iters=220, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=0, dz_ref=0, torque=0) # shift
            self.model.action(num_iters=220, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=self.obj.get_height()-height_difference, dz_ref=0, torque=0) # down
            self.model.action(num_iters=200, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=self.obj.get_height()-height_difference, dz_ref=0, torque=grasp_torque/10) # grasp
            self.model.action(num_iters=100, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=self.obj.get_height()-height_difference, dz_ref=0, torque=grasp_torque) # firm

            #rising_flag = self.model.action(num_iters=200, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=up_height, dz_ref=0, torque=grasp_torque) # up

            # declare victory if any target is lifted off table
            up_flag = False
            for obj_num in range(len(self.model.objs)):
                up_flag = up_flag or self.model.obj_z(obj_num) - self.z0 > 0.01
            self.is_gripping = up_flag

            obs = self.render()
            reward = int(up_flat)
            print(reward)

            return obs, reward, True, dict()
        else:
            gripper_goal = action[:2]
            gripper_angle = action[2]

            self.model.action(num_iters=220, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=up_height, dz_ref=0, torque=grasp_torque) # shift
            self.model.action(num_iters=100, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=up_height, dz_ref=0, torque=-grasp_torque) # release
            #self.model.action(num_iters=100, x_ref=gripper_goal[0], y_ref=gripper_goal[1], z_ref=0, dz_ref=0, torque=0) # release

            # declare victory if target is lifted off table
            up_flag = self.model.obj_z() - self.z0 > 0.01
            self.is_gripping = up_flag

    def render(self):
        return self.model._render(mode='rgb_array').transpose(2,0,1)

class Grasper():
    """
    execute grasps in the environment defined by the given XML
    """
    def __init__(self, model_path, objs=None, num_objects=1, img_size=[64,64], render=False, seed=1337, debug=False):
        self.objs = objs
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.init_qpos = self.get_pos()
        self.init_qvel = self.get_vel()
        self.viewer = None
        self._viewers = {}
        self.timestep = self.model.opt.timestep
        self.img_size = img_size
        self.render = render
        self.debug = debug

    def step(self, ctrl, n_frames):
        ''' step the simulation forward `n_frames` steps '''
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def action(self, num_iters=301, x_ref=0, y_ref=0, z_ref=0, dz_ref=0, torque=0, dt=1):
        ''' move end effector to desired location and apply desired force '''
        z0 = self.obj_z()
        max_rise = 0.
        for i in range(num_iters):
            self.step([x_ref, y_ref, z_ref, dz_ref, torque], dt)
            max_rise = self.obj_z()-z0
            if self.render:
                self._render()

        rising_flag = max_rise > 0.0002
        return rising_flag

    def idle(self, time=100):
        ''' idle simulation for given time '''
        for i in range(time):
            if self.render:
                self._render()

    def close(self, time=100):
        ''' close viewing window '''
        self._render(close=True)

    def obj_z(self, obj_num=0):
        ''' get height of object '''
        return self.get_body_com("target%d" % obj_num)[2]

    def check_contact(self, obj_num=0):
        ''' check for contact between gripper and object '''
        gripper = np.round(self.get_body_com("gripper"), 7)
        target = np.round(self.get_body_com("target%d" % obj_num), 7)
        logger.debug(f"gripper: {gripper[2]}")
        logger.debug(f"target: {target[2]}")
        return abs(gripper[2]-target[2]) < contact_difference

    def check_obj_contact(self):
        ''' check if objects are touching each other '''
        # TODO verify this is what it's doing
        d = self.sim.data
        for coni in range(d.ncon):
            con = d.contact[coni]
            if con.geom1 > 3 and con.geom2 > 3:
                return True

        return False

    def reset_targets(self):
        ''' generate a new scene by randomizing object locations and orientations '''

        for i in range(50):
            qpos = self.init_qpos
            pos_xs = []
            pos_ys = []
            target_angles = []

            for i in range(len(self.objs)):
                pos_x = random.uniform(-0.1, 0.1)
                pos_y = random.uniform(-0.1, 0.1)
                # pos_x = -0.1
                # pos_y = 1
                target_angle = (np.random.rand()-0.5)*np.pi*2
                qpos[5+4*i:7+4*i] = [pos_x, pos_y]
                qpos[8+4*i] = target_angle

                pos_xs.append(pos_x)
                pos_ys.append(pos_y)
                target_angles.append(target_angle)

            self.set_state(qpos, self.init_qvel) # forward kinematics is calculated here

            # if objects are not colliding, the scene is valid
            if not self.check_obj_contact():
                break

        return pos_xs, pos_ys, target_angles

    def raise_obj(self, num, height=.1):
        qpos = self.sim.data.qpos.copy()
        qpos[7+4*num] += height
        self.set_state(qpos, self.init_qvel)

    def image_without(self, num):
        self.raise_obj(num)
        image = self._render(mode='rgb_array')
        self.raise_obj(num, -.1)
        return image

    def get_obj_angle(self, obj_num):
        ''' get orientation of object '''
        qpos = self.init_qpos
        qpos_idx = 8 + 4*obj_num
        return qpos[qpos_idx]

    def set_gripper_angle(self, angle):
        qpos = self.init_qpos
        qpos[3] = angle
        self.set_state(qpos, self.init_qvel)

    def reset_gripper(self, pos_x, pos_y, target_angle, obj_num, gripper_angle=None):
        ''' generate a goal location and orientation for the gripper,
        given location and orientation of the target object '''
        # 0, 1, 2: gripper location; 3: gripper angle; 4: fingertip; 5, 6, 7: target location;
        # 8: target angle;
        qpos = self.init_qpos

        if gripper_angle is None:
            gripper_angle = (np.random.rand()-0.5)*np.pi

        qpos[0:2] = [pos_x, pos_y] # gripper location
        qpos[3] = gripper_angle

        self.set_state(qpos, self.init_qvel)

        gripper_goal = np.squeeze(self.random_position([pos_x, pos_y], target_angle, obj_num, scale=1.5))

        return gripper_goal, gripper_angle

    def move_gripper_deterministic(self, pos_x, pos_y, gripper_angle):
        ''' Move gripper to the set pos_x, pos_y, angle '''
        # 0, 1, 2: gripper location; 3: gripper angle; 4: fingertip; 5, 6, 7: target location;
        # 8: target angle;
        qpos = self.init_qpos

        qpos[0:2] = [pos_x, pos_y] # gripper location
        qpos[3] = gripper_angle

        self.set_state(qpos, self.init_qvel)

        gripper_goal = np.array([pos_x, pos_y])#np.squeeze(self.random_position([pos_x, pos_y], target_angle, obj_num))

        logger.debug(f"gripper_goal: {gripper_goal}")
        return gripper_goal, gripper_angle

    def random_position(self, target_pos, target_angle, obj_num, scale=1):
        ''' choose a random point on the convex hull of the target object '''
        obj = self.objs[obj_num]
        minx, maxx = obj.get_x()
        miny, maxy = obj.get_y()
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)

        x *= scale
        y *= scale

        c, s = np.cos(target_angle), np.sin(target_angle)
        R = np.array([[c, -s], [s, c]])
        rotated = np.dot(R, np.array([x, y]))
        translated = rotated + np.array(target_pos)
        #return translated
        # Don't randomize position when reseting gripper
        return np.array(target_pos)

    def viewer_setup(self, setup=True):
        ''' view simulation from above '''
        if self.debug:
            # view from an angle
            self.viewer.cam.elevation = -30
            # track the object
            self.viewer.cam.trackbodyid = -1
            # camera distance
            self.viewer.cam.distance = self.model.stat.extent * 1.0
        else:
            # overhead view
            self.viewer.cam.elevation = -90
            # disable body id tracking
            self.viewer.cam.trackbodyid = -1
            # camera distance
            self.viewer.cam.distance = 0.3  # changed from .2 b/c multiple objects messes up camera position


    def get_pos(self):
        return self.sim.data.qpos.flatten()

    def get_vel(self):
        return self.sim.data.qvel.flatten()

    def set_pos(self, pos_lst, delta=True):
        pos = self.get_pos()
        vel = self.get_vel()
        for idx, value in pos_lst:
            if delta:
                pos[idx] += value
            else:
                pos[idx] = value
        self.set_state(pos, vel)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.sim.data.qpos[:] = np.copy(qpos)
        self.sim.data.qvel[:] = np.copy(qvel)
        self.sim.forward()

    def _render(self, mode='human', close=False, width=default_im_size, height=default_im_size):
        if close:
            if self.viewer is not None:
                self._get_viewer(mode).finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]

        elif mode == 'human':
            self._get_viewer(mode).render()


    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = .095
        return self.viewer

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def get_body_comvel(self, body_name):
        idx = self.sim.body_names.index(six.b(body_name))
        return self.sim.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.sim.body_names.index(six.b(body_name))
        return self.sim.data.xmat[idx].reshape((3, 3))

env = GraspingWorld()
env.reset()

