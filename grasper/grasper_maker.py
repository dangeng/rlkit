import os
import numpy as np
from contextlib import contextmanager
import tempfile
import glob
import random
import logging
from .mock_object import MockObject

logger = logging.getLogger('grasp sim')

class MJCModel(object):
    ''' base class wrapping mj model loaded from XML '''
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """
        Usage:

        model = MJCModel('grasper')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        """
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def save(self, path):
        with open(path, 'w') as f:
            self.root.write(f)

    def close(self):
        self.file.close()

class MJCModelRegen(MJCModel):
    def __init__(self, name, regen_fn):
        super(MJCModelRegen, self).__init__(name)
        self.regen_fn = regen_fn

    def regenerate(self):
        self.root = self.regen_fn().root

class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items():
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:

            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"


def grasper(gripper_pos=[0.0, 0.0, 0.2], target_pos=[0.0, 0.0, 0.0], friction=[2, 0.001, 0.001], target_objs=None):
    '''
    instantiate mujoco model for grasping task
     - tabletop with randomized background
     - target object(s) imported as meshes with randomized textures
     - 2-finger agent capable of 3-D movement and grasping
    '''
    target_pos, gripper_pos, friction = list(target_pos), list(gripper_pos), list(friction)
    gripper_angle = 0

    # model setup
    mjcmodel = MJCModel('grasper')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local", meshdir="meshes", texturedir="textures")
    mjcmodel.root.option(timestep="0.01",gravity="0 0 -9.81", tolerance=1e-12, impratio=10, noslip_iterations=100, cone='elliptic')
    size = mjcmodel.root.size()
    size.add_attr('nconmax', 1000)
    default = mjcmodel.root.default()
    default.joint(armature="1", damping=1, limited="true")
    default.geom(friction=friction, condim=4, contype="1", conaffinity="1", solimp="0.999 0.999 0.01", solref="0.01 1", rgba="1 1 1 1")

    texture_paths = [os.path.basename(x) for x in glob.glob('grasper/assets/textures/object_textures/*.png')]
    object_texture = f"object_textures/{np.random.choice(texture_paths)}"
    
    # define table and target assets
    asset = mjcmodel.root.asset()
    if target_objs:
        for i, target_obj in enumerate(target_objs):
            asset.texture(name='target%d_texture' % i, file=object_texture)
            asset.material(shininess='0.3', specular='1', name='target%d_material'%i, rgba='1 1 1 1', texture='target%d_texture' % i)
    asset.texture(name='table_texture', file="wood_table.png", type='2d')
    asset.material(shininess='0.1', specular='0.4', name='table_material', texture='table_texture', texrepeat="10 10")

    # define table and lighting
    worldbody = mjcmodel.root.worldbody()
    worldbody.light(directional="true", castshadow="false", cutoff="100", diffuse="1 1 1", specular=".1 .1 .1", exponent="1", pos="0 0 1.3", dir="-0 0 -1.3")
    worldbody.geom(name="table", condim=4, material='table_material', type="plane", rgba="1 1 1 1", pos="0 0 0", size="1 1 0.1", conaffinity="1", contype="1", friction=[1, 1, 1])

    # define agent to be a 2-finger gripper
    gripper = worldbody.body(name="gripper", pos=gripper_pos)
    gripper.geom(name="link0", condim=4, type="capsule", fromto=[-0.012, 0, 0, 0.012, 0, 0], rgba="0.9 0.3 0.3 1", size="0.01", conaffinity="1", contype="1", density=2000)
    gripper.joint(name="joint_x", type="slide", limited="false", pos="0 0 -1.5", axis="1 0 0")
    gripper.joint(name="joint_y", type="slide", limited="false", pos="0 0 -1.5", axis="0 1 0")
    gripper.joint(name="joint_z", type="slide", limited="false", pos="0 0 -1.5", axis="0 0 1")
    gripper.joint(name="gripper_hinge_z", type="hinge", pos="0 0 -1.5", axis="0 0 1", armature="0", damping="0.0001", stiffness="0", limited="false")
    fingertip1 = gripper.body(name="fingertip1", pos=[-0.012, 0, -0.01])
    fingertip1.joint(axis=[1, 0, 0], name="joint_tip", pos="0 0 0", limited="true", range="0 0.024", type="slide")
    fingertip1.geom(condim=4, size="0.0025 0.0025 0.02", name="ft1", rgba="0.9 0.3 0.3 1", type="box", conaffinity="1", contype="1", friction="2 0.010 0.0002", density=2000)
    fingertip2 = gripper.body(name="fingertip2", pos=[0.012, 0, -0.01])
    fingertip2.geom(condim=4, size="0.0025 0.0025 0.02", name="ft2", rgba="0.9 0.3 0.3 1", type="box", conaffinity="1", contype="1", friction="2 0.010 0.0002", density=2000)

    # define target objects
    targets = []
    for i, target_obj in enumerate(target_objs):
        target = worldbody.body(name="target%d" % i, pos=target_pos)

        if type(target_obj) == MockObject:
            xs, ys, z = target_obj.get_x(), target_obj.get_y(), target_obj.get_height()
            size_x = xs[1] - xs[0]
            size_y = ys[1] - ys[0]
            if target_obj.obj_type == 'rectangle':
                target.geom(condim=4, size=' '.join([str(size_x), str(size_y), str(z)]), name="targetbox%d" % i, material="target%d_material"%i, rgba="0.1 0.3 0.3 1", type="box", conaffinity="1", contype="1", friction="2 0.010 0.0002", density=2000)
            elif target_obj.obj_type == 'sphere':
                target.geom(condim=4, size=' '.join([str(size_x), str(size_y), str(z)]), name="targetsphere%d" % i, material="target%d_material"%i, rgba="0.1 0.3 0.3 1", type="sphere", conaffinity="1", contype="1", friction="2 0.010 0.0002", density=2000)
            elif target_obj.obj_type == 'cylinder':
                target.geom(condim=4, size=' '.join([str(size_x), str(size_y), str(z)]), name="targetcylinder%d" % i, material="target%d_material"%i, rgba="0.1 0.3 0.3 1", type="cylinder", conaffinity="1", contype="1", friction="2 0.010 0.0002", density=2000)
            elif target_obj.obj_type == 'ellipsoid':
                target.geom(condim=4, size=' '.join([str(size_x), str(size_y), str(z)]), name="targetellipsoid%d" % i, material="target%d_material"%i, rgba="0.1 0.3 0.3 1", type="ellipsoid", conaffinity="1", contype="1", friction="2 0.010 0.0002", density=2000)

        else:
            object_names = target_obj.filenames
            target_scale = target_obj.scale
            target_density = target_obj.density

            for object_name in object_names:
                logger.debug(f"adding mesh: {object_name}")
                # TODO: scale objects with respect to gripper size
                asset.mesh(file=target_obj.obj_name + "/" + object_name+".stl", name=object_name+"_mesh", scale=[target_scale, target_scale, target_scale])
                target.geom(condim=4, name=object_name+"_geom", conaffinity="1", contype="1", rgba="1 1 1 1", type="mesh", mesh=object_name+"_mesh", material="target%d_material"%i, density=target_density, friction="2 0.010 0.0002")
        targets.append(target)

    for i, target in enumerate(targets):
        target.joint(name="target%d_x"%i, type="slide", pos="0 0 0", axis="1 0 0", armature="0", damping="0", stiffness="0", limited="false")
        target.joint(name="target%d_y"%i, type="slide", pos="0 0 0", axis="0 1 0",
        armature="0", damping="0", stiffness="0", limited="false")
        target.joint(name="target%d_z"%i, type="slide", pos="0 0 0", axis="0 0 1",
        armature="0", damping="0", stiffness="0", limited="false")
        target.joint(name="target%d_hinge_z"%i, type="hinge", pos="0 0 0", axis="0 0 1", armature="0", damping="0", stiffness="0", limited="false")


    # define actuators (allow gripper fingers to move and apply force)
    actuator = mjcmodel.root.actuator()
    actuator.position(joint="joint_x", ctrllimited="false", kp=2)
    actuator.position(joint="joint_y", ctrllimited="false", kp=2)
    actuator.position(joint="joint_z", ctrllimited="false", kp=16)
    actuator.velocity(joint="joint_z", ctrllimited="false", kv=8)
    actuator.motor(joint="joint_tip", ctrlrange="-5.0 5.0", ctrllimited="true")

    return mjcmodel

