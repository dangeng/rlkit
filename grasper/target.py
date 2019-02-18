import os
import numpy as np
import stl
import logging

logger = logging.getLogger('grasp sim')


class MeshObject(object):
    '''
    compute basic informations about the mesh object
    '''
    def __init__(self, obj_name, vhacd, stl_path='assets/meshes', target_size=0.025, target_mass=0.05):
        self.stl_path = stl_path
        self.obj_name = obj_name
        self.vhacd = vhacd
        self.target_size = target_size
        self.target_mass = target_mass
        self.filenames = self._decompose()
        self.coordinates, self.volume = self._compute_coordinates_and_volume()
        self.scale = self._compute_scale()
        self.density = self._compute_density()

    def _decompose(self):
        names = []
        if self.vhacd:
            for root, dirs, files in os.walk('{}'.format(self.stl_path + '/' + self.obj_name), topdown=False):
                for name in files:
                    if name.startswith('Shape') and name.endswith('.stl'):
                        names.append(name[:-4])
        else:
            names.append(self.obj_name)
        return names

    def _compute_coordinates_and_volume(self):
        ''' compute coordinates of smallest cube containing object,
        and its volume '''
        minx = maxx = miny = maxy = minz = maxz = None
        volume = 0.0
        for name in self.filenames:
            fn = 'grasper/{}/{}/{}.stl'.format(self.stl_path, self.obj_name, name)
            obj = stl.mesh.Mesh.from_file(fn)
            vol, cog, inertia = obj.get_mass_properties()
            volume += vol
            for p in obj.points:
                # p contains (x, y, z)
                if minx is None:
                    minx = p[stl.Dimension.X]
                    maxx = p[stl.Dimension.X]
                    miny = p[stl.Dimension.Y]
                    maxy = p[stl.Dimension.Y]
                    minz = p[stl.Dimension.Z]
                    maxz = p[stl.Dimension.Z]
                else:
                    maxx = max(p[stl.Dimension.X], maxx)
                    minx = min(p[stl.Dimension.X], minx)
                    maxy = max(p[stl.Dimension.Y], maxy)
                    miny = min(p[stl.Dimension.Y], miny)
                    maxz = max(p[stl.Dimension.Z], maxz)
                    minz = min(p[stl.Dimension.Z], minz)
        return [minx, maxx, miny, maxy, minz, maxz], volume

    def _compute_scale(self):
        ''' compute scale factor required to rescale mesh to desired size '''
        minx, maxx, miny, maxy, minz, maxz = self.coordinates
        max_length = max((maxx-minx),max((maxy-miny),(maxz-minz)))
        return self.target_size / max_length

    def _compute_density(self):
        return self.target_mass / ( self.volume * self.scale**3 )

    def get_x(self):
        minx, maxx, miny, maxy, minz, maxz = self.coordinates
        return minx * self.scale, maxx * self.scale

    def get_y(self):
        minx, maxx, miny, maxy, minz, maxz = self.coordinates
        return miny * self.scale, maxy * self.scale

    def get_height(self):
        minx, maxx, miny, maxy, minz, maxz = self.coordinates
        return (maxz - minz) * self.scale
