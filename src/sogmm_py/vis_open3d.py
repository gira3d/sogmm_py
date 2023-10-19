"""Custom wrapper over Open3D visualization."""

import open3d
import numpy as np
import matplotlib.pyplot as plt
import time


class VisOpen3D:
    """Custom wrapper class over Open3D visualization.

    Attributes
    ----------
    __vis : open3d.visualization.Visualizer
        Visualizer object used throughout the class.
    """

    def __init__(self, width=1920, height=1080, visible=True, window_name='Open3D'):
        """
        Parameters
        ----------
        width : int, optional (default: 1920)
            Width of the visualizer window.
        height : int, optional (default: 1080)
            Height of the visualizer window.
        visibile : bool, optional (default: True)
            Whether the window is visible or not.
        """
        self._w = width
        self._h = height
        self._v = visible
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=visible, window_name=window_name)
        self.O3D_K = np.array([[935.30743609,   0.,         959.5],
                               [0.,         935.30743609, 539.5],
                               [0.,           0.,           1.]])

        if visible:
            self.poll_events()
            self.update_renderer()

    def __del__(self):
        self.__vis.destroy_window()

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def get_render_option(self):
        return self.__vis.get_render_option()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def create_window(self):
        self.__vis.create_window(
            width=self._w, height=self._h, visible=self._v)

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def add_geometries(self, datas):
        for data in datas:
            self.add_geometry(data)

    def update_view_point(self, intrinsic=None, extrinsic=None):
        if intrinsic is None:
            intrinsic = self.O3D_K
        if extrinsic is None:
            extrinsic = np.eye(4)

        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        param = open3d.io.read_pinhole_camera_parameters(filename)
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic
        self.update_view_point(intrinsic, extrinsic)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_image(self, filename):
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()

        return depth

    def capture_depth_image(self, filename):
        self.__vis.capture_depth_image(filename, do_render=True)

    def draw_camera(self, intrinsic=None, extrinsic=None, width=1920, height=1080, scale=1, color=None):
        # intrinsics
        if intrinsic is None:
            intrinsic = self.O3D_K
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        if extrinsic is None:
            extrinsic = np.eye(4)
        R = extrinsic[0:3, 0:3]
        t = extrinsic[0:3, 3]

        geometries = draw_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_points3D(self, points3D, color=None):
        geometries = draw_points3D(points3D, color)
        for g in geometries:
            self.add_geometry(g)

    def visualize_pcld(self, pcld=None, pcld_pose=None, K=None, W=None, H=None, scale=1.0, color=None):
        if pcld is None:
            return False

        self.add_geometry(pcld)
        self.draw_camera(intrinsic=K, extrinsic=pcld_pose,
                         width=W, height=H, scale=1.0, color=color)
        self.update_view_point(extrinsic=np.linalg.inv(pcld_pose))

    def frustrum(self, pose, K, W, H, scale=1.0, color=None):
        self.draw_camera(intrinsic=K, extrinsic=pose,
                         width=W, height=H, scale=scale, color=color)

    def frustrums(self, poses, K, W, H, colors, delay=None):
        for i, p in enumerate(poses):
            self.frustrum(p, K, W, H, colors[i])
            if delay is not None:
                self.poll_events()
                self.update_renderer()
                time.sleep(delay)

    def sphere(self, r, point=None, color=[0.1, 0.1, 0.7]):
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=r)
        if point is not None:
            mesh_sphere.translate(point)

        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)

        self.add_geometry(mesh_sphere)


def draw_camera(K, R, t, width, height, scale=1, color=None, viz_axis=True):
    geoms = []

    # default color
    if color is None:
        color = [0.8, 0.2, 0.8]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array([[K[0, 0] * s,  0, K[0, 2]],
                   [0,  K[1, 1] * s, K[1, 2]],
                   [0,            0, K[2, 2]]])
    Kinv = np.linalg.inv(Ks)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    if viz_axis:
        axis = create_coordinate_frame(T, scale=scale*0.5)
        geoms.append(axis)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ]

    # pixel to camera coordinate system
    points = [scale * Kinv @ p for p in points_pixel]

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 4],
        [3, 4],
        [1, 3],
    ]
    colors = [color for _ in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    geoms.append(line_set)

    return geoms


def create_coordinate_frame(T, scale=0.25):
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame


def draw_points3D(points3D, color=None):
    # color: default value
    if color is None:
        color = [0.8, 0.2, 0.8]

    geometries = []
    for pt in points3D:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01,
                                                            resolution=20)
        sphere.translate(pt)
        sphere.paint_uniform_color(np.array(color))
        geometries.append(sphere)

    return geometries
