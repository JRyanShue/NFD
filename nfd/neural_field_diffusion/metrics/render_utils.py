import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import matplotlib

matplotlib.use("Agg")
import pyglet

pyglet.options['shadow_window'] = False

import time
from PIL import Image
import numpy as np
import pyrender
import trimesh
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
)
import pyrr
from matplotlib import pyplot as plt


def scale_to_unit_sphere(mesh, evaluate_metric=False):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    if evaluate_metric:
        vertices /= 2
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


SIZE = None


class Render:

    def __init__(self, size, camera_pose, background=None):
        self.size = size
        global SIZE
        SIZE = size
        self.camera_pose = camera_pose
        self.background = background

    def render(self,
               path,
               clean=True,
               intensity=3.0,
               mesh=None,
               only_render_images=False):
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = prepare_mesh(path, color=False, clean=clean)
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print("Error loading material!")

        triangle_id, normal_map, depth_image, p_image = None, None, None, None
        if not only_render_images:
            triangle_id, normal_map, depth_image, p_image = correct_normals(
                mesh, self.camera_pose, correct=True)
        mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        rendered_image, _ = pyrender_rendering(mesh1,
                                               viz=False,
                                               light=True,
                                               camera_pose=self.camera_pose,
                                               intensity=intensity,
                                               bg_color=self.background)

        return triangle_id, rendered_image, normal_map, depth_image, p_image

    def render_normal(self, path, clean=True, intensity=6.0, mesh=None):
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print("Error loading material!")

        triangle_id, normal_map, depth_image, p_image = correct_normals(
            mesh, self.camera_pose, correct=True)

        return normal_map, depth_image


def correct_normals(mesh, camera_pose, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    a, b, index_tri, sign, p_image = trimesh_ray_tracing(
        mesh, camera_pose, resolution=SIZE * 2, rayintersector=rayintersector)
    if correct:
        mesh.faces[index_tri[sign > 0]] = np.fliplr(
            mesh.faces[index_tri[sign > 0]])

    normalmap = render_normal_map(
        pyrender.Mesh.from_trimesh(mesh, smooth=False),
        camera_pose,
        SIZE,
        viz=False,
    )

    return b, normalmap, a, p_image


def init_light(scene, camera_pose, intensity=6.0):
    direc_l = DirectionalLight(color=np.ones(3), intensity=intensity)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=intensity,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    point_l = PointLight(color=np.ones(3), intensity=2 * intensity)
    direc_l_node = scene.add(direc_l, pose=camera_pose)
    point_l_node = scene.add(point_l, pose=camera_pose)
    spot_l_node = scene.add(spot_l, pose=camera_pose)


class CustomShaderCache:

    def __init__(self):
        self.program = None

    def get_program(self,
                    vertex_shader,
                    fragment_shader,
                    geometry_shader=None,
                    defines=None):
        if self.program is None:
            current_work_dir = os.path.dirname(__file__)
            print(current_work_dir)
            self.program = pyrender.shader_program.ShaderProgram(
                current_work_dir + "/shades/mesh.vert",
                current_work_dir + "/shades/mesh.frag",
                defines=defines)
        return self.program


def render_normal_map(mesh, camera_pose, size, viz=False):
    scene = pyrender.Scene(bg_color=(255, 255, 255))
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    renderer._renderer._program_cache = CustomShaderCache()

    normals, depth = renderer.render(scene)

    world_space_normals = normals / 255 * 2 - 1

    if viz:
        image = Image.fromarray(normals, "RGB")
        image.show()

    return world_space_normals


def pyrender_rendering(mesh,
                       camera_pose,
                       viz=False,
                       light=False,
                       intensity=3.0,
                       bg_color=None):
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.)
    camera = scene.add(camera, pose=camera_pose)
    # light
    if light:
        init_light(scene, camera_pose, intensity=intensity)

    scene.set_pose(camera, camera_pose)

    if light:
        color, depth = r.render(scene,
                                flags=pyrender.constants.RenderFlags.ALL_SOLID
                                | pyrender.constants.RenderFlags.FACE_NORMALS)
    else:
        color, depth = r.render(scene,
                                flags=pyrender.constants.RenderFlags.FLAT)

    return color, depth


def create_pose(eye):
    target = np.zeros(3)
    camera_pose = np.array(
        pyrr.Matrix44.look_at(eye=eye,
                              target=target,
                              up=np.array([0.0, 1.0, 0])).T)
    return np.linalg.inv(np.array(camera_pose))


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, rayintersector=None):
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    scene.camera_transform = M @ extra
    scene.camera.resolution = [resolution, resolution]
    scene.camera.fov = fov, fov
    origins, vectors, pixels = scene.camera_rays()

    index_tri, index_ray, points = rayintersector.intersects_id(
        origins, vectors, multiple_hits=False, return_locations=True)
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    sign = trimesh.util.diagonal_dot(mesh.face_normals[index_tri],
                                     vectors[index_ray])

    pixel_ray = pixels[index_ray]
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    b = np.ones(scene.camera.resolution, dtype=np.int32) * -1
    p_image = np.ones(
        [scene.camera.resolution[0], scene.camera.resolution[1], 3],
        dtype=np.float32) * -1

    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
    b[pixel_ray[:, 0], pixel_ray[:, 1]] = index_tri
    p_image[pixel_ray[:, 0], pixel_ray[:, 1]] = points

    return a, b, index_tri, sign, p_image
