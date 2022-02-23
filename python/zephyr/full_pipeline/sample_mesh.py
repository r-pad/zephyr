import open3d as o3d
import numpy as np
import os, glob
import subprocess

def sampleMeshPointCloudYcb(input_root, output_root):
    input_meshes = glob.glob(os.path.join(input_root, "*"))
    input_meshes.sort()

    input_names = [p.split('/')[-1] for p in input_meshes]
    input_meshes = [os.path.join(p, 'textured.obj') for p in input_meshes]
    for i in range(len(input_names)):
        output_pcd = os.path.join(output_root, "%s.pcd" % input_names[i])
        sampleMeshPointCloud(input_meshes[i], output_pcd,
                         n_samples=100000, leaf_size=0.007, write_normals=True)
        output_ply = output_pcd[:-3] + 'ply'
        pcd = o3d.io.read_point_cloud(output_pcd)
        o3d.io.write_point_cloud(output_ply, pcd)
    return input_names

def sampleMeshPointCloud(inpuT_file, output_file, n_samples=None, leaf_size=None, write_normals=True, write_colors=False):
    cmd = ["pcl_mesh_sampling"]
    cmd += [inpuT_file]
    cmd += [output_file]

    args = []
    if n_samples is not None:
        args += ["-n_samples " + str(n_samples)]
    if leaf_size is not None:
        args += ["-leaf_size " + str(leaf_size)]
    if write_normals:
        args += ["-write_normals"]
    if write_colors:
        args += ['-write_colors']
    args += ["-no_vis_result"]
    cmd += args

    cmd = " ".join(cmd)

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if len(stderr) > 0:
        raise Exception(stderr)
