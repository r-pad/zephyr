import numpy as np
import cv2
import os
from scipy.io import loadmat


'''
Load the rendered images and depths on specific view angles
'''
def loadGridData(grid_folder, grid_indices, obj_id, return_meta = False):
    filename_format = '%s/obj_%06d/' % (grid_folder, obj_id) + '{:04d}-{}.{}'

    images = []
    depths = []
    masks = []
    quats = []
    metas = []

    for j in grid_indices:
        # print(filename_format.format(j, 'color', 'png'))
        img = cv2.cvtColor(cv2.imread(filename_format.format(j, 'color', 'png')), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(filename_format.format(j, 'depth', 'png'), cv2.IMREAD_UNCHANGED)
        mask = np.bitwise_and((cv2.imread(filename_format.format(j, 'label', 'png'))==obj_id)[:,:,0],
                              depth > 0).astype(np.uint8)
        q = np.load(filename_format.format(j, 'trans', 'npy'))
        meta = loadmat(filename_format.format(j, "meta", 'mat'))

        images.append(img)
        depths.append(depth)
        masks.append(mask)
        quats.append(q)
        metas.append(meta)

    if return_meta:
        return images, depths, masks, quats, metas
    else:
        return images, depths, masks, quats

'''
project the model point cloud according to a specific viewpoint
'''
def projectModelPCKeypoints(model_pc, trans_mat, observed_depth, meta_data, eps=0.02):
    model_pc = transformPoints(trans_mat, model_pc)

    x, y, z = model_pc[:, 0], model_pc[:, 1], model_pc[:, 2]
    depth_map = z * meta_data['depth_scale']
    y_map = x * meta_data['fx'] / z + meta_data['cx']
    x_map = y * meta_data['fy'] / z + meta_data['cy']

    # print(trans_mat)
    # blend = np.zeros((480, 640, 3), dtype=np.uint8)
    # blend[x_map.astype(int), y_map.astype(int)] = 255
    # plt.figure(dpi=300)
    # plt.imshow(blend)
    # plt.show()
    # return

    observed_depth_map = observed_depth[x_map.astype(int), y_map.astype(int)]
    # print(observed_depth_map - depth_map/meta_data['depth_scale'])
    choose = np.absolute(observed_depth_map - depth_map/meta_data['depth_scale']) < eps
    x_map = x_map[choose]
    y_map = y_map[choose]

    keypoints = np.vstack([x_map, y_map]).T

    return keypoints, choose

'''
Batch process projectModelPCKeypoints
    The input model_pc should be in meters
    depth / meta['factor_depth'] shoudl convert the depth map into meters
'''
def bopProjectModelPC(model_pc, grid_indices, depths, metas, camera, save_path):
    model_idx = np.arange(len(model_pc))

    for i, gid in enumerate(grid_indices):
        depth = depths[i]
        meta = metas[i]

        depth = depth / meta['factor_depth'] # this converts the depth to meter
        trans_mat = meta['poses'].squeeze()

        keypoints, choose = projectModelPCKeypoints(
            model_pc[:, :3], trans_mat, observed_depth=depth, meta_data = camera, eps=0.02
        )
        pc_chose = model_pc[choose]
        id_chose = model_idx[choose]

        save_file = os.path.join(save_path, "%04d-grid.npz" % gid)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        np.savez(save_file, keypoints=keypoints, pc_chose=pc_chose, id_chose=id_chose)

'''
Load the results by bopProjectModelPC()
'''
def bopLoadGridKeypoints(data_path, grid_indices, leaf_size=7):
    keypoints_all = []
    pc_all = []
    kpt_idx_all = []
    for gid in grid_indices:
        data = np.load(os.path.join(data_path, "%04d-grid.npz" % (gid)))
        keypoints_all.append(data['keypoints'])
        pc_all.append(data['pc_chose'])
        kpt_idx_all.append(data['id_chose'])
    return keypoints_all, pc_all, kpt_idx_all

'''
Colorize the model points by the renders
'''
def getModelPointsCloud(model_points, model_normals, grid_images, keypoints_all, kpt_idx_all):
    model_colors = np.zeros(model_points.shape)
    model_counts = np.zeros([len(model_points), 1])
    for r_img, kps, idxs in zip(grid_images, keypoints_all, kpt_idx_all):
        kps = np.round(kps).astype(int)
        model_colors[idxs] += r_img[kps[:,0], kps[:,1]]/255.
        model_counts[idxs] += 1
    model_colors /= model_counts
    return model_points, model_normals, model_colors

def transformPoints(mat, pc):
    if pc.shape[1] == 3:
        shape_N3 = True
        pc = pc.T
    transformed = mat[:3, :3].dot(pc) + mat[:3, 3:4]
    if shape_N3:
        transformed = transformed.T
    return transformed
