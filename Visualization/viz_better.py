import os, shutil, sys
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
sys.path.append('./Visualization')
import pickle
import smplx
import pyrender
from pyrender.constants import RenderFlags as RF
import trimesh
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import util
import numpy as np
from pytorch3d import transforms

def add_ground(ground_plane=None, length=20.0, color0=[0.8, 0.8, 0.8], color1=[0.99, 0.99, 0.99], tile_width=2, xyz_orig=None, alpha=1.0):
    '''
    If ground_plane is none just places at origin with +z up.
    If ground_plane is given (a, b, c, d) where a,b,c is the normal, then this is rendered. To more accurately place the floor
    provid an xyz_orig = [x,y,z] that we expect to be near the point of focus.
    '''
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = int(length / tile_width)
    vertices = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            start_loc = [-radius + j*tile_width, radius - i*tile_width]
            cur_verts = np.array([[start_loc[0], start_loc[1], 0.0],
                                    [start_loc[0], start_loc[1]-tile_width, 0.0],
                                    [start_loc[0]+tile_width, start_loc[1]-tile_width, 0.0],
                                    [start_loc[0]+tile_width, start_loc[1], 0.0]])
            cur_faces = np.array([[0, 1, 3], [1, 2, 3]])
            cur_faces += 4 * (i*num_cols + j) # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_face_colors = np.array([cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            face_colors.append(cur_face_colors)

    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    face_colors = np.concatenate(face_colors, axis=0)

    if ground_plane is not None:            
        # compute transform between identity floor and passed in floor
        a, b, c, d = ground_plane
        # rotation
        old_normal = np.array([0.0, 0.0, 1.0])
        new_normal = np.array([a, b, c])
        new_normal = new_normal / np.linalg.norm(new_normal)
        v = np.cross(old_normal, new_normal)
        ang_sin = np.linalg.norm(v)
        ang_cos = np.dot(old_normal, new_normal)
        skew_v = np.array([[0.0, -v[2], v[1]],
                        [v[2], 0.0, -v[0]],
                        [-v[1], v[0], 0.0]])
        R = np.eye(3) +  skew_v + np.matmul(skew_v, skew_v)*((1.0 - ang_cos) / (ang_sin**2))
        # translation
        # project point of focus onto plane
        if xyz_orig is None:
            xyz_orig = np.array([0.0, 0.0, 0.0])
        # project origin onto plane
        plane_normal = np.array([a, b, c])
        plane_off = d
        direction = -plane_normal
        s = (plane_off - np.dot(plane_normal, xyz_orig)) / np.dot(plane_normal, direction)
        itsct_pt = xyz_orig + s*direction
        t = itsct_pt

        # transform floor
        vertices = np.dot(R, vertices.T).T + t.reshape((1, 3))
    vertices += [0.,0.,0.02]
    ground_tri = trimesh.creation.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors, process=False)
    ground_mesh = pyrender.Mesh.from_trimesh(ground_tri, smooth=False)
    return ground_mesh


if __name__ == "__main__":
    
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load File Paths
    Basepath        = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/GroundLink/moshpp/"
    PredPath        = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/phys_grd/ProcessedData/"
    participant     = 's004'
    # trial           = 's004_20220524_sidestretch_2'
    trial = 's004_20220524_taichi_4'
    threshold       = 0.1
    fps             = 250.0
    ckp             = 'phys_grd_S5_0.002'

    Testing = False
    sourcemotion = os.path.join(Basepath+participant, trial + '_stageii.npz')
    gt_file = os.path.join(PredPath+util.participants[participant]+'/preprocessed', trial+'.pth')
    predicted = os.path.join(PredPath+util.participants[participant]+'/prediction/' + ckp, trial+'.pth')
    gt_data = torch.load(gt_file)
    prediction = torch.load(predicted)
    motion = np.load(sourcemotion)
    
    print(prediction.keys())
    grf_L = prediction["prediction"][:,0,3:] * 0.75
    grf_R = prediction["prediction"][:,1,3:] * 0.75

    smpl_trans = torch.from_numpy(motion["trans"]).float().to(device)
    smpl_poses = torch.from_numpy(motion["poses"]).float().to(device)
    smpl_betas = torch.from_numpy(motion["betas"]).float().to(device)
    smpl_gender = motion["gender"].item()

    camera          = pyrender.PerspectiveCamera(yfov=np.pi/5.0, aspectRatio=3.0)
    camera_rot      = transforms.euler_angles_to_matrix(torch.deg2rad(torch.Tensor([0, -90, -90])), 'XYZ')
    camera_trs      = torch.Tensor([[-6.0, -0.8, 1.1]]).T
    camera_pose     = torch.cat((camera_rot, camera_trs), axis=-1)
    camera_pose     = torch.cat((camera_pose, torch.Tensor([[0,0,0,1]])), axis=0)
    
    light1          = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light1_pose     = camera_pose
    
    light2          = pyrender.DirectionalLight(color=np.ones(3), intensity=1.6)
    light2_rot      = transforms.euler_angles_to_matrix(torch.deg2rad(torch.Tensor([0, 0, 0])), 'XYZ')
    light2_trs      = torch.Tensor([[-0.1, -2.0, 4.0]]).T
    light2_pose     = torch.cat((light2_rot, light2_trs), axis=-1)
    light2_pose     = torch.cat((light2_pose, torch.Tensor([[0,0,0,1]])), axis=0)

    smpl_model      = smplx.create(model_path = "./Visualization/models",
                                   model_type = "smplx",
                                   gender = "neutral",
                                   use_face_contour = False,
                                   num_betas = 16,
                                   num_expression_coeffs = 10,
                                   ext = "npz").to(device)

    # Create the figure
    fig             = plt.figure(figsize=(21,9),frameon=False)
    ax              = fig.add_subplot()             # main axis
    scene           = pyrender.Scene(bg_color=[1.0, 1.0, 1.0], ambient_light=(0.3, 0.3, 0.3))
    ground_mesh     = add_ground(length=25.0, color0=[1.0, 1.0, 1.0], color1=[0.8, 0.8, 0.8])
    scene.add(camera, pose=camera_pose)
    scene.add(light1, pose=light1_pose)
    scene.add(light2, pose=light2_pose)
    scene.add(ground_mesh)

    sm              = trimesh.creation.uv_sphere(radius=0.012)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]

    result2 = torch.load("mesh_s004_taichi.pt")
    result1 = torch.load("mesh_s004.pt")
    
    for ri, result in enumerate([result1, result2]):
        
        vertices = result[0]
        joints = result[1]
        print(ri)

        if ri == 0: frames = [0, 300, 600]
        if ri == 1: frames = [1000, 1300, 1600]

        for i, f in enumerate(frames):
            vertex_colors   = np.ones([vertices.shape[1], 4]) * [0.65098039,  0.64117647,  0.85882353, 0.7 + i*0.15]
            tri_mesh        = trimesh.Trimesh(vertices[f]+[-i*0.2, -ri*3, 0], smpl_model.faces, vertex_colors=vertex_colors)
            mesh            = pyrender.Mesh.from_trimesh(tri_mesh)
            scene.add(mesh)

            pose = joints[f]
            right_contact = pose[[8,11],:].mean(0) +[-i*0.28,-ri*3,0]
            left_contact = pose[[7,10],:].mean(0) +[-i*0.28,-ri*3,0]
            
            vL  = np.stack([left_contact, (grf_L[f] + left_contact)])
            vR  = np.stack([right_contact, (grf_R[f] + right_contact)])

            arrow_mesh = trimesh.creation.cylinder(radius=0.01, segment=vL)
            arrow_mesh.visual.vertex_colors  = [0, 255, 0, 100 + i*50]
            arrow_pyrender_mesh = pyrender.Mesh.from_trimesh(arrow_mesh)
            scene.add(arrow_pyrender_mesh)

            arrow_mesh = trimesh.creation.cylinder(radius=0.01, segment=vR)
            arrow_mesh.visual.vertex_colors  = [0, 255, 0, 100 + i*50]
            arrow_pyrender_mesh = pyrender.Mesh.from_trimesh(arrow_mesh)
            scene.add(arrow_pyrender_mesh)

        for i, j in enumerate(range(frames[0],frames[1],20)):
            pose = joints[j]
            joints_temp     = pose[0,:]+[-i*0.07,-ri*3,0]
            tfs             = np.tile(np.eye(4), (joints_temp.shape[0], 1, 1))
            tfs[:, :3, 3]   = joints_temp 
            joints_pcl      = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

    r               = pyrender.OffscreenRenderer(2100, 900)
    color, _        = r.render(scene, flags = RF.SHADOWS_DIRECTIONAL | RF.ALL_SOLID)
    ax.imshow(color)

    plt.savefig(f'animations/Teaser.png', transparent=True, dpi=500, format='png', facecolor='white',
                bbox_inches=matplotlib.transforms.Bbox(np.array([[3.5, 2.2],[18.6, 5.8]])))

    plt.close('all')
