
import numpy as np
import os
import torch

participants = {
    "s001" : "S1",
    "s002" : "S2",
    "s003" : "S3",
    "s004" : "S4",
    "s005" : "S5",
    "s006" : "S6",
    "s007" : "S7",
	}

# system = 'Windows'
system = 'Ubuntu'

motiontype = {
    'tree' : 'yoga',
    'treearms' : 'yoga',
    'chair' : 'yoga',
    'squat' : 'yoga',
    'worrier1' : 'warrior',
    'worrier2' : 'warrior',
    'sidestretch' : 'side_stretch',
    'dog' : 'hand',
    'jumpingjack' : 'jump',
    'walk' : 'walk',
    'walk_00': 'walk',
    'hopping' : 'hopping',
    'ballethighleg' : 'ballet_high',
    'balletsmalljump' : 'ballet_jump',
    'whirl' : 'dance',
    'lambadadance' : 'yoga',
    'taichi' : 'taichi',
    'step' : 'stairs',
    'tennisserve' : 'tennis',
    'tennisgroundstroke' : 'tennis',
    'soccerkick' : 'kicking',
    'idling' : 'idling',
    'idling_00' : 'idling',
    'static' : 'static',
    'ballet_high_leg' : 'ballet_high'
}

from scipy.spatial.transform import Rotation


def parse_motion_force(sourcemotion, contactdata, outputfile, mass: float):
    if os.path.exists(outputfile):
        print("File exists.. Skipping..")
        pass

    # load motion
    moshpp = np.load(sourcemotion, allow_pickle=True)
    # load force
    force_data = np.load(contactdata, allow_pickle=True)
    mocap = {}
    num_joints = 55
    num_body_joints = 22
    mocap["gender"] = moshpp["gender"]

    # load model file to remove pelvis offset from SMPL-X model
    modelpath = 'Visualization/models/smplx/' + str(mocap["gender"])
    # modelpath = '../../../Data/QTM_SOMA_MOSH/support_files/smplx/' + str(mocap["gender"])
    modelfile = os.path.join(modelpath, 'model.npz')
    modeldata = np.load(modelfile, allow_pickle=True)
    # print(modeldata.keys())
    # for k in modeldata.files: print(k)
    pelvis_offset = modeldata['J'][0]

    num_frames = min(len(moshpp['poses']), len(force_data.item()["CoP"]))
    mocap["angles"] = torch.reshape(torch.tensor(moshpp["poses"]), (len(moshpp['poses']), num_joints, 3))[:num_frames,:num_body_joints,:]
    # mocap["angles"] = torch.index_select(mocap["angles"], 2, torch.LongTensor([0,2,1]))
    
    mocap["trans"] = torch.tensor(moshpp["trans"]).unsqueeze(1)[:num_frames]+pelvis_offset
    # mocap["trans"] = torch.index_select(mocap["trans"], 2, torch.LongTensor([0,2,1]))
    mocap["shape"] = torch.tensor(moshpp["betas"]).unsqueeze(1).repeat(num_frames, 1, 3)
    mocap["framerate"] = float(moshpp["mocap_framerate"])
    
    contact = {}

    COP = force_data.item()["CoP"][:num_frames]
    GRF = force_data.item()["GRF"][:num_frames]

    rotate_z = mocap["angles"][:,0].clone()
    rotate = torch.zeros(num_frames, 3)
    rotate[:,2] = rotate_z[:,2]
    pelvis_rot = torch.tensor(Rotation.from_rotvec(rotate.numpy()).as_matrix())
    pelvis_t_project = mocap["trans"].clone()
    pelvis_t_project[:,:,2] = 0.0

    transformation_mat = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)


    transformation_mat[:, :3, :3] = pelvis_rot
    mocap["to_global_rot"] = pelvis_rot
    rotation_mat_inv = torch.inverse(mocap["to_global_rot"])
    transformation_mat[:, :3, 3] = pelvis_t_project.squeeze(1)
    mocap["to_global"] = transformation_mat # double tensor


    transformation_mat_inv = torch.inverse(transformation_mat)


    homo = torch.ones(num_frames, 2, 1)
    homo_COP = torch.cat((COP, homo), dim=-1)



    CoP_local = torch.matmul(transformation_mat_inv, homo_COP.transpose(-1, -2)).transpose(-1, -2)
    # GRF_local = torch.matmul(rotation_mat_inv, GRF.type('torch.DoubleTensor').transpose(-1, -2)).transpose(-1, -2)

    # shift CoP to projected pelvis=
    contact["CoP"] = CoP_local[:, :, :-1].type('torch.FloatTensor')
    contact["GRF"] = GRF.type('torch.FloatTensor')
    contact["mass"] = torch.Tensor([mass]).type('torch.FloatTensor')


    homo_pelvis_one = torch.ones(num_frames, 1, 1)
    homo_pelvis = torch.cat((mocap["trans"], homo_pelvis_one), dim=-1).type('torch.FloatTensor')
    pelvis_local = torch.matmul(transformation_mat_inv, homo_pelvis.transpose(-1, -2)).transpose(-1, -2)

    mocap["poses"] = torch.cat((pelvis_local[:, :, :-1], mocap["angles"]), dim=1).type('torch.FloatTensor')

    # Simulation
    z_true 	= mocap["poses"][:,0,2] # B, T
    seq_len = z_true.shape[0]
    z0      = z_true[0].clone() # Initial solution
    v0      = torch.zeros_like(z0)
    z, v, res_grf = [], [], []
    z.append(z0)
    v.append(v0)
    res_grf.append(contact["GRF"][0,:,-1].sum(-1))
    dt  = 1/90
    for f in range(seq_len-1):
        rgt     = 50 * (z_true[f+1] - z[-1]) - 12 * v[-1]
        vt      = v[-1] + (rgt - 1) * dt
        zt      = z[-1] + vt * dt
        v.append(vt)
        z.append(zt)
        res_grf.append(rgt)
    # z_sim   = torch.stack(z, dim=1)
    contact["GRF_phys"] = torch.stack(res_grf, dim=0)

    torch.save(mocap | contact, outputfile)


import torch
import os
import glob

import time
from tqdm import tqdm

sub_mass = {
    "s001" : 69.81,
    "s002" : 66.68,
    "s003" : 53.07,
    "s004" : 71.67,
    "s005" : 90.7,
    "s006" : 48.99,
    "s007" : 63.96,
    "AMASS" : 80.0,
	}

for participant in participants:
    print("Participant ID: " + participant)
    print("Mass: " + str(sub_mass[participant]) + "kg")
    Datapath = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/GroundLink/"
    # mocap has npz format
    inputMocap = Datapath + 'moshpp/' + participant
    inputContact = Datapath + 'force/' + participant


    datasetPath = '/mimer/NOBACKUP/groups/alvis_cvl/cuole/phys_grd/ProcessedData/'
    outputPath = datasetPath + participants[participant] + '/preprocessed'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    path = os.path.join(inputContact + '/*.npy')
    forcefiles = glob.glob(path)

    pbar = tqdm(forcefiles)
    pbar.set_description("Processing: %s"%participant)


    for forcefile in pbar:
        if system == 'Windows':
            bar = '\\'
        else:
            bar = '/'
        trial = os.path.splitext(forcefile)[0].split(bar)[-1]
        motion = trial[14:-2]
        if motiontype[motion] == 'ballet_high':
            continue
        if participant == 's001' and motion == 'idling':
            continue
        outputfile = outputPath + '/' + trial +'.pth'
        if os.path.exists(outputfile):
            print("Skipping: " + trial)
            continue
        else:
            sourcemotion = inputMocap + '/' + trial + "_stageii.npz"
            sourceforce = inputContact + '/' + trial + '.npy'
            if not os.path.exists(sourcemotion):
                print(motion)
                print("motion file not exists.. Skipping...")
            else:
                parse_motion_force(sourcemotion, sourceforce, outputfile, sub_mass[participant])

print("Processed ALL participants!")

