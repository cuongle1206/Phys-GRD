
import sys
sys.path.append('./UnderPressure')

import models, metrics
import os, shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

shutil.rmtree('./animations/tmp_imgs')
os.makedirs('./animations/tmp_imgs', exist_ok=True)

wild = {
    "S1" : False,
    "S2" : False,
    "S3" : False,
    "S4" : False,
    "S5" : False,
    "S6" : False,
    "S7" : False,
    "AMASS": True,
    "w074" : True,
	}


sub_mass = {
    "S1" : 69.81,
    "S2" : 66.68,
    "S3" : 53.07,
    "S4" : 71.67,
    "S5" : 90.7,
    "S6" : 48.99,
    "S7" : 63.96,
    "AMASS" : 80.0,
	}


# system = 'Windows'
system = 'Ubuntu'

save_img = False
save_high_res_img = True

ROOT = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/phys_grd/ProcessedData/"
subj = "S5"
mass = sub_mass[subj]
folder = "Male2MartialArtsKicks_c3d"
if wild[subj]:
    path = ROOT + subj + "/" + folder + "/preprocessed"
else:
    path = ROOT + subj + "/preprocessed"

print(path)
filepath = os.path.join(path, "*.pth")
files = sorted(glob.glob(filepath))

k=20


# checkpointname = 'pretrained_s7_noshape' 
checkpointname = 'baseline'
# checkpointname = 'phys_grd_S5_0.001'
# checkpointname = 'phys_grd_S5_0.002'
# checkpointname = 'phys_grd_S5_0.003'
# checkpointname = 'phys_grd_S5_0.004'
checkpointname = 'phys_grd_S5_0.005'
checkpointfile = './GRF/checkpoint/' + checkpointname + '.tar'
pred_path = ROOT + subj + "/prediction/"
if not os.path.exists(pred_path):
    os.mkdir(pred_path)
if wild[subj]:
    pred_path_AMASS = ROOT + subj + "/" + folder + "/prediction/"
    if not os.path.exists(pred_path_AMASS):
        os.mkdir(pred_path_AMASS)

if system == 'Windows':
    bar = '\\'
else:
    bar = '/'

bl_ckp = torch.load('./GRF/checkpoint/baseline.tar')
baseline_model = models.DeepNetwork(state_dict=bl_ckp["model"]).eval()

checkpoint = torch.load(checkpointfile)
model = models.DeepNetwork(state_dict=checkpoint["model"]).eval()
print("Sucessfully loaded model.")

import time
from tqdm import tqdm

pbar = tqdm(files)
pbar.set_description("Predicting: %s"%subj)

vGRF_L, vGRF_R, vRPE = [], [], []
recon_metric = torch.nn.MSELoss(reduction='mean')


j = 0

names = []
for file in pbar:
    trial = os.path.splitext(file)[0].split(bar)[-1]
    names.append(trial)
    ref_data = torch.load(file)
    poses = ref_data["poses"]
    trans = ref_data["trans"]
    cop   = ref_data["CoP"]
    grf   = ref_data["GRF"]
    mass  = ref_data["mass"]
    weight = (9.81 * mass)
    grf_L = grf[:,0,-1]
    grf_R = grf[:,1,-1]
    grf_sum = grf[...,-1].sum(-1) * (1e3/weight)

    # fig     = plt.figure()
    # ax1     = fig.add_subplot(211)
    # ax2     = fig.add_subplot(212)

    fig     = plt.figure(figsize=(4,12))
    ax1     = fig.add_subplot(311)
    ax2     = fig.add_subplot(312)
    ax3     = fig.add_subplot(313)

    with torch.no_grad():
        GL_pred = baseline_model.GRFs(poses.float().unsqueeze(0)).squeeze(0)
        GRFs_pred = model.GRFs(poses.float().unsqueeze(0)).squeeze(0)
        
        # Simulation
        z_true 	= poses[:,0,2] # B, T
        seq_len = z_true.shape[0]
        z0      = z_true[0].clone() # Initial solution
        v0      = torch.zeros_like(z0)
        pred_grf_sum = GRFs_pred[...,-1].sum(-1)
        # z, v, res_grf = [], [], []
        # z.append(z0)
        # v.append(v0)
        # res_grf.append(grf_sum[0])
        # dt      = 1/90
        # for f in range(seq_len-1):
        #     # rgt     = 50 * (z_true[f+1] - z[-1]) - 12 * v[-1]
        #     rgt     = pred_grf_sum[f]
        #     vt      = v[-1] + (rgt - 1) * dt
        #     zt      = z[-1] + vt * dt
        #     res_grf.append(rgt)
        #     # v.append(vt)
        #     # z.append(zt)
        #     if f % 125 == 0:
        #         v.append(v0)
        #         z.append(z_true[f])
        #     else:
        #         v.append(vt)
        #         z.append(zt)
        # z_sim   = torch.stack(z, dim=0)

    vGRF_L.append(metrics._mse_loss(GRFs_pred[:,0,-1], grf[:,0,-1]))
    vGRF_R.append(metrics._mse_loss(GRFs_pred[:,1,-1], grf[:,1,-1]))
    # vRPE.append(recon_metric(z_sim, z_true)*1e3)

    # ax1.set_xlim([0, seq_len])
    # ax1.set_ylim([-0., 3.0])
    # ax1.set_title("vGRF")
    # ax1.set_ylabel("N/kg")
    # ax1.set_xticks([])
    # ax1.plot(grf_sum)
    # ax1.plot(res_grf)

    # ax2.set_xlim([0, seq_len])
    # ax2.set_ylim([0., 2])
    # ax2.set_title("z position")
    # ax2.set_ylabel("m")
    # ax2.set_xlabel("t")
    # ax2.plot(z_true, label="force plate")
    # ax2.plot(z_sim, label="physics simulation")
    # ax2.legend(loc='lower right')

    ax1.set_xlim([0, seq_len])
    ax1.set_ylim([-0., 3.0])
    ax1.set_title("vGRF Left")
    ax1.set_ylabel("N/kg")
    ax1.set_xticks([])
    ax1.plot(grf_L, '-', label="force plate")
    ax1.plot(GL_pred[:,0,-1], label="GroundLink")
    ax1.plot(GRFs_pred[:,0,-1], label="Ours")
    ax1.legend(loc='upper right')

    ax2.set_ylim([-0., 3.0])
    ax2.set_xlim([0, seq_len])
    ax2.set_title("vGRF Right")
    ax2.set_ylabel("N/kg")
    ax2.set_xticks([])
    ax2.plot(grf_R, '-')
    ax2.plot(GL_pred[:,1,-1])
    ax2.plot(GRFs_pred[:,1,-1])

    ax3.set_xlim([0, seq_len])
    ax3.set_ylim([-0., 3.0])
    ax3.set_title("vGRF Total")
    ax3.set_ylabel("N/kg")
    ax3.set_xlabel("t")
    ax3.plot(grf_sum, '-')
    ax3.plot(GL_pred[:,:,-1].sum(-1))
    ax3.plot(GRFs_pred[:,:,-1].sum(-1))

    plt.savefig(f'./animations/tmp_imgs/frame_{j}.png', transparent=True, dpi=300, format='png', facecolor='white', bbox_inches='tight')
    j+=1
    plt.close()

for i in range(len(names)):
    # print(f"{names[i]} " + " vGRF_L = {:.2f}, vGRF_R = {:.2f}, vRPE = {:.2f} ".format(vGRF_L[i], vGRF_R[i], vRPE[i]))
    print(f"{names[i]} " + str(i) + " vGRF_L = {:.2f}, vGRF_R = {:.2f}".format(vGRF_L[i], vGRF_R[i]))

print("vGRF_L = {:.2f}, vGRF_R = {:.2f}  ".format(torch.stack(vGRF_L).mean().item(), torch.stack(vGRF_R).mean().item()))
# print("vRPE = {:.2f}, ".format(torch.stack(vRPE).mean().item()))
# print("vRPE = {:.2f}, ".format(torch.stack(vRPE).std().item()))


