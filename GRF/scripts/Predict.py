
import sys
sys.path.append('../../UnderPressure')

import models
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

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
subj = "S4"
mass = sub_mass[subj]
folder = "Male2MartialArtsKicks_c3d"
if wild[subj]:
    path = ROOT + subj + "/" + folder + "/preprocessed"
else:
    path = ROOT + subj + "/preprocessed"

print(path)
filepath = os.path.join(path, "*.pth")
files = glob.glob(filepath)

k=20


# checkpointname = 'pretrained_s7_noshape'
# checkpointname = 'baseline'
checkpointname = 'phys_grd_S5_0.002'
checkpointfile = '../checkpoint/' + checkpointname + '.tar'
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

checkpoint = torch.load(checkpointfile)
model = models.DeepNetwork(state_dict=checkpoint["model"]).eval()
print("Sucessfully loaded model.")

import time
from tqdm import tqdm

pbar = tqdm(files)
pbar.set_description("Predicting: %s"%subj)

for file in pbar:
    trial = os.path.splitext(file)[0].split(bar)[-1]
    ref_data = torch.load(file)
    poses = ref_data["poses"]
    trans = ref_data["trans"]

    with torch.no_grad():
        GRFs_pred = model.GRFs(poses.float().unsqueeze(0)).squeeze(0)

    if not wild[subj]:
        post_process_path = pred_path + checkpointname 
        if not os.path.exists(post_process_path):
            os.mkdir(post_process_path)
        output_w_prediction = os.path.join(post_process_path, trial + ".pth")

        weight = 9.81*mass

        output_pred = {}
        output_pred["GRF"] = ref_data["GRF"]
        output_pred["CoP"] = ref_data["CoP"]

        output_pred["prediction"] = GRFs_pred
        torch.save(output_pred, output_w_prediction)
    else:
        outputpath = pred_path_AMASS + checkpointname
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        output = os.path.join(outputpath, trial + ".pth")
        torch.save(GRFs_pred, output)

