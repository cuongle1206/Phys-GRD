# Specify motion and force, display in aitviewer

import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres


from aitviewer.renderables.lines import Lines
from aitviewer.renderables.arrows import Arrows


import math as m
import torch

import util

# Load File Paths
Basepath = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/datasets/GroundLink/moshpp/"
PredPath = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/phys_grd/ProcessedData/"
# participant = 's007'
# trial = 's007_20220705_tennisgroundstroke_1'
participant = 's004'
# trial = 's004_20220524_sidestretch_2'
trial = 's004_20220524_taichi_4'
threshold = 0.1
fps = 250.0
# ckp = 'pretrained_s7_noshape'
# ckp = 'noshape_s7_3e6_73_3e-6'
# ckp = 'baseline'
ckp = 'phys_grd_S5_0.002'

# End of Loading File Paths


Testing = False
sourcemotion = os.path.join(Basepath+participant, trial + '_stageii.npz')
if Testing:
    gt_file = os.path.join(PredPath+util.participants[participant]+'/test', trial+'.pth')
else:
    gt_file = os.path.join(PredPath+util.participants[participant]+'/preprocessed', trial+'.pth')
predicted = os.path.join(PredPath+util.participants[participant]+'/prediction/' + ckp, trial+'.pth')


if __name__ == "__main__":
    # Load an AMASS sequence and make sure it's sampled at 60 fps. 
    # This loads the SMPL-X model.
    # We set transparency to 0.5 and render the joint coordinates systems.
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    color_gt = (83 / 255, 189 / 255, 255 / 255, 1.0)
    color_pred = (255 / 255, 130 / 255, 53/255, 1.0)

    mesh = (102/255,102/255,102/255,0.5)
    fp_color = (127/255,127/255,128/255,1)
    seq_amass = SMPLSequence.from_amass(
        npz_data_path=sourcemotion,
        fps_out=fps,
        color=mesh,
        name=trial,
        show_joint_angles=True,
    )
    print(seq_amass.vertices.shape)
    print(seq_amass.joints.shape)
    print()
    torch.save([seq_amass.vertices, seq_amass.joints], "../mesh_s004_taichi.pt")


    # ptc_amass = PointClouds(seq_amass.vertices, position=np.array([-1.0, 0.0, 0.0]), color=c, z_up=True)

    # line_strip = util.get_fp()
    # line_renderable = Lines(line_strip, color = fp_color, mode="lines")

    # CoP, CoP_pred, GRF, GRF_pred = util.get_data_pred(gt_file, predicted, threshold)

    # arrow_renderables = Arrows(
    #             CoP.numpy(),
    #             CoP.numpy()+GRF.numpy(),
    #             color= color_gt,
    #             is_selectable=True,
    #         )
    
    # arrow_renderables_pred = Arrows(
    #             CoP_pred.numpy(),
    #             CoP_pred.numpy()+GRF_pred.numpy(),
    #             color= color_pred,
    #             is_selectable=True,
    #         )
    

    # # Display in the viewer.
    # v = Viewer()
    # v.run_animations = True
    # v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    # v.scene.add(seq_amass)

    # v.scene.add(line_renderable)
    # v.scene.add(arrow_renderables)
    # v.scene.add(arrow_renderables_pred)

    # v.run()
