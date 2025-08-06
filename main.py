
import os, shutil, sys
sys.path.append('./UnderPressure')
# UnderPressure NN related
import anim, metrics, models, util
from data import TOPOLOGY, Contacts, Dataset

# Python
import math, time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# PyTorch
import torch
from torch.nn.utils.rnn import pad_sequence
import smplx

MIRROR_LUT = TOPOLOGY.lut(TOPOLOGY.mirrored())
training_loss = []
validation_loss = []
test_loss = []

def split(dataset, ratio, window_length):
    # select random non-overlapping windows for validation set
    nframes = sum(item["poses"].shape[-3] for item in dataset)
    valid_nwindows =  int((1 - ratio) * nframes) // window_length + 1
    windows = []
    for index, item in enumerate(dataset):
        starts = torch.arange(0, item["poses"].shape[-3] - 2 * window_length + 1, window_length)
        indices = torch.full_like(starts, index)
        windows.append(torch.stack([indices, starts], dim=-1))
    windows = torch.cat(windows)
    windows = windows[torch.randperm(windows.shape[0])[:valid_nwindows]]

    # split according to selected windows
    valid_items, train_items = [], []
    for index, item in enumerate(dataset):
        valid_starts = windows[windows[:, 0] == index, 1].sort()[0]
        starts = torch.cat([valid_starts, torch.as_tensor([0]), valid_starts + window_length])
        stops = torch.cat([valid_starts + window_length, valid_starts, torch.as_tensor([item["poses"].shape[-3]])])
        items = dataset.slices(index, starts, stops)
        train_items += [item for item in items[len(valid_starts):] if item["poses"].shape[-3] > 0]
        valid_items += items[:len(valid_starts)]
    return Dataset(train_items), Dataset(valid_items)

def prepare(split_ratio, sequence_length, sequence_overlap, train_ids, test_id):
	# split
    dataset = Dataset.trainset(train_ids=train_ids)["poses", "shape", "CoP", "GRF", "GRF_phys", "mass"]
    trainset, validset = split(dataset, split_ratio, sequence_length)
    
    # test
    testset = Dataset.testset(test_id=test_id)["poses", "shape", "CoP", "GRF", "GRF_phys","mass"]
    # slice into overlapping windows
    if set(a.shape[-3] for a in trainset["poses"]) != {sequence_length}:
        trainset = trainset.windowed(sequence_length, sequence_overlap)
    if set(a.shape[-3] for a in validset["poses"]) != {sequence_length}:
        validset = validset.windowed(sequence_length, 0)
    # if set(a.shape[-3] for a in testset["poses"]) != {sequence_length}:
    #     testset = testset.windowed(sequence_length, 0)
    return trainset, validset, testset

def rnd_transform(positions, forces):
	bs, device = positions.shape[0], positions.device
	
	# Mirrorring
	mirror = torch.rand(bs, device=device) < 0.5 # N

	positions_mirrored = positions.clone()

	positions_mirrored[mirror] = positions[mirror][..., MIRROR_LUT, :]
	forces_mirrored = forces.clone()
	forces_mirrored[mirror] = forces[mirror][..., [1, 0], :]
	
	return positions_mirrored, forces_mirrored	# N x F x J x 3, N x F x 2 x 16

class Trainer(util.Timeline):
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        # model and optimiser
        self.model = models.DeepNetwork()
        self.model = self.model.initialize().to(self.device)

        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=kwargs["learning_rate"])
        self.mse_weight = kwargs["mse_weight"]
        self.phys_weight = kwargs["phys_weight"]
        self.recon_weight = kwargs["recon_weight"]

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")

            if kwargs["test_subject"] == "S1": train_ids   = ["S2","S3","S4","S5","S6","S7"]
            if kwargs["test_subject"] == "S2": train_ids   = ["S1","S3","S4","S5","S6","S7"]
            if kwargs["test_subject"] == "S3": train_ids   = ["S1","S2","S4","S5","S6","S7"]
            if kwargs["test_subject"] == "S4": train_ids   = ["S1","S2","S3","S5","S6","S7"]
            if kwargs["test_subject"] == "S5": train_ids   = ["S1","S2","S3","S4","S6","S7"]
            if kwargs["test_subject"] == "S6": train_ids   = ["S1","S2","S3","S4","S5","S7"]
            if kwargs["test_subject"] == "S7": train_ids   = ["S1","S2","S3","S4","S5","S6"]
            test_id     = [kwargs["test_subject"]]
            trainset, validset, testset = prepare(kwargs["split_ratio"], kwargs["sequence_length"], kwargs["sequence_overlap"], train_ids, test_id)
            torch.save(trainset, os.path.join(folder_path, 'trainset.pt'))
            torch.save(validset, os.path.join(folder_path, 'validset.pt'))
            torch.save(testset, os.path.join(folder_path, 'testset.pt'))
            print(f"Trainset, validset and testset saved to '{folder_path}'")
        else:
            print(f"Folder '{folder_path}' already exists.")
            trainset = torch.load(os.path.join(folder_path, 'trainset.pt'))
            validset = torch.load(os.path.join(folder_path, 'validset.pt'))
            testset  = torch.load(os.path.join(folder_path, 'testset.pt'))
            print(f"Trainset, validset and testset loaded from '{folder_path}'")
        print(f"Trainset len: {len(trainset)}")
        print(f"Validset len: {len(validset)}")
        print(f"Testset len: {len(testset)}")

        dataloader = trainset.dataloader(
			batch_size = kwargs["batch_size"],
			shuffle = True,
			device = self.device,
		)
        
        self.validset = dict(poses = torch.stack(list(validset["poses"])),
                             cop = torch.stack(list(validset["CoP"])),
                             grf = torch.stack(list(validset["GRF"])),
                             mass = torch.stack(list(validset["mass"])))
        self.testset  = testset
		
		# logging support
        self.ckp = kwargs["ckp"]
		
		# instanciate timeline
        num_epochs = int(kwargs["iterations"] / len(dataloader) + 0.5)
        super().__init__(dataloader, num_epochs, *[
            util.Schedule(period=100,	fn=self._losses_logging),	# log loss values every X batches
            util.Schedule(period=1000,	fn=self._validation),		# validation every X batches
            util.Schedule(period=1000,	fn=self._test),		# test every X batches
		])

        self.recon_metric = torch.nn.MSELoss(reduction='none')
        # self.recon_metric = torch.nn.L1Loss(reduction='none')
        
        self.dt     = kwargs["step_size"] # Simulation rate

    def iteration(self, batch):
		# Modified for GroundLink
        poses, cop, grf, grf_phys, mass = batch["poses"], batch["CoP"], batch["GRF"], batch["GRF_phys"], batch["mass"]
        
        seq_len = poses.shape[1]
        weight  = (9.81 * mass).repeat(1, seq_len)

        contact_cop_grf = torch.cat((cop, grf*(1e3/weight).unsqueeze(-1).unsqueeze(-1)),3)
        poses, forces_target = rnd_transform(poses.float(), contact_cop_grf)

		# make predictions and compute loss
        contact_pred = self.model.GRFs(poses)
        
        self.mse = metrics._mse_loss(contact_pred, forces_target)
        # self.mse = metrics._mse_loss(contact_pred, contact_cop_grf)

		# # Simulation
        # z_true 	= poses[:,:,0,2] # B, T
        # _, seq_len = z_true.shape
        # z0      = z_true[:,0].clone() # Initial solution
        # v0      = torch.zeros_like(z0)
        # z, v, res_grf = [], [], []
        # z.append(z0)
        # v.append(v0)
        # res_grf.append(forces_target[:,0,:,-1].sum(-1))
        # for f in range(seq_len-1):
        #     rgt     = self.model.kp * (z_true[:,f+1] - z[-1]) - self.model.kd * v[-1]
        #     vt      = v[-1] + (rgt - 1) * self.dt
        #     zt      = z[-1] + vt * self.dt
        #     v.append(vt)
        #     z.append(zt)
        #     res_grf.append(rgt)
        # z_sim   = torch.stack(z, dim=1)
        # phys_grf = torch.stack(res_grf, dim=1)

        body_grf_sum = contact_pred[...,-1].sum(-1) # normalized by subject's weight
        self.phys_loss = metrics._mse_loss(body_grf_sum, grf_phys) # Physics-informed loss
        # self.recon  = self.recon_metric(z_sim, z_true).sum(1).mean() # Reconstruction loss
        loss        = (self.mse_weight * self.mse + self.phys_weight * self.phys_loss) # Total loss

		# optimize
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        torch.cuda.empty_cache()
    
    def _losses_logging(self):
        item, epoch = self.item + 1, self.epoch + 1
        print("[{}/{}][{}/{}]   MSE = {:.5e}, Phys = {:.5e}".format(item, self.nitems, epoch, self.nepochs,
                                                                                    self.mse_weight * self.mse,
                                                                                    self.phys_weight * self.phys_loss))
        training_loss.append("[{}/{}][{}/{}]   MSE = {:.5e}\n".format(item, self.nitems, epoch, self.nepochs, self.mse))
	
    def _validation(self):
        print("Validation #{}".format(self.iter))
		
		# Modified for GroundLink
		# Make predictions
        with torch.no_grad():
            contact_pred = []
            for poses in self.validset["poses"].split(128):
                contact_pred.append(self.model.GRFs(poses.float().to(self.device)).detach().cpu())
            forces_pred = torch.cat(contact_pred)
            
            mass = self.validset["mass"]
            weight  = (9.81 * mass).repeat(1, 140)

            target = torch.cat((self.validset["cop"], self.validset["grf"]*(1e3/weight).unsqueeze(-1).unsqueeze(-1)),3)
            rmse = metrics.RMSE(forces_pred, target=target).item()
            mse = metrics._mse_loss(forces_pred, target)
            
            torch.save(dict(model=self.model.state_dict()), self.ckp)
            print("RMSE = " + str(rmse))
            print("MSE = " + str(mse))
            validation_loss.append("Validation #{}   RMSE = {}    MSE = {}\n".format(self.iter, rmse, mse))

    def _test(self):
        print("Test #{}".format(self.iter))
		
        vGRF_L, vGRF_R, vRPE = [], [], []
        recon_metric = torch.nn.MSELoss(reduction='mean')
        with torch.no_grad():
            for i, sample in enumerate(list(self.testset)):
                poses = sample["poses"].to(self.device)
                # cop   = sample["CoP"]
                grf   = sample["GRF"].to(self.device)
                mass  = sample["mass"].to(self.device)
                weight  = (9.81 * mass).repeat(poses.shape[0])
                GRFs_pred = self.model.GRFs(poses.float().unsqueeze(0)).squeeze(0)

                # Simulation
                z_true 	= poses[:,0,2] # B, T
                seq_len = z_true.shape[0]
                z0      = z_true[0].clone() # Initial solution
                v0      = torch.zeros_like(z0)
                z, v    = [], []
                z.append(z0)
                v.append(v0)
                pred_grf_sum = GRFs_pred[...,-1].sum(-1)
                for f in range(seq_len-1):
                    rgt     = pred_grf_sum[f]
                    vt      = v[-1] + (rgt - 1) * self.dt
                    zt      = z[-1] + vt * self.dt
                    if f % 250 == 0:
                        v.append(v0)
                        z.append(z_true[f])
                    else:
                        v.append(vt)
                        z.append(zt)
                z_sim   = torch.stack(z, dim=0)

                vGRF_L.append(metrics._mse_loss(GRFs_pred[:,0,-1], grf[:,0,-1] * (1e3/weight)))
                vGRF_R.append(metrics._mse_loss(GRFs_pred[:,1,-1], grf[:,1,-1] * (1e3/weight)))
                vRPE.append(recon_metric(z_sim, z_true)*1e3)

            print("vGRF_L = {:.2f}, vGRF_R = {:.2f}  ".format(torch.stack(vGRF_L).mean().item(),
                                                                            torch.stack(vGRF_R).mean().item(),))
            print("vRPE = {:.2f}, ".format(torch.stack(vRPE).mean().item()))

def vis_sample(sample_id):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset  = torch.load(os.path.join(folder_path, 'testset.pt'))

    body_pose = testset["poses"][sample_id].float().to(device)
    body_shape = testset["shape"][sample_id].float().to(device)
    body_cop = testset["CoP"][sample_id]
    body_grf = testset["GRF"][sample_id]
    body_mass = testset["mass"][sample_id]
    body_weight = (9.81 * body_mass)/1e3
    body_grf_sum = body_grf.sum(1)
    seq_len = body_pose.shape[0]
    smpl_model  = smplx.create(model_path = "./Visualization/models",
                               model_type = "smpl",
                               gender = "neutral",
                               use_face_contour = False,
                               num_betas = 16,
                               num_expression_coeffs = 10,
                               ext = "npz").to(device)
    pad         = torch.zeros(seq_len, 2, 3, device=device)
    output      = smpl_model(betas=body_shape[:,:,0],
                            body_pose=torch.cat((body_pose[:,2:,:],pad), dim=1),
                            global_orient=body_pose[:,1:2,:],
                            transl=body_pose[:,0,:],
                            return_verts=True)
    vertices    = output.vertices.detach().cpu().numpy().squeeze()
    joints      = output.joints.detach().cpu().numpy().squeeze()
    # print('Vertices shape =', vertices.shape)
    # print('Joints shape =', joints.shape)

    fig     = plt.figure(figsize=(12,6))
    ax1     = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=0., azim=-90, vertical_axis='z')
    ax2     = fig.add_subplot(222)
    ax3     = fig.add_subplot(224)
    frames  = []
    parent  = [0,1,4,7,  0,2,5,8,  0,3,6, 9,12,  9,13,16,18,  9,14,17,19]
    child   = [1,4,7,10, 2,5,8,11, 3,6,9, 12,15, 13,16,18,20, 14,17,19,21]

    dt      = 1/90
    # g       = torch.Tensor([1]) # normalized
    z_true 	= output.joints.detach().cpu()[:,0,2]
    print(z_true)
    z, v, a = [], [], []
    z0      = z_true[0].clone()
    v0      = torch.zeros_like(z0)
    a0      = torch.zeros_like(z0)
    z.append(z0)
    v.append(v0)
    a.append(a0) 
    for f in range(seq_len-1):
        at = 50*(z_true[f+1] - z[-1]) - 12*v[-1]
        vt = v[-1] + (at - 1) * dt
        zt = z[-1] + vt * dt
        a.append(at)
        v.append(vt)
        z.append(zt)
    a_plot = torch.stack(a, dim=0)
    v_plot = torch.stack(v, dim=0)
    z_plot = torch.stack(z, dim=0)

    ax2.set_xlim([0, seq_len])
    ax2.set_ylim([-0., 3.0])
    ax2.plot(body_grf_sum[:,2]/body_weight)
    ax2.plot(a_plot)

    ax3.set_xlim([0, seq_len])
    ax3.set_ylim([-0., 2])
    ax3.plot(z_true)
    ax3.plot(z_plot)
    print(torch.norm((z_plot - z_true)))
    plt.savefig(f'./animations/tmp_imgs/frame.png', transparent=True, dpi=100, format='png', facecolor='white', bbox_inches='tight')
    plt.close()





if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-device", default="cuda", type=str,		help="Device used to run training ................. default: cuda")
    parser.add_argument("-test_subject", default="S5", type=str,    help="Test subject ID ............................. default: S5")
    parser.add_argument("-learning_rate", default=3e-5, type=float, help="Adam optimisation algorithm learning rate ... default: 3e-5")
    parser.add_argument("-mse_weight", default=2e-3, type=float,	help="MSE loss weight ............................. default: 0.002")
    parser.add_argument("-phys_weight", default=2e-3, type=float,	help="Phys loss weight ............................ default: 0.002")
    parser.add_argument("-recon_weight", default=1e-5, type=float,	help="Recon loss weight ........................... default: 0.001")
    parser.add_argument("-batch_size", default=64, type=int,		help="Batch size .................................. default: 64")
    parser.add_argument("-iterations", default=1e5, type=int,		help="Number of training iterations ............... default: 1e8")
    parser.add_argument("-split_ratio", default=0.7, type=float,	help="Train/Validation split ratio ................ default: 0.9")
    parser.add_argument("-sequence_length", default=140, type=int,	help="Training sequences length ................... default: 240")
    parser.add_argument("-sequence_overlap", default=139, type=int,	help="Training sequences overlap .................. default: 239")
    parser.add_argument("-step_size", default=1/90, type=float,	    help="Simulation step size ........................ default: 1/60")

    args = parser.parse_args()
    checkpointpath = './GRF/checkpoint'
    checkpointname = "phys_grd_" + args.test_subject + "_" + str(args.phys_weight)
    checkpoint = os.path.join(checkpointpath, checkpointname+'.tar')
    parser.add_argument("-ckp", default=checkpoint, type=Path,		help="Path to make checkpoint during training ..... default: 'checkpoint.tar'")

    util.seed_everything(42)
    folder_path = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/phys_grd/dataset_" + args.test_subject
    # Trainer(**vars(parser.parse_args()))
    Trainer(**vars(parser.parse_args())).run()

    # shutil.rmtree('./animations/tmp_imgs')
    # os.makedirs('./animations/tmp_imgs', exist_ok=True)
    # vis_sample(10)



