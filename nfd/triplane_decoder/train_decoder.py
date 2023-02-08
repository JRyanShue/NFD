import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


from axisnetworks import *
device = torch.device('cuda')
from dataset_3d import *


dataset = MultiOccupancyDataset('../watertight_subset_pts')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

model = MultiTriplane(500, input_dim=3, output_dim=1).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters())

losses = []

step = 0
for epoch in range(30000):
    loss_total = 0
    for obj_idx, X, Y in dataloader:
        # X, Y = X.float().cuda(), Y.float().cuda()
        X, Y = X.float(), Y.float()

        preds = model(obj_idx, X)
        loss = nn.BCEWithLogitsLoss()(preds, Y)
        # loss = nn.functional.mse_loss(preds, Y)

        # # # DENSITY REG
        rand_coords = torch.rand_like(X) * 2 - 1
        rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
        d_rand_coords = model(obj_idx, rand_coords)
        d_rand_coords_offset = model(obj_idx, rand_coords_offset)
        loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 3e-1

	loss += model.tvreg() * 1e-2
	loss += model.l2reg() * 1e-3
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        step += 1
        # if step%50 == 0: print(loss.item())

        loss_total += loss
    print(f"Epoch: {epoch} \t {loss_total.item():01f}")
    if epoch%20 == 0:
        torch.save(model.net.state_dict(), "decoder_net_ckpt/decoder_loss="+str(loss_total.item())+".pt")


