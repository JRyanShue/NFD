import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from axisnetworks import *
device = torch.device('cuda')
from dataset_3d import *
from matplotlib import pyplot as plt



def train_single(in_file, out_file):
    dataset = OccupancyDataset(in_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = MultiTriplane(1, input_dim=3, output_dim=1).to(device)
    model.net.load_state_dict(torch.load('models/decoder_500_net_only.pt'))

    model.embeddings.train()
    model.net.eval()
    for param in model.net.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters())

    losses = []

    step = 0
    for epoch in range(40):
        start = time.time()
        loss_total = 0
        for X, Y in dataloader:
            X, Y = X.float(), Y.float()

            preds = model(0, X)
            loss = nn.BCEWithLogitsLoss()(preds, Y)

            # # # DENSITY REG
            rand_coords = torch.rand_like(X) * 2 - 1
            rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
            d_rand_coords = model(0, rand_coords)
            d_rand_coords_offset = model(0, rand_coords_offset)
            loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 6e-1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # if step%50 == 0: print(loss.item())

            loss_total += loss
        print(f"Epoch: {epoch} \t {loss_total.item():01f}")
        print(time.time() - start)
        losses.append(loss_total.item())

    triplane0 = model.embeddings[0].cpu().detach().numpy()
    triplane1 = model.embeddings[1].cpu().detach().numpy()
    triplane2 = model.embeddings[2].cpu().detach().numpy()

    res = np.concatenate((triplane0, triplane1, triplane2))
    np.save(out_file, res)
    print("Triplane Dims: "+str(res.shape))
   




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    train_single(args.input, args.output)
