import argparse
import numpy as np
import os
import sys
import yaml
import torch

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['seed'] = args.seed

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        dataset = 'val'
        supervisor.dcrnn_model = supervisor.dcrnn_model.eval()

        val_iterator = supervisor._data['{}_loader'.format(dataset)].get_iterator()
        gradients = []
        #explain = torch.tensor()
        for _, (x, y) in enumerate(val_iterator):
            x, y = supervisor._prepare_data(x, y)

            #print('x shape:', x.shape)  # [T, 64, 414]
            x = torch.tensor(x, requires_grad=True)
        
            output = supervisor.dcrnn_model(x)

            #print('output shape:', output.shape)   # [T, 64, 207]
            supervisor.dcrnn_model.zero_grad()
            torch.sum(output[0,:,:]).backward()

            with torch.no_grad():
                gradient = x.grad.detach().cpu().numpy()
                gradients.append(gradient)
                #torch.sum(output[:,])
                #loss = self._compute_loss(y, output)
                #losses.append(loss.item())

                #y_truths.append(y.cpu())
                #y_preds.append(output.cpu())
        #mean_score, outputs = supervisor.evaluate('test')
        np.savez_compressed(args.args.output_filename, gradients)
        #np.savez_compressed(args.output_filename, **outputs)
        #print("MAE : {}".format(mean_score))
        #print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--seed', default='1')
    parser.add_argument('--output_filename', default='data/gradients.npz')
    args = parser.parse_args()
    run_dcrnn(args)
