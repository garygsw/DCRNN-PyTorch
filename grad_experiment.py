import argparse
import numpy as np
import os
import sys
import yaml
import torch

from lib.utils import load_graph_data, DataLoader
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def get_gradients(data_loader, model):
    gradients = []
    for i, (x, y) in enumerate(data_loader.get_iterator()):
        print('getting gradients for batch %s out of %s ...', (i, data_loader.num_batch))
        x, y = model._prepare_data(x, y)
        x.requires_grad = True
        output = model.dcrnn_model(x)
        model.dcrnn_model.zero_grad()
        torch.sum(output).backward()
        with torch.no_grad():
            gradient = x.grad.detach().cpu().numpy()
            gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients

def analyze_gradients(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['seed'] = args.seed

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        dataset = 'val'
        supervisor.dcrnn_model = supervisor.dcrnn_model.eval()
        data_loader = supervisor._data['{}_loader'.format(dataset)]
        gradients = get_gradients(data_loader, supervisor)
        np.savez_compressed('heatmaps/val_gradients.npz', gradients)

def analyze_smoothgrad(args, noise_scale=15, num_noise=50):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['seed'] = args.seed

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        dataset = 'val'
        supervisor.dcrnn_model = supervisor.dcrnn_model.eval()
        data_loader = supervisor._data['{}_loader'.format(dataset)]

        smoothgrads = []
        for _, (x, y) in enumerate(data_loader.get_iterator()):
            x, y = supervisor._prepare_data(x, y)
            #print('x shape:', x.shape)
            noised_inputs = torch.stack([torch.zeros_like(x) for _ in range(num_noise)])
            input_noise_scale = noise_scale * (np.max(x.numpy()) - np.min(x.numpy()))

            for i in range(num_noise):
                noised_inputs[i] = x + input_noise_scale * torch.randn_like(x)
                noised_data_loader = DataLoader(noised_inputs[i], y, batch_size=64, pad_with_last_sample=False)
                gradient = get_gradients(noised_data_loader, supervisor)
                smoothgrads.append(gradient)

        smoothgrads = np.mean(smoothgrads, axis=0)
        np.savez_compressed('heatmaps/val_smoothgrads.npz', smoothgrads)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--seed', default='1')
    args = parser.parse_args()


    # Generate gradients attributions
    analyze_gradients(args)
    #analyze_smoothgrad(args)
    #analyze_ig(args)
    #analyze_smoothtaylor(args)

