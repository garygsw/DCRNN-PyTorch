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
        print('getting gradients for batch %s out of %s ...' % (i, data_loader.num_batch))
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
    print('analyzing grads...')
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['seed'] = args.seed

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        dataset = 'val'
        supervisor.dcrnn_model = supervisor.dcrnn_model.eval()
        data_loader = supervisor._data['{}_loader'.format(dataset)]
        gradients = []
        for i, (x, y) in enumerate(data_loader.get_iterator()):
            print('getting gradients for batch %s out of %s ...', (i, data_loader.num_batch))
            x, y = supervisor._prepare_data(x, y)
            x.requires_grad = True
            output = supervisor.dcrnn_model(x)
            supervisor.dcrnn_model.zero_grad()
            torch.sum(output).backward()
            with torch.no_grad():
                gradient = x.grad.detach().cpu().numpy()
                gradients.append(gradient)
        gradients = np.array(gradients)
        np.savez_compressed('heatmaps/val_gradients.npz', gradients)

def analyze_smoothgrad(args, noise_scale=0.15, num_noise=50):
    print('analyzing smooth grads...')
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
            print('getting smoothgrad for batch %s out of %s ...' % (_, data_loader.num_batch))
            #print('x shape:', x.shape)
            input_noise_scale = noise_scale * (np.max(x.cpu().numpy()) - np.min(x.cpu().numpy()))

            gradients = []
            for i in range(num_noise):
                noised_input = x + input_noise_scale * torch.randn_like(x)
                noised_input.requires_grad = True
                output = supervisor.dcrnn_model(noised_input)
                supervisor.dcrnn_model.zero_grad()
                torch.sum(output).backward()
                with torch.no_grad():
                    gradient = noised_input.grad.detach().cpu().numpy()
                    gradients.append(gradient)

            smoothgrad = np.mean(gradients, axis=0)
            smoothgrads.append(smoothgrad)
        np.savez_compressed('heatmaps/val_smoothgrads.npz', smoothgrads)


def analyze_ig(args, baseline="zero", steps=50):
    print('analyzing IG...')
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['seed'] = args.seed

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        dataset = 'val'
        supervisor.dcrnn_model = supervisor.dcrnn_model.eval()
        data_loader = supervisor._data['{}_loader'.format(dataset)]

        igs = []
        for _, (x, y) in enumerate(data_loader.get_iterator()):
            x, y = supervisor._prepare_data(x, y)
            print('getting IG for batch %s out of %s ...' % (_, data_loader.num_batch))
            #print('x shape:', x.shape)
            baseline = torch.zeros_like(x)
            baseline = supervisor.standard_scaler.transform(baseline)
            
            diff = x - baseline
            scaled_inputs = [baseline + (float(i) / steps) * diff for i in range(steps + 1)]
            
            gradients = []
            for i in range(len(scaled_inputs)):
                scaled_input = scaled_inputs[i]
                scaled_input.requires_grad = True
                output = supervisor.dcrnn_model(scaled_input)
                supervisor.dcrnn_model.zero_grad()
                torch.sum(output).backward()
                with torch.no_grad():
                    gradient = scaled_input.grad.detach().cpu().numpy()
                    gradients.append(gradient)

            ig= np.mean(gradients, axis=0)
            igs.append(ig)
        np.savez_compressed('heatmaps/val_igs.npz', igs)


def analyze_smoothtaylor(args, num_roots=50, noise_scale=0.15):
    print('analyzing smoothtaylor...')
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['seed'] = args.seed

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        dataset = 'val'
        supervisor.dcrnn_model = supervisor.dcrnn_model.eval()
        data_loader = supervisor._data['{}_loader'.format(dataset)]

        smoothtaylors = []
        for _, (x, y) in enumerate(data_loader.get_iterator()):
            x, y = supervisor._prepare_data(x, y)
            print('getting smoothtaylor for batch %s out of %s ...' % (_, data_loader.num_batch))

            input_noise_scale = noise_scale * (np.max(x.cpu().numpy()) - np.min(x.cpu().numpy()))
            
            gradients = []
            noised_inputs = []
            for i in range(num_roots):
                noised_input = x + input_noise_scale * torch.randn_like(x)
                noised_input.requires_grad = True
                noised_inputs.append(noised_input)
                output = supervisor.dcrnn_model(noised_input)
                supervisor.dcrnn_model.zero_grad()
                torch.sum(output).backward()
                with torch.no_grad():
                    gradient = noised_input.grad.detach().cpu().numpy()
                    gradients.append(gradient)

            smoothtaylor = np.mean([(x - noised_inputs[i]).numpy() * gradients[i]
                             for i in range(num_roots)], axis=0)
            smoothtaylors.append(smoothtaylor)
        np.savez_compressed('heatmaps/val_smoothtaylors.npz', smoothtaylors)




if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--seed', default='1')
    args = parser.parse_args()


    # Generate gradients attributions
    #analyze_gradients(args)
    analyze_smoothgrad(args)
    #analyze_ig(args)
    analyze_smoothtaylor(args)

