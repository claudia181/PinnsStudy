import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from evaluate import evaluate

# Plot data ------------------------------------------------------------------------------
def plot_points(dataset_file, labels_idx = 1, points_idx = 0, save = False, plot_file = 'plot.png'):
    dataset = torch.load(dataset_file, weights_only=False)
    dataloader = DataLoader(dataset, 1024)
    points = []
    labels = []
    for batch in dataloader:
        points.append(batch[points_idx])
        labels.append(batch[labels_idx])
    points = torch.cat(points)
    labels = torch.cat(labels)
    font = {'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=1, c=labels.flatten(), cmap='inferno')
    plt.title('Points (color = u)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig(fname=plot_file)

# Plot model stats -----------------------------------------------------------------------
def plot_model_stats(filepaths: list, labels: list, to_plot: list):
    plt.figure(figsize=(8, 4))
    for filepath, label in zip(filepaths, labels):
        # Load stacked curves
        data = np.load(filepath)

        steps = data[:, 0]
        tot_losses = data[:, 1]
        out_losses = data[:, 2]
        der_losses = data[:, 3]
        hes_losses = data[:, 4]
        pde_losses = data[:, 5]
        bc_losses = data[:, 6]
        ic_losses = data[:, 7]
        losses = data[:, 8]

        if "tot" in to_plot:
            if len(filepaths) <= 2: print(f"Last tot: {tot_losses[-1]}")
            plt.plot(steps, tot_losses, label=f'Total Loss {label}')
        if "out" in to_plot:
            if len(filepaths) <= 2: print(f"Last out: {out_losses[-1]}")
            plt.plot(steps, out_losses, label=f'OUT Loss {label}')
        if "der" in to_plot:
            if len(filepaths) <= 2: print(f"Last der: {der_losses[-1]}")
            plt.plot(steps, der_losses, label=f'DER Loss {label}')
        if "hes" in to_plot:
            if len(filepaths) <= 2: print(f"Last hes: {hes_losses[-1]}")
            plt.plot(steps, hes_losses, label=f'HES Loss {label}')
        if "pde" in to_plot:
            if len(filepaths) <= 2: print(f"Last pde: {pde_losses[-1]}")
            plt.plot(steps, pde_losses, label=f'PDE Loss {label}')
        if "bc" in to_plot:
            if len(filepaths) <= 2: print(f"Last bc: {bc_losses[-1]}")
            plt.plot(steps, bc_losses, label=f'BC Loss {label}')
        if "ic" in to_plot:
            if len(filepaths) <= 2: print(f"Last ic: {ic_losses[-1]}")
            plt.plot(steps, ic_losses, label=f'IC Loss {label}')
        if "train" in to_plot and "train_stats" in filepath:
            if len(filepaths) <= 2: print(f"Last train loss: {losses[-1]}")
            plt.plot(steps, losses, label=f'train Loss {label}')
        if "test" in to_plot and "test_stats" in filepath:
            if len(filepaths) <= 2: print(f"Last test loss: {losses[-1]}")
            plt.plot(steps, losses, label=f'test Loss {label}')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    # plt.ylim(0, 100) # Example: y-axis from 0 to 100
    plt.tight_layout()
    plt.show()

def print_model_info(filepath):
    # Load the entire checkpoint dictionary
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")
        return
    checkpoint = torch.load(filepath, weights_only=False)

    print("\n--- Model Parameters ---")
    print(f"PDE Name: {checkpoint.get('pde', 'N/A')}")
    print(f"PDE Parameters: {checkpoint.get('pde_params', 'N/A')}")
    print(f"Hidden Units: {checkpoint.get('hidden_units', 'N/A')}")
    print(f"Initial Learning Rate (lr_init): {checkpoint.get('lr_init', 'N/A')}")
    print(f"Batch size (batch_size): {checkpoint.get('batch_size', 'N/A')}")
    print(f"BC Loss Weight: {checkpoint.get('bc_weight', 'N/A')}")
    print(f"IC Loss Weight: {checkpoint.get('ic_weight', 'N/A')}")
    print(f"Physics Loss Weight: {checkpoint.get('phy_weight', 'N/A')}")
    print(f"Distillation Loss Weight: {checkpoint.get('distill_weight', 'N/A')}")
    print(f"EWC Loss Weight: {checkpoint.get('ewc_weight', 'N/A')}")
    print(f"Activation Function: {checkpoint.get('activation', 'N/A')}")

def plot_loss(model_sequence: str, losses: list, data_sequence: list, title: str = "Forgetting"):
    steps = [i for i in range(len(model_sequence))]
    
    plt.figure(figsize=(8, 4))

    for i, data_path in enumerate(data_sequence):
        if  "out" in losses:
            out_losses = []
        if  "der" in losses:
            der_losses = []
        if  "hes" in losses:
            hes_losses = []
        if "pde" in losses:
            pde_losses = []

        for model_path in model_sequence:
            out_loss, der_loss, hes_loss, pde_loss = evaluate(model_path=model_path, data_path=data_path)
            if  "out" in losses:
                out_losses.append(out_loss)
            if  "der" in losses:
                der_losses.append(der_loss)
            if  "hes" in losses:
                hes_losses.append(hes_loss)
            if "pde" in losses:
                pde_losses.append(pde_loss)

        if  "out" in losses:
            plt.plot(steps, out_losses, label=f"OUT Loss Data {i}", marker=".")
        if  "der" in losses:
            plt.plot(steps, der_losses, label=f"DER Loss Data {i}", marker=".")
        if  "hes" in losses:
            plt.plot(steps, hes_losses, label=f"HES Loss Data {i}", marker=".")
        if "pde" in losses:
            plt.plot(steps, pde_losses, label=f"PDE Loss Data {i}", marker=".")

    plt.xticks(steps, [f"Model {i}" for i in steps], rotation=45)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    # plt.ylim(0, 100) # Example: y-axis from 0 to 100
    plt.tight_layout()
    plt.show()
    