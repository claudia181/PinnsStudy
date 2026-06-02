import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import numpy as np
from numpy.typing import NDArray
import os
from load_store_utils import load_stats, resume_model
from generate import X, U, DU, D2U
from typing import List, Tuple
from model import PdeNet
from data_utils import extract_TensorDataset, include_time_in_input
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation

def evaluate(
        model_path: str,
        dataset: str|TensorDataset,
        model_name: str = "Model",
        print_info: bool = False
    ) -> tuple:
    """
    Evaluate a model on a dataset and return loss terms values.

    Parameters
    ----------
    model_path : str
        Model file path.
    dataset : str|TensorDataset
        Dataset on which the model is evaluated.
    model_name : str
        Name of the model (for printing).
    print_info : bool
        If True, the computed loss terms are printed.
    
    Returns
    -------
    tuple
        out_loss, der_loss, hes_loss, res_loss
    """
    model = resume_model(model_path=model_path)
    if type(dataset) is str:
            if not os.path.exists(dataset):
                raise ValueError(f"'File {dataset}' not found.")
            dataset = torch.load(os.path.join(dataset), weights_only=False)
    if model.time_in_input:
        dataset = include_time_in_input(dataset)
    out_loss, der_loss, hes_loss, res_loss = model.evaluate(dataset=dataset)
    if print_info:
        print(f"-------- {model_name} --------")
        print(f"OUT loss: {out_loss}")
        print(f"DER loss: {der_loss}")
        print(f"HES loss: {hes_loss}")
        print(f"RES loss: {res_loss}")
    return out_loss, der_loss, hes_loss, res_loss

# Plot data ------------------------------------------------------------------------------
def plot_points(
        dataset: str|TensorDataset|list,
        labels_idx: int|list = U,
        points_idx: int|list = X,
        labels_name: str|list = "u",
        save: bool = False,
        dst_file: str = "points.png",
        show: bool = True,
        cmap: str|list = "inferno",
        title: str = "",
        subtitles: list = [],
        single_row: bool = False,
        figsize: tuple = (5, 5),
        vmin: float|list = None,
        vmax: float|list = None
    ) -> None:
    """
    Plot the dataset(s) points, colored according to the label values.

    Parameters
    ----------
    dataset : str|TensorDataset|list
        The dataset filepath/TensorDataset object or TensorDataset list.
    labels_idx : int|list,
        Labels column index in the dataset(s).
    points_idx : int|list
        Spatial coordinates column index in the dataset(s).
    labels_name : str|list,
    save : bool
        If True, the plot is saved.
    dst_file : str
        Filepath where to save the plot.
    show : bool
        If True, the plot is showed.
    cmap : str|list
    title : str
    subtitles : list
    single_row : bool
    figsize : tuple
    vmin : float
    vmax : float
    
    Returns
    -------
    None
    """
    if type(dataset) is str:
        datasets = [torch.load(dataset, weights_only=False)]
    elif type(dataset) is list:
        datasets = []
        for ds in dataset:
            if type(ds) is str:
                datasets.append(torch.load(ds, weights_only=False))
            else:
                datasets.append(ds)
    else:
        datasets = [dataset]

    if type(labels_idx) is int:
        labels_idx = [labels_idx for _ in datasets]
    if type(labels_name) is str:
        names = [labels_name for _ in datasets]
    else:
        names = labels_name
    if type(points_idx) is int:
        points_idx = [points_idx for _ in datasets]
    pointss = [dataset.tensors[i] for dataset, i in zip(datasets, points_idx)]
    labelss = []
    if type(cmap) is str:
        cmaps = []
        for dataset, i in zip(datasets, labels_idx):
            if i > -1:
                labelss.append(dataset.tensors[i])
                cmaps.append(cmap)
            else:
                labelss.append(torch.zeros_like(dataset.tensors[U]))
                cmaps.append("grey")
    else:
        cmaps = cmap
        for dataset, i in zip(datasets, labels_idx):
            labelss.append(dataset.tensors[i])

    if vmin is None or type(vmin) is float:
        vmins = [vmin for _ in datasets]
        vmaxs = [vmax for _ in datasets]
    else:
        vmins = vmin
        vmaxs = vmax

    font = {'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=figsize)

    if len(labels_idx) <= 4 or single_row:
        gs = GridSpec(1, len(labels_idx), figure=fig)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(labels_idx))]
    elif len(labels_idx) <= 8:
        ncols = int(np.ceil(len(labels_idx)/2))
        gs = GridSpec(2, ncols, figure=fig)
        axes = []
        for i in range(ncols):
            axes.append(fig.add_subplot(gs[0, i]))
            axes.append(fig.add_subplot(gs[1, i]))
    elif len(labels_idx) <= 16:
        ncols = int(np.ceil(len(labels_idx)/3))
        gs = GridSpec(3, ncols, figure=fig)
        axes = []
        for i in range(ncols):
            axes.append(fig.add_subplot(gs[0, i]))
            axes.append(fig.add_subplot(gs[1, i]))
            axes.append(fig.add_subplot(gs[2, i]))
    elif len(labels_idx) <= 24:
        ncols = int(np.ceil(len(labels_idx)/4))-1
        gs = GridSpec(5, ncols, figure=fig)
        axes = []
        for i in range(ncols):
            axes.append(fig.add_subplot(gs[0, i]))
            axes.append(fig.add_subplot(gs[1, i]))
            axes.append(fig.add_subplot(gs[2, i]))
            axes.append(fig.add_subplot(gs[3, i]))
            axes.append(fig.add_subplot(gs[4, i]))
    else:
        raise ValueError("Too many plots.")
    
    if subtitles == []:
        subtitles = ["" for _ in axes]
    for ax, points, labels, name, cmap, vmin, vmax, subtitle in zip(axes, pointss, labelss, names, cmaps, vmins, vmaxs, subtitles):
        if vmin is not None and vmax is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], s=1, alpha=1, c=labels.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            scatter = ax.scatter(points[:, 0], points[:, 1], s=1, alpha=1, c=labels.flatten(), cmap=cmap)

        #if ax == axes[-1]:
        #    cbar = fig.colorbar(scatter, ax=ax)
        #    cbar.set_label(name)
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        if subtitle != "":
            ax.set_title(subtitle)
        
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.plot()
    fig.subplots_adjust(right=0.85, wspace=0.5, hspace=0.5) 
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7]) # [left, bottom, width, height]
    fig.colorbar(scatter, cax=cbar_ax, label=names[-1])
    #cbar = fig.colorbar(scatter, ax=axes, fraction=0.046, pad=0.04)
    #cbar.set_label(names[-1])
    
    #plt.gca().set_aspect('equal', adjustable='box')
    fig.suptitle(title)
    #if caption != "":
    #    fig.text(
    #        0.01, 0.5, caption, 
    #        ha='left', va='center', fontsize=10, 
    #        linespacing=1.5
    #    )
    #    # rect=[left, bottom, right, top]
    #    plt.tight_layout(rect=[0.2, 0, 1, 1])
    #else:
    #    fig.tight_layout()
    #plt.tight_layout()
    if save:
        plt.savefig(fname=dst_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_points_grid(
        dataset: str|TensorDataset,
        labels_idx: list = [1, 2, 3, 4, 5, 6, 7, 8],
        label_names: list = ["u", "ux", "uy",  "uxx", "uxy", "uyx", "uyy", "res"],
        points_idx: int = 0,
        subtitles: list = ["", "", "", "", "", "", "", ""],
        save: bool = False,
        dst_file: str = "points_grid.png",
        show: bool = True,
        cmap: str = "bwr",
        title: str = "",
        figsize: tuple = (12, 4.3),
        vmins: list = [None, None, None, None, None, None, None, None],
        vmaxs: list = [None, None, None, None, None, None, None, None]
    ) -> tuple:
    """
    Plot the dataset points values in a grid, colored according to the labels values.

    Parameters
    ----------
    dataset : str|TensorDataset
        The dataset filepath/TensorDataset object.
    labels_idx : list
        Labels column indices in the dataset.
    labels_names : list
    points_idx : int
        Spatial coordinates column index in the dataset.
    subtitles : list
    save : bool
        If True, the plot is saved.
    show : bool
        If True, the plot is showed.
    dst_file : str
        Filepath where to save the plot.
    cmap : str
    title : str
    figsize : tuple
    vmins : List[float]
    vmaxs : List[float]
    
    Returns
    -------
    tuple
        vmins, vmaxs
    """
    if type(dataset) is str:
        dataset = torch.load(dataset, weights_only=False)
    points = dataset.tensors[points_idx]
    labelss = [dataset.tensors[i] for i in labels_idx]

    font = {'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=figsize)
    if len(labels_idx) <= 4:
        gs = GridSpec(1, len(labels_idx), figure=fig)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(labels_idx))]
    elif len(labels_idx) <= 8:
        ncols = int(np.ceil(len(labels_idx)/2))
        gs = GridSpec(2, ncols, figure=fig)
        axes = []
        for i in range(ncols):
            axes.append(fig.add_subplot(gs[0, i]))
            axes.append(fig.add_subplot(gs[1, i]))
    else:
        raise ValueError("Too many plots.")

    for ax, labels, label_name, subtitle, vmin, vmax in zip(axes, labelss, label_names, subtitles, vmins, vmaxs):
        ax.set_title(subtitle)
        if vmin is not None and vmax is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], s=1, alpha=1, c=labels.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            scatter = ax.scatter(points[:, 0], points[:, 1], s=1, alpha=1, c=labels.flatten(), cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(label_name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.plot()

    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(dst_file)
    if show:
        plt.show()
    else:
        plt.close()
    vmins = [torch.min(labels) for labels in labelss]
    vmaxs = [torch.max(labels) for labels in labelss]
    return vmins, vmaxs

def get_animation(
        dataset: ConcatDataset,
        figsize: tuple = (3, 2.5),
        cmap: str = "inferno",
        interval: int = 400,
        filename: str = "animation.gif",
        save: bool = False
    ) -> animation.ArtistAnimation:
    """
    Return and save the .gif animation relative to the ConcatDataset dataset.

    Parameters
    ----------
    dataset : ConcatDataset
        Each dataset in dataset.datasets correspond to a time instant.
    figsize : tuple
    cmap : str
    interval : int
        Delay between frames in milliseconds.
    filename : str
    save: bool
        If True, the .gif is saved.
    
    Returns
    -------
    animation.ArtistAnimation
    """
    datas = [ds.tensors[U] for ds in dataset.datasets]
    vmax, vmin = max(d.max() for d in datas),  min(d.min() for d in datas)
    points = dataset.datasets[0].tensors[X]
    x, y = points[:, 0], points[:, 1]
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for data in datas:
        scatter = ax.scatter(x=x, y=y, c=data.flatten(), s=1, alpha=1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        artists.append([scatter])
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("u")
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=interval)
    if save:
        ani.save(filename, writer="pillow")
    return ani

def get_pendulum_animation(
        t: NDArray, u: NDArray, l: float, 
        figsize: tuple = (6, 6),
        xmin: float = -1.2, xmax: float = 1.2,
        ymin: float = -1.2, ymax: float = 1.2,
        grid: bool = True,
        marker_path: str = ".", 
        markersize_path: float = 10,
        #linewidth_path: float = 10, 
        #trasparency_path: float = 0.5, 
        marker_color_path: str = "orange", 
        marker_cmap_path: str = "Oranges",
        #linestyle_path: str = "", 
        marker_rod: str = "o", 
        markersize_rod: float = 10, 
        linewidth_rod: float = 2, 
        color_rod: str = "purple", 
        linestyle_rod: str = "-",
        animation_delay: float = 10,
        filename: str = "Pendulum/simulation.gif"
    ) -> animation.ArtistAnimation:
    """
    Return and save the .gif animation of the simulation (t, u, l).

    Parameters
    ----------
    t : NDArray
        Time instants of the simulation.
    u : NDArray
        Angle sequence of the simulation.
    l : NDArray
        Rod length.

        Some plot parameters ...

    filename : str
        File to save (must contain the extension .gif)
    save: bool
        If True, the .gif is saved.
    
    Returns
    -------
    animation.ArtistAnimation
    """

    #t = np.linspace(0, 10, 200)
    #angles = 0.8 * np.exp(-0.1 * t) * np.cos(5 * t) # Damped oscillation
    #L = 1.0
    # Convert polar coords into cartesian coords
    x = l * np.sin(u)
    y = -l * np.cos(u)

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.6)

    # Path
    path, = ax.plot(
        [], [], 
        #color=color_path, 
        #linestyle=linestyle_path, 
        #lw=linewidth_path, 
        #alpha=trasparency_path, 
        marker=marker_path, 
        markersize=markersize_path
    )

    # Rod + bob
    rod, = ax.plot(
        [], [], 
        color=color_rod, 
        linestyle=linestyle_rod, 
        lw=linewidth_rod, 
        marker=marker_rod, 
        markersize=markersize_rod
    )

    path = ax.scatter([], [], c=[], cmap=marker_cmap_path, s=markersize_path)

    def update(i):
        # Update scatter points: 
        # x and y are positions, c is the color sequence (0 to i)
        if i <= 10:
            path.set_offsets(np.c_[x[:i], y[:i]])
            path.set_array(np.arange(i))
        else:
            path.set_offsets(np.c_[x[i-10:i], y[i-10:i]])
            path.set_array(np.arange(start=i-10, stop=i))

        rod.set_data([0, x[i]], [0, y[i]])
        return rod, path

    # Animation logic
    #def update(i):
    #    # Update the path using all points up to the current frame
    #    path.set_data(x[:i], y[:i])
#
    #    # Update the main pendulum
    #    rod.set_data([0, x[i]], [0, y[i]])
#
    #    return rod, path

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(t), interval=animation_delay, blit=True)

    # Save as GIF
    ani.save(filename, writer='pillow')

    return ani

def plot_timeseries_and_phase_portrait(
        dataset,
        u_idx = 1,
        du_idx = 2,
        t_idx = -1,
        cmap: str = "inferno",
        s: int = 1,
        marker: str = ".",
        r: tuple|int = None,
        figsize: tuple = (4, 2)
    ):
    """
    Plot time series and phase protrait.

    Parameters
    ----------
    dataset : str | TensorDataset
    u_idx : int
        u column index in the dataset.
    du_idx : int
        du column index in the dataset.
    t_idx : int
        t column index in the dataset.
    cmap : str
    s : int
        Scatter param.
    figsize : tuple

    Returns
    -------
    None
    """
    if type(dataset) is str:
        dataset = torch.load(dataset, weights_only=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1024)
    t = []
    u = []
    du = []
    for batch in dataloader:
        t.append(batch[t_idx])
        u.append(batch[u_idx])
        du.append(batch[du_idx])
    t = torch.cat(t).numpy()
    u = torch.cat(u).numpy()
    du = torch.cat(du).numpy()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    
    if r is None:    
        ax0.plot(t, u)
        scatter = ax1.scatter(u, du, c=t, cmap=cmap, s=s, marker=marker)
    elif type(r) is int:
        ax0.plot(t[r:], u[r:])
        scatter = ax1.scatter(u[r:], du[r:], c=t[r:], cmap=cmap, s=s, marker=marker)
    else:
        ax0.plot(t[r[0]:r[1]], u[r[0]:r[1]])
        scatter = ax1.scatter(u[r[0]:r[1]], du[r[0]:r[1]], c=t[r[0]:r[1]], cmap=cmap, s=s, marker=marker)

    ax0.set_xlabel("t")
    ax0.set_ylabel("u")
    ax0.set_title("Time series")

    #cbar = ax1.set_colorbar(scatter)
    fig.colorbar(scatter, ax=ax1, label='Time')
    #cbar.set_label('Time')
    ax1.plot()
    ax1.set_xlabel("u")
    ax1.set_ylabel("du")
    ax1.set_title("Phase portrait")

    plt.tight_layout()
    plt.show()

def plot_timeseries(
        dataset: str|TensorDataset,
        u_idx: int = 1,
        t_idx: int = -1,
        unwrap: bool = False,
        r: tuple|int = None,
        figsize: tuple = (4, 2)
    ) -> None:
    """
    Plot the timeseries given by the dataset columns (t_idx, u_idx).

    Parameters
    ----------
    dataset : str|TensorDataset
        The dataset from which to take the timeseries.
    u_idx : int
        Index of the values column.
    t_idx : int
        Index of the time column.
    unwrap : bool
    r : tuple|int
        Time instant from which to start to plot.
    figsize : tuple
    
    Returns
    -------
    None
    """
    if type(dataset) is str:
        dataset = torch.load(dataset, weights_only=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1024)
    t = []
    u = []
    for batch in dataloader:
        t.append(batch[t_idx])
        u.append(batch[u_idx])
    t = torch.cat(t).numpy()
    u = torch.cat(u).numpy()

    if unwrap:
        u = np.unwrap(u)
    plt.figure(figsize=figsize)
    if r is None:
        plt.plot(t, u)
    elif type(r) is int:
        plt.plot(t[r:], u[r:])
    else:
        plt.plot(t[r[0]:r[1]], u[r[0]:r[1]])
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title(f"Time series")
    plt.show()

def plot_timeseries2(
        datasets: List[str] | List[TensorDataset],
        u_idxs: List[int],
        t_idxs: List[int],
        labels: List[str],
        r: Tuple[int] | int = None,
        figsize: tuple = (4, 2)
    ) -> None:
    """
    Plot multiple timeseries given by the datasets columns (t_idxs[i], u_idxs[i]).

    Parameters
    ----------
    dataset : List[str] | List[TensorDataset]
        The dataset from which to take the timeseries.
    u_idxs : List[int]
        List of indices of the values columns.
    t_idxs : List[int]
        List of indices of the time columns.
    labels : List[str]
        List of labels for the plot.
    r : Tuple[int] | int
        - Time instant from which to start to plot or
        - time interval to plot.
    figsize : tuple
    
    Returns
    -------
    None
    """
    ts = []
    us = []
    plt.figure(figsize=figsize)
    for dataset, t_idx, u_idx, label in zip(datasets, t_idxs, u_idxs, labels):
        if type(dataset) is str:
            dataset = torch.load(dataset, weights_only=False)
        dataloader = DataLoader(dataset=dataset, batch_size=1024)
        t = []
        u = []
        for batch in dataloader:
            t.append(batch[t_idx])
            u.append(batch[u_idx])
        t = torch.cat(t).numpy()
        u = torch.cat(u).numpy()
        if type(r) is int:
            t = t[r:]
            u = u[r:]
        elif type(r) is list or type(r) is tuple:
            t = t[r[0]:r[1]]
            u = u[r[0]:r[1]]
        ts.append(t)
        us.append(u)

        plt.plot(t, u, label=label)

    plt.xlabel("t")
    plt.ylabel("u")
    plt.legend()
    plt.title(f"Time series")
    plt.show()
    
def plot_phase_portrait(
        dataset: str | TensorDataset,
        u_idx: int = 1,
        du_idx: int = 2,
        t_idx: int = -1,
        wrap: bool = False,
        cmap: str = "inferno",
        s: int = 1,
        marker: str = ".",
        r: Tuple[int] | int = None,
        figsize: tuple = (4, 2)
    ) -> None:
    """
    Plot the scatter phase protrait from u, du, t in dataset.

    Color indicates time. Position of each point indicates the system state at the time corresponding to the color of the point.

    Parameters
    ----------
    dataset : str | TensorDataset
    u_idx : int
        Index of the u (function values) column in the dataset.
    du_idx : int
        Index of the du (function derivative values) column in the dataset.
    t_idx : int
        Index of the t (time values) column in the dataset.
    wrap : bool
    cmap : str
    s : int
    marker : str
    r : Tuple[int] | int
        Start time or time interval to plot.
    figsize : tuple
    
    Returns
    -------
    None
    """
    if type(dataset) is str:
        dataset = torch.load(dataset, weights_only=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1024)
    t = []
    u = []
    du = []
    for batch in dataloader:
        t.append(batch[t_idx])
        u.append(batch[u_idx])
        du.append(batch[du_idx])
    t = torch.cat(t).numpy()
    u = torch.cat(u).numpy()
    du = torch.cat(du).numpy()

    if wrap:
        u = (u + np.pi) % (2*np.pi) - np.pi
    
    plt.figure(figsize=figsize)
    if r is None:
        scatter = plt.scatter(u, du, c=t, cmap=cmap, s=s, marker=marker)
    elif type(r) is int:
        scatter = plt.scatter(u[r:], du[r:], c=t[r:], cmap=cmap, s=s, marker=marker)
    else:
        scatter = plt.scatter(u[r[0]:r[1]], du[r[0]:r[1]], c=t[r[0]:r[1]], cmap=cmap, s=s, marker=marker)
    # Add a color bar to show the relationship between color and time
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time')
    plt.plot()#plt.plot(u, du, marker, ms=1)
    plt.xlabel("u")
    plt.ylabel("du")
    plt.title(f"Phase portrait")
    plt.show()

def plot_phase_portrait2(
        dataset: ConcatDataset,
        t: int = 0,
        velocity_idx: int = -2,
        space_idx: int = 0,
        start: int = 0,
        stop: int = None,
        n_points: int = 500,
        normalize: bool = True,
        figsize: tuple = (6, 6),
        title: str = "",
        color: str = "blue",
        alpha: float = 0.8
    ):
    """
    Plot the arrows phase protrait (vector field) from velocity and space informations in dataset.

    Parameters
    ----------
    dataset : ConcatDataset
    t : int
        Index of the TesnorDataset in dataset.datsets to plot.
    velocity_idx: int
        Index of the velocity values column in the dataset.
    space_idx : int
        Index of the space values column in the dataset.
    start : int
        Linspace start.
    stop : int
        Linspace stop.
    n_points : int
        Linspace n_points.
    normalize : bool
        If True, vectors are normalizer (for a better visualization).
    figsize : tuple
    title : str
    color : str
    alpha : float
        plt.quiver parameter.
    
    Returns
    -------
    None
    """
    if stop is None: stop = len(dataset.datasets[0].tensors[0]) - 1
    vx = dataset.datasets[t][:][velocity_idx][np.linspace(start=start, stop=stop, num=n_points), 0]
    vy = dataset.datasets[t][:][velocity_idx][np.linspace(start=start, stop=stop, num=n_points), 1]
    x = dataset.datasets[t][:][space_idx][np.linspace(start=start, stop=stop, num=n_points), 0]
    y = dataset.datasets[t][:][space_idx][np.linspace(start=start, stop=stop, num=n_points), 1]

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    if normalize:
        # Normalize the vectors for better visualization
        norm = np.sqrt(vx**2 + vy**2)
    else:
        norm = np.ones_like(vx)

    vx_norm, vy_norm = vx / norm, vy / norm

    # plt.quiver to plot the vector field
    plt.quiver(x, y, vx_norm, vy_norm, color=color, alpha=alpha)

    plt.grid(True)
    plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def plot_dynamics(
    dataset,
    u_idx=1,
    du_idx=2,
    t_idx=-1,
    n_quiver_points=400,
    figsize=(14, 5),
    cmap="inferno",
    color="purple",
    marker=".",
    markersize=2,
    points=True,
    arrows=True,
    norm=True
):
    """
    Combines Vector Field, Trajectory, and Time Series into one figure.
    """
    # 1. Data Extraction (Unified)
    # If it's a ConcatDataset, we'll pull the first dataset for the trajectory
    data_source = dataset.datasets[0] if hasattr(dataset, 'datasets') else dataset
    
    loader = DataLoader(data_source, batch_size=len(data_source))
    batch = next(iter(loader))
    
    t = batch[t_idx].numpy()
    u = batch[u_idx].numpy()
    du = batch[du_idx].numpy()

    # Create Figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    
    # --- Subplot 1: Phase Portrait (Vector Field + Trajectory) ---
    ax1 = fig.add_subplot(gs[:, :2])
    
    if arrows:
        # Plot Vector Field (Quiver)
        # We sample a subset of points to avoid clutter
        indices = np.linspace(0, len(u) - 1, n_quiver_points, dtype=int)
        vx, vy = du[indices], u[indices] # Adjust indices based on your physics
    
        if norm:
            # Simple normalization for quiver arrows
            norm = np.hypot(vx, vy)
        else:
            norm = 1
        qn_x, qn_y = vx / norm, vy / norm
        ax1.quiver(vy, vx, qn_x, qn_y, color=color, alpha=0.3, label='Vector Field')

    if points:
        # Plot Trajectory (Scatter colored by time)
        sc = ax1.scatter(u, du, c=t, cmap=cmap, marker=marker, s=markersize, label='Trajectory')
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('Time')
    
    ax1.set_xlabel("Position (u)")
    ax1.set_ylabel("Velocity (du)")
    ax1.set_title("Trajectory")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Subplot 2: Time Series (u vs t) ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t, u, color='firebrick', lw=1.5)
    ax2.set_ylabel("u (Position)")
    ax2.set_title("Time Series")
    ax2.grid(True)

    # --- Subplot 3: Time Series (du vs t) ---
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(t, du, color='royalblue', lw=1.5)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("du (Velocity)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

# Plot model stats -----------------------------------------------------------------------
def plot_model_stats(
        data: str | dict,
        keys: list,
        subkeys: list,
        ylabel: str = "",
        title: str = "",
        figsize: tuple = (8, 4),
        save: bool = False,
        dst_file: str = "model_stats.png",
        show: bool = True
    ) -> None:
    """
    Plot losses, weights or grad norms contained in data.

    Parameters
    ----------
    data : str | dict
        Model stats dictionary directory path or dictionary of model stats.
    keys : list
        Keys of the dictionary stats to plot.
    subkeys : list
        Subkeys of the stats to plot.
    ylabel : str
    title : str
    figsize : tuple
    save : bool
        If True, the plot are saved.
    dst_file : str
        The file where to save the plot.
    show : bool
        If True, the plot is showed.
    
    Returns
    -------
    None
    """
    if type(data) is str:
        data = load_stats(directory=data, key_list=keys)
    plt.figure(figsize=figsize)
    for key in keys:
        for subkey in subkeys:
            steps = data[key]["step_list"]
            losses_or_weights_or_grad_norms = data[key][subkey]
            plt.plot(steps, losses_or_weights_or_grad_norms, label=f"{key} {subkey}")
            if len(keys) <= 2:
                print(f"Last value {key} {subkey}: {losses_or_weights_or_grad_norms[-1]}")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    # plt.ylim(0, 100) # Example: y-axis from 0 to 100
    plt.tight_layout()
    if save:
        plt.savefig(dst_file)
    if show:
        plt.show()
    else:
        plt.close()

def plot_model_stats_grid(
        data: str | dict,
        keys: list,
        subkeys: list,
        figsize: tuple = (18, 3.5),
        legend: bool = True,
        title: str = "",
        save: bool = False,
        dst_file: str = "model_stats_grid.png",
        show: bool = True
    ) -> None:
    """
    Plot a grid of losses, weights or grad norms contained in data.

    Parameters
    ----------
    data : str | dict
        Model stats dictionary directory path or dictionary of model stats.
    keys : list
        Keys of the dictionary stats to plot.
    subkeys:  list
        Subkeys of the stats to plot.
    figsize : tuple
    legend : bool
    title : str
    save : bool
        If True, the plots are saved.
    dst_file : str
        The file where to save the plots.
    show : bool
        If True, the plot is showed.

    Returns
    -------
    None
    """
    if type(data) is str:
        data0 = load_stats(directory=data, key_list=keys)
        if "grad_norms" in keys and \
            "weighted_loss" in subkeys and \
            "weighted_loss" not in data0["grad_norms"].keys():
            data = load_stats(directory=data, key_list=keys+["train_loss_grad_norm"])
            step = int(len(data["train_loss_grad_norm"]) / len(data["grad_norms"]["res_loss"]))
            stop = len(data["train_loss_grad_norm"])
            data["grad_norms"]["weighted_loss"] = [data["train_loss_grad_norm"][i] for i in range(0, stop, step)]
        else:
            data = data0
    fig = plt.figure(figsize=figsize)
    if len(subkeys) <= 4:
        gs = GridSpec(1, len(subkeys), figure=fig)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(subkeys))]
    elif len(subkeys) <= 8:
        ncols = int(np.ceil(len(subkeys)/2))
        gs = GridSpec(2, ncols, figure=fig)
        axes = []
        for i in range(ncols):
            axes.append(fig.add_subplot(gs[0, i]))
            axes.append(fig.add_subplot(gs[1, i]))
    else:
        raise ValueError("Too many plots.")
    
    stepss = [data[key]["step_list"] for key in keys]
    for ax, subkey in zip(axes, subkeys):
        for key, steps in zip(keys, stepss):
            losses_or_weights_or_grad_norms = data[key][subkey]
            ax.plot(steps, losses_or_weights_or_grad_norms, label=f"{key} {subkey}")
        ax.set_xlabel("step")
        #ax.set_ylabel(subkey)
        ax.set_title(subkey)
        if legend:
            ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(dst_file)
    if show:
        plt.show()
    else:
        plt.close()

def plot_loss_comp(
        stats_dict: dict,
        phase: str,
        loss_prefix: str = None,
        figsize: tuple = (8, 4),
        save: bool = False,
        dst_file: str = "loss_comp.png",
        show: bool = True
    ) -> None:
    """
    Show losses of multiple models (whose names are the keys of stats_dict) in the same plot.

    Parameters
    ----------
    stats_dict : dict
    phase : str
        Subkey.
    loss_prefix : str
        Subsubkey prefix.
    figsize : tuple
    save : bool
        If True, the plot is saved.
    dst_file : str
        The file where to save the plot.
    show : bool
        If True, the plot is showed.
    
    Returns
    -------
    None
    """
    if loss_prefix is not None:
        loss_prefix = loss_prefix.lower()
    plt.figure(figsize=figsize)
    for model_name in stats_dict.keys():
        if loss_prefix is not None:
            loss_values = stats_dict[model_name][phase][loss_prefix+"_loss"]
            steps = stats_dict[model_name][phase]["step_list"]
        else:
            loss_values = stats_dict[model_name][phase]
            steps = list(range(len(loss_values)))
        plt.plot(steps, loss_values, label=f"{model_name}")

    plt.xlabel("Step")
    if loss_prefix is not None:
        plt.ylabel(f"{loss_prefix.upper()} loss")
        plt.title(f"{loss_prefix.upper()} loss")
    else:
        plt.ylabel(phase)
        plt.title(phase)
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    # plt.ylim(0, 100) # Example: y-axis from 0 to 100
    plt.tight_layout()
    if save:
        plt.savefig(dst_file)
    if show:
        plt.show()
    else:
        plt.close()

def print_model_info(model_dir: str) -> None:
    """
    Print informations of the model in model_dir.

    Parameters
    ----------
    model_dir : str
        Model directory.
    
    Returns
    -------
    None
    """
    # Load the checkpoint dictionary
    if not os.path.exists(model_dir):
        raise ValueError(f"Model file '{model_dir}/model.pth' not found in '{model_dir}'.")
    checkpoint = torch.load(f"{model_dir}/model.pth", weights_only=False)
    print("\n=============== Checkpoint ===============")
    print(f"PDE: {checkpoint.get('pde', 'N/A')}")
    print(f"PDE parameters in input: {checkpoint.get('pde_params_in_input', 'N/A')}")
    print(f"Time in input: {checkpoint.get('time_in_input', 'N/A')}")
    print(f"Space in input: {checkpoint.get('space_in_input', 'N/A')}")
    print(f"Fourier features: {checkpoint.get('fourier_features', 'N/A')}")
    print(f"|Input units|: {checkpoint.get('input_units', 'N/A')}")
    print(f"Initial learning rate: {checkpoint.get('lr_init', 'N/A')}")
    print(f"Batch size: {checkpoint.get('batch_size', 'N/A')}")
    print(f"Scheduler: {checkpoint.get('scheduler', 'N/A')}")
    print("-----------------------------------------\nModes:\n-----------------------------------------")
    print(f"\tsystem: {checkpoint.get('sys_mode', 'N/A')}")
    print(f"\tdistillation: {checkpoint.get('distill_mode', 'N/A')}")
    print(f"\tEWC: {checkpoint.get('ewc_mode', 'N/A')}")
    print(f"\tDWA: {checkpoint.get('dwa_mode', 'N/A')}")
    print("-----------------------------------------\nSystem loss terms weights:\n-----------------------------------------")
    print(f"\tBC: {checkpoint.get('bc_weight', 'N/A')}")
    print(f"\tIC: {checkpoint.get('ic_weight', 'N/A')}")
    print(f"\tRES: {checkpoint.get('res_weight', 'N/A')}")
    print(f"\tOUT: {checkpoint.get('out_weight', 'N/A')}")
    print(f"\tDER: {checkpoint.get('der_weight', 'N/A')}")
    print(f"\tHES: {checkpoint.get('hes_weight', 'N/A')}")
    print("-----------------------------------------\nUnlabeled, Distillation and EWC loss terms weights:\n-----------------------------------------")
    print(f"\tNL: {checkpoint.get('nl_weight', 'N/A')}")
    print(f"\tOUT: {checkpoint.get('distill_out_weight', 'N/A')}")
    print(f"\tDER: {checkpoint.get('distill_der_weight', 'N/A')}")
    print(f"\tHES: {checkpoint.get('distill_hes_weight', 'N/A')}")
    print(f"\tEWC: {checkpoint.get('ewc_weight', 'N/A')}")
    print("-----------------------------------------\nDWA:\n-----------------------------------------")
    print(f"\talpha: {checkpoint.get('alpha', 'N/A')}")
    print(f"\tweighted avg frequency: {checkpoint.get('weighted_avg_frequency', 'N/A')}")
    print("-----------------------------------------\nImportances:\n-----------------------------------------")
    print(f"\tsystem: {checkpoint.get('sys_importance', 'N/A')}")
    print(f"\tNL: {checkpoint.get('nl_importance', 'N/A')}")
    print(f"\tdistillation: {checkpoint.get('distill_importance', 'N/A')}")
    print(f"\tEWC: {checkpoint.get('ewc_importance', 'N/A')}")
    print(f"\tBC: {checkpoint.get('bc_importance', 'N/A')}")
    print(f"\tIC: {checkpoint.get('ic_importance', 'N/A')}")
    print("==========================================")

def plot_loss(
        model_sequence: List[str],
        losses: List[str],
        data_sequence: List[str] | List[TensorDataset],
        data_ids: List[str] = None,
        title: str = "Forgetting",
        figsize: tuple = (8, 4),
        legend_fontsize: int = 8,
        save: bool = False,
        dst_file: str = "loss.png",
        show: bool = True
    ) -> None:
    """
    Plot multiple loss terms of multiple models.

    Parameters
    ----------
    model_sequence : List[str]
        List of models paths.
    losses : List[str]
        List of losses ids in ["out", "der", "hes", "res"].
    data_sequence : List[str] | List[TensorDataset]
        List of datasets.
    data_ids : List[str]
        Data labels for legend.
    title : str
        The plot title.
    figsize : tuple
        The figure size.
    legend_fontsize : int
    save : bool
        If True, the plot is saved.
    dst_file : str
        The file where to save the plot.
    show : bool
        If True, the plot is showed.
    
    Returns
    -------
    None
    """
    steps = [i for i in range(len(model_sequence))]
    if data_ids is None:
       data_ids = [i for i in range(len(data_sequence))]
    
    plt.figure(figsize=figsize)

    for i, data in zip(data_ids, data_sequence):
        if  "out" in losses:
            out_losses = []
        if  "der" in losses:
            der_losses = []
        if  "hes" in losses:
            hes_losses = []
        if  "res" in losses:
            res_losses = []

        for model_path in model_sequence:
            out_loss, der_loss, hes_loss, res_loss = evaluate(model_path=model_path, dataset=data)
            if  "out" in losses:
                out_losses.append(out_loss)
            if  "der" in losses:
                der_losses.append(der_loss)
            if  "hes" in losses:
                hes_losses.append(hes_loss)
            if  "res" in losses:
                res_losses.append(res_loss)

        if  "out" in losses:
            plt.plot(steps, out_losses, label=f"Out Loss Data {i}", marker=".")
        if  "der" in losses:
            plt.plot(steps, der_losses, label=f"Der1 Loss Data {i}", marker=".")
        if  "hes" in losses:
            plt.plot(steps, hes_losses, label=f"Der2 Loss Data {i}", marker=".")
        if  "res" in losses:
            plt.plot(steps, res_losses, label=f"Res Loss Data {i}", marker=".")

    plt.xticks(steps, [f"{i}" for i in steps])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    # plt.ylim(0, 100) # Example: y-axis from 0 to 100
    plt.tight_layout()
    if save:
        plt.savefig(dst_file)
    if show:
        plt.show()
    else:
        plt.close()

def plot_time_losses(
        models: List[str],
        model_labels: List[str],
        loss: str,
        datas: List[str] | List[TensorDataset],
        logscale: bool = False,
        title: str = "",
        figsize: tuple = (8, 4),
        legend_fontsize: int = 8,
        save: bool = False,
        dst_file: str = "loss.png",
        show: bool = True
    ) -> None:
    """
    Plot loss of multiple models.

    Parameters
    ----------
    models : List[str]
        List of models paths.
    model_labels : List[str]
        List of model labels.
    loss : str
        Loss id in ["out", "der", "hes", "res"].
    datas : List[str] | List[TensorDataset]
        List of datasets.
    logscale : bool
        Apply logscale to the y axis.
    title : str
    figsize : tuple
    legend_fontsize : int
    save : bool
        If True, the plot is saved.
    dst_file : str
        The file where to save the plot.
    show : bool
        If True, the plot is showed.
    
    Returns
    -------
    None
    """
    steps = [i for i in range(len(datas))]
    
    plt.figure(figsize=figsize)

    for model_path, model_label in zip(models, model_labels):
        losses = []
        for data in datas:
            out_loss, der_loss, hes_loss, res_loss = evaluate(model_path=model_path, dataset=data)
            if  loss == "out":
                losses.append(out_loss)
            if  loss == "der":
                losses.append(der_loss)
            if  loss == "hes":
                losses.append(hes_loss)
            if  loss == "res":
                losses.append(res_loss)

        if  loss == "out":
            plt.plot(steps, losses, label=f"Out Loss {model_label}", marker=".")
        if  loss == "der":
            plt.plot(steps, losses, label=f"Der1 Loss {model_label}", marker=".")
        if  loss == "hes":
            plt.plot(steps, losses, label=f"Der2 Loss {model_label}", marker=".")
        if  loss == "res":
            plt.plot(steps, losses, label=f"Res Loss {model_label}", marker=".")

    if logscale:
        plt.yscale("log")
    plt.xticks(steps, [f"{i}" for i in steps])
    plt.xlabel("Task")
    plt.ylabel("Loss")
    if title != "":
        plt.title(title)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)
    #plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    # plt.ylim(0, 100) # Example: y-axis from 0 to 100
    plt.tight_layout()
    if save:
        plt.savefig(dst_file)
    if show:
        plt.show()
    else:
        plt.close()

def plot_subsample(
        dataset: TensorDataset,
        indices: list,
        vmin: float = None,
        vmax: float = None
    ) -> None:
    """
    Plot the function at a subsample of the points in dataset (the rows whose index is in indices).

    Parameters
    ----------
    dataset : TensorDataset
        The full dataset.
    indices : list
        The subset of points (rows) to plot.
    vmin : float
    vmax : float
    
    Returns
    -------
    None
    """
    if vmin is not None: vmin = [vmin]
    if vmax is not None: vmax = [vmax]
    plot_points(
        dataset = TensorDataset(
            dataset.tensors[X][indices],
            dataset.tensors[U][indices]
        ),
        points_idx = 0,
        labels_idx = 1,
        figsize = (2.8, 2),
        vmin = vmin,
        vmax = vmax
    )

def plot_performance_comparison(
        models: List[PdeNet] | List[str],
        models_names: List[str],
        datasets: List[ConcatDataset],
        datasets_names: List[str],
        subset: dict,
        figsize: tuple,
        time_indexes: int = [0],
        rows: int = 1,
        cols: int = None,
        colors: List[str] = [
            "tab:orange",
            "tab:blue",
            "tab:purple",
            "tab:red",
            "tab:green",
            "tab:yellow",
            "tab:pink"
        ],
        bar_labels: bool = True,
        yaxis_visible: bool = True,
        save: bool = False,
        dst_files: List[str] = [],
        show: bool = True
    ):
    """
    Create one figure per dataset.

    Inside each figure, create a grid of bar plots,
    one per loss name.

    Parameters
    ----------
    lossesss : List[List[List[float]]]
        Shape = [num_datasets][num_subsets][4 losses].
    datasets_names : List[str]
        Names of datasets (e.g. ["train", "val"]).
    figsize : tuple
    rows : int
        Number of rows per grid.
    cols : int
        Number of columns per grid.
    bar_labels : bool
        If True, the bars are labeled with the value.
    yaxis_visible : bool
        If False the y-axis labels are not showed.
    save : bool
        If True, the plot is saved.
    dst_files : List[str]
        The files where to save the plot.
    show : bool
        If True, the plot is showed.
    
    Returns
    -------
    None
    """
    if type(models[0]) is str:
        models = [resume_model(model_path=model) for model in models]
    lossess = [[] for _ in datasets]
    loss_names = ["out_loss", "der1_loss", "der1x_loss", "der1t_loss", "der2_loss", "der2x_loss", "der2t_loss", "res_loss"]
                
    for model in models:
        for ds, losses in zip(datasets, lossess):
            ds_subset = extract_TensorDataset(
                dataset=ds,
                time_indexes=time_indexes,
                spatial_ranges=subset
            )
            if model.time_in_input:
                ds_subset = include_time_in_input(ds_subset)
            losses.append(model.evaluate(ds_subset, split_space_time=True))


    num_losses = len(loss_names)
    num_models = len(models)

    if rows == 1:
        cols = len(loss_names)

    d_idx = 0
    for ds_name, dst_file in zip(datasets_names, dst_files):

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        fig.suptitle(f"{ds_name}", fontsize=14)

        for l_idx, loss_name in enumerate(loss_names):

            ax = axes[l_idx]

            # Values for bar heights
            values = [lossess[d_idx][m_idx][l_idx] for m_idx in range(num_models)]

            bars = ax.bar(range(num_models), values, color=colors[:num_models])

            if bar_labels:
                #labels = [f"{val:.2e}" for val in values]
                labels = []
                for val in values:
                    base, exponent = f"{val:.2e}".split('e')
                    exponent = int(exponent)
                    labels.append(f"${base} \\times 10^{{{exponent}}}$")
                bar_labels = ax.bar_label(bars, labels=labels, padding=3)
                for i, text_obj in enumerate(bar_labels):
                    text_obj.set_color(colors[i])
                ax.set_ylim(0, max(values) * 1.3)
            ax.set_title(loss_name)
            ax.set_xlabel("model")
            ax.set_ylabel("loss")
            ax.set_xticks(range(num_models))
            ax.set_xticklabels(models_names)

            formatter = ticker.ScalarFormatter(useMathText=True) # useMathText makes it look like 10^n
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1)) # Forces scientific notation for numbers outside this range
            ax.yaxis.set_major_formatter(formatter)
            if not yaxis_visible:
                ax.get_yaxis().set_visible(False)

        # Remove unused axes if num_losses < rows*cols
        for i in range(num_losses, rows * cols):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            plt.savefig(dst_file)
        if show:
            plt.show()
        else:
            plt.close()
        d_idx += 1


def print_model_performances(
        model: PdeNet | str,
        datasets: List[ConcatDataset],
        datasets_names: List[str],
        subsets: List[dict],
        time_indexes: List[int] = [0]
    ) -> List[List[List[float]]]:
    """
    Print model [out, der, hes, res] loss values on the given datasets.

    Parameters
    ----------
    model : PdeNet | str
    datasets : List[ConcatDatasets]
    datasets_names : List[str]
    spatial_rangess : List[dict]
    time_indexes : List[int]

    Returns
    -------
    List[List[List[float]]]
        Shape = [num_datasets][num_subsets][4 losses].
    """
    if type(model) is str:
        model = resume_model(model_path=model)
    lossesss = [[] for _ in datasets]
    strings = ["out_loss", "der_loss", "hes_loss", "res_loss"]
    ds_subsets = []
    for subset in subsets:
        for ds, lossess in zip(datasets, lossesss):
            ds_subset = extract_TensorDataset(dataset=ds, time_indexes=time_indexes , spatial_ranges=subset)
            ds_subsets.append(ds_subset)
            lossess.append(model.evaluate(ds_subset))
        
    s = f"{'subset':<8}{'metric':<12}"
    for name in datasets_names:
        s += f"{name:<16}"
    print(s)
    
    for i in range(len(strings)):
        j = 0
        for k in range(len(lossesss[0])):
            if j == 0:
                print("-" * 80)
            s = f"{j:<8}{strings[i]:<12}"
            for lossess in lossesss:
                s += f"{lossess[k][i]:<16.10f}"
            print(s)
            j += 1
    return lossesss

def plot_model_performances(
        lossesss: List[List[List[float]]],
        datasets_names: List[str],
        figsize: tuple,
        rows: int = 1,
        cols: int = None,
        colors: List[str] = [
            "tab:orange",
            "tab:blue",
            "tab:purple",
            "tab:red",
            "tab:green",
            "tab:yellow",
            "tab:pink"
        ],
        save: bool = False,
        dst_files: List[str] = [],
        show: bool = True
    ):
    """
    Create one figure per dataset.

    Inside each figure, create a grid of bar plots,
    one per loss name.

    Parameters
    ----------
    lossesss : List[List[List[float]]]
        Shape = [num_datasets][num_subsets][4 losses].
    datasets_names : List[str]
        Names of datasets (e.g. ["train", "val"]).
    figsize : tuple
    rows : int
        Number of rows per grid.
    cols : int
        Number of columns per grid.
    save : bool
        If True, the plot is saved.
    dst_files : List[str]
        The files where to save the plot.
    show : bool
        If True, the plot is showed.
    
    Returns
    -------
    None
    """
    loss_names = ["out_loss", "der1_loss", "der2_loss", "res_loss"]
    num_losses = len(loss_names)
    num_subsets = len(lossesss[0])

    if rows == 1:
        cols = len(loss_names)

    d_idx = 0
    for ds_name, dst_file in zip(datasets_names, dst_files):

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        fig.suptitle(f"{ds_name}", fontsize=14)

        for l_idx, loss_name in enumerate(loss_names):

            ax = axes[l_idx]

            # Values for bar heights
            values = [lossesss[d_idx][s][l_idx] for s in range(num_subsets)]

            ax.bar(range(num_subsets), values, color=colors[:num_subsets])
            ax.set_title(loss_name)
            ax.set_xlabel("subset index")
            ax.set_ylabel("loss")
            ax.set_xticks(range(num_subsets))

            formatter = ticker.ScalarFormatter(useMathText=True) # useMathText makes it look like 10^n
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1)) # Forces scientific notation for numbers outside this range
            ax.yaxis.set_major_formatter(formatter)

        # Remove unused axes if num_losses < rows*cols
        for i in range(num_losses, rows * cols):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            plt.savefig(dst_file)
        if show:
            plt.show()
        else:
            plt.close()
        d_idx += 1