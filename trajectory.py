"""
trajectory.py
===========

This module implements the trajectory class, instantiable in trajectory objects representing trajectories of dynamical systems.\n
For the moment, it considers advection-reaction-diffusion systems (allowing trajectories with velocities and sources).

Functions:
- `stream_spacetime_samples`: Subsample a sequence of frames uniformly at random.

Classes:
- `Trajectory`: Implements a trajectory object.
"""

import torch
from typing import List
from derivatives_computation import derivative
from collections import deque
import numpy as np

def stream_spacetime_samples(v_slice: int, nt: int, n_samples: int, seed: int = 42, device: str = "cpu") -> List[torch.Tensor]:
    """
    Sample `n_samples` uniform random indexes along an `nt`-long stream of frames.\n
    Each frame is a grid of `v_slice` points and any index correspond to a point.

    Parameters
    ----------
    v_slice : int
        Volume/area (number of grid points) of a slice/frame (nx * ny).
    nt : int
        Length of the stream (trajectory).
    n_samples : int
        Number of uniform random samples to return.
    seed : int
        Seed for reproducibility.
    device : str

    Returns
    -------
    _List[torch.Tensor[int]]_
    - A list with a uniformly at random subsampled frame for each time instant of the trajectory.
    """
    # Seed NumPy generator for hypergeometric draws (np.random.hypergeometric)
    np_rng = np.random.default_rng(seed)
    # Seed PyTorch generator for spatial sampling (torch.randperm)
    torch_gen = torch.Generator(device=device).manual_seed(seed)

    n_rem = n_samples  # Remaining samples needed
    collected_batches = [] # List of sampled batches (one item per frame)
    
    for t in range(nt):
        if n_rem <= 0:
            break

        # Remaining spacetime volume
        v_rem = v_slice * (nt - t)
        
        # Sample the number of samples to draw for time t from an hypergeometric distribution
        n_samples_t = np.random.hypergeometric(
            ngood=v_slice, # This time slice
            nbad=v_rem - v_slice, # Future volume
            nsample=n_rem # Remaining samples needed
        )
        
        if n_samples_t > 0:
            # Uniformly select n_samples_t unique spatial indices
            spatial_indices = torch.randperm(v_slice, device=device)[:n_samples_t]

            # Append the subsampled frame of indexes
            collected_batches.append(spatial_indices)

        # Remaining samples needed
        n_rem -= n_samples_t

    return collected_batches

# ===================================== Trajectory class =====================================
class Trajectory:
    """
    Class for the trajectory of a (advection-reaction-diffusion) dynamical system.

    Attributes
    ----------
    shape : str
        Shape of the spatial domain of the system ("rectangle" or "circle").
    status : str
        Status of the trajectory:
        - "open" if the trajectory hasn't reached the termination.
        - "closed" if the trajectory has reached the terminal state.
    f : List[torch.Tensor]
        The evolving (scalar) function field (the unknown function of the corresponding IBVP), list of subsampled frames.
    df : List[torch.Tensor]
        The evolving f-gradient (vector) field, list of subsampled frames.
    d2f : List[torch.Tensor]
        The evolving f-2nd derivative (vector) field, list of subsampled frames.
    velocity : List[torch.Tensor]
        The evolving velocity (vector) field, list of subsampled frames.
    source : List[torch.Tensor]
        The evolving source (scalar) field, list of subsampled frames.
    x : List[torch.Tensor]
        The x coordinates subsampled for each frame.
    y : List[torch.Tensor]
        The y coordinates subsampled for each frame.
    t : List[torch.Tensor]
        List of trajectory simulated times.
    nx : int
        Number of grid points along the horizontal x-side.
    ny : int
        Number of grid points along the vertical y-side.
    nt : int
        Number of points along the timeline.
    dx : float
        Spatial resolution (distance btw adjacent points in a frame).
    dt : float
        Temporal resolution (distance btw adjacent points in the timeline).
    nt_count : int
        The number of points along the timeline that have been simulated.
    snapshot_times : List[float]
        List of points along the timeline where the full (not subsampled) frame has to be stored.
    f_full : List[torch.Tensor]
        List of full snapshots of the evolving (scalar) function field (list of not subsampled frames).
    df_full : List[torch.Tensor]
        List of full snapshots of the evolving f-gradient (vector) field (list of not subsampled frames).
    d2f_full : List[torch.Tensor]
        List of full snapshots of the evolving f-2nd derivative (vector) field (list of not subsampled frames).
    velocity_full : List[torch.Tensor]
        List of full snapshots of the evolving velocity (vector) field (list of not subsampled frames).
    source_full : List[torch.Tensor]
        List of full snapshots of the evolving source (scalar) field (list of not subsampled frames).
    t_full : List[torch.Tensor]
        List of timeline points where a full snapshot has been stored.
    x_full : torch.Tensor
        x-coordinates of the full grid (shape (nx * ny,)).
    y_full : torch.Tensor
        y-coordinates of the full grid (shape (nx * ny,)).
    """
    def __init__(
            self,
            x_coords: torch.Tensor | np.ndarray,
            y_coords: torch.Tensor | np.ndarray,
            nt: int,
            dt: float,
            n_samples: int,
            snapshot_times: List[float],
            shape: str = "rectangle",
            seed: int = 42,
            nx: int = None,
            ny: int = None,
            dx: float = None
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        x_coords : torch.Tensor
            x-coordinates of the full grid (shape (nx * ny,)).
        y_coords : torch.Tensor
            y-coordinates of the full grid (shape (nx * ny,)).
        nt : int
            Number of points along the timeline.
        dt : float
            Temporal resolution (distance btw adjacent points in the timeline).
        n_samples : int
            Total number of trajectory samples to store (len(self.f)).
        snapshot_times : List[float]  
            Points along the timeline where to take a full snapshot of the trajectory.
        shape : str
            Shape of the spatial domain of the system ("rectangle" or "circle").
        seed : int
            Seed for the uniform random sampling of the n_samples points along the trajectory.
        nx : int
            Number of grid points along the horizontal x-side.
        ny : int
            Number of grid points along the vertical y-side.
        dx : float
            Spatial resolution (distance btw adjacent points in a frame).

        Returns
        -------
        None
        """
        if isinstance(x_coords, np.ndarray):
            x_coords = torch.from_numpy(x_coords)
        if isinstance(y_coords, np.ndarray):
            y_coords = torch.from_numpy(y_coords)

        if shape == "rectangle":
            if nx is None:
                raise ValueError(f"nx argument is required for rectangular shape.")
            if ny is None:
                raise ValueError(f"ny argument is required for rectangular shape.")
            if dx is None:
                raise ValueError(f"dx argument is required for rectangular shape.")
        self.shape = shape
        self.status = "open"
        self.f = []
        self.df = []
        self.d2f = []
        self.velocity = []
        self.source = []
        self.x = []
        self.y = []
        self._f_buf = deque(maxlen=3)
        self._df_buf = deque(maxlen=3)
        self._d2f_buf = deque(maxlen=3)
        self._velocity_buf = deque(maxlen=3)
        self._source_buf = deque(maxlen=3)
        self.x_full = x_coords
        self.y_full = y_coords
        self.t = []
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.nt_count = 0
        self.snapshot_times = snapshot_times
        self._pending_snapshots = []
        self._sampled_indexes = stream_spacetime_samples(v_slice=len(self.x_full), nt=nt, n_samples=n_samples, seed=seed)
        self.f_full = []
        self.df_full = []
        self.d2f_full = []
        self.velocity_full = []
        self.source_full = []
        self.t_full = []

    def _subsample(self, tensor_list: List[torch.Tensor], t_index: int) -> List[torch.Tensor]:
        """
        Subsample each tensor in `tensor_list`, according to the `t_index` set of indexes.

        Parameters
        ----------
        tensor_list : List[torch.Tensor]
            Tensors to subsample.
        t_index : int
            Index of the set of indexes in `self._sampled_indexes`.
        
        Returns
        -------
        List[torch.Tensor]
        - The subsampled tensors.
        """
        return [tensor[self._sampled_indexes[t_index]] for tensor in tensor_list]
    
    def append(
            self,
            t: float,
            f_snapshot: torch.Tensor | np.ndarray,
            velocity_snapshot: torch.Tensor | np.ndarray,
            source_snapshot: torch.Tensor | np.ndarray
    ) -> None:
        """
        Append a new frame to the trajectory.

        Parameters
        ----------
        t : float
            Temporal coordinate of the frame to append.
        f_snapshot : torch.Tensor | np.ndarray
            Full f field of the xy frame to append.
        velocity_snapshot : torch.Tensor | np.ndarray
            Full velocity field of the xy frame to append.
        source_snapshot : torch.Tensor | np.ndarray
            Full source field of the xy frame to append.

        Returns
        -------
        _None_
        """
        if self.status == "closed":
            raise ValueError(f"Trajectory closed.")
        if isinstance(f_snapshot, np.ndarray):
            f_snapshot = torch.from_numpy(f_snapshot)
        if isinstance(velocity_snapshot, np.ndarray):
            velocity_snapshot = torch.from_numpy(velocity_snapshot)
        vx = velocity_snapshot[0]
        vy = velocity_snapshot[1]
        velocity_snapshot = torch.stack([vx, vy], dim=1)
        if isinstance(source_snapshot, np.ndarray):
            source_snapshot = torch.from_numpy(source_snapshot)
        t = torch.tensor(t, device=f_snapshot.device)

        #if self.shape == "rectangle":
        #    f_snapshot = f_snapshot.reshape(self.nx, self.ny)
        self.t.append(t)

        self.nt_count += 1
        self._f_buf.append(f_snapshot)
        self._velocity_buf.append(velocity_snapshot)
        self._source_buf.append(source_snapshot)

        if self.nt_count >= 3:
            if self.nt_count == 3: indexes = [0, 1]
            else: indexes = [1]

            for i in indexes:
                x, y, f, velocity, source = self._subsample(tensor_list=[self.x_full, self.y_full, self._f_buf[i], self._velocity_buf[i], self._source_buf[i]], t_index=self.nt_count-1)

                self.x.append(x)
                self.y.append(y)
                self.f.append(f)
                self.velocity.append(velocity)
                self.source.append(source)

                if self.shape == "rectangle":

                    self._df_buf.append(derivative(f=self._f_buf, nx=self.nx, ny=self.ny, dx=self.dx, dt= self.dt, order=1)[i])
                    self._d2f_buf.append(derivative(f=self._f_buf, nx=self.nx, ny=self.ny, dx=self.dx, dt= self.dt, order=2)[i])
                    df, d2f = self._subsample(tensor_list=[self._df_buf[i], self._d2f_buf[i]], t_index=self.nt_count-1)

                    self.df.append(df)
                    self.d2f.append(d2f)

                if self.nt_count == self.nt:
                    self._close()

            if len(self._pending_snapshots) > 0:
                self._store_full_snapshot()

        if self.snapshot_times != [] and abs(t - torch.tensor(list(self.snapshot_times))).min() < 1e-2:
            self._pending_snapshots.append(self.nt_count-1)
            if self.status == "closed":
                self._store_full_snapshot()

    def _close(self) -> None:
        """
        Close a completed trajectory (a trajectory which has reached the end of its timeline).
        """
        x, y, f, velocity, source = self._subsample(tensor_list=[self.x_full, self.y_full, self._f_buf[-1], self._velocity_buf[-1], self._source_buf[-1]], t_index=self.nt_count-1)

        self.x.append(x)
        self.y.append(y)
        self.f.append(f)
        self.velocity.append(velocity)
        self.source.append(source)

        if self.shape == "rectangle":

            self._df_buf.append(derivative(f=self._f_buf, nx=self.nx, ny=self.ny, dx=self.dx, dt= self.dt, order=1)[-1])
            self._d2f_buf.append(derivative(f=self._f_buf, nx=self.nx, ny=self.ny, dx=self.dx, dt= self.dt, order=2)[-1])

            df, d2f = self._subsample(tensor_list=[self._df_buf[-1], self._d2f_buf[-1]], t_index=self.nt_count-1)
            self.df.append(df)
            self.d2f.append(d2f)
        
        self.status = "closed"

    def _store_full_snapshot(self) -> None:
        """
        Append full snapshots to the trajectory according to the content of `self._pending_snapshots`.
        """
        for ti in self._pending_snapshots:
            self.t_full.append(self.t[ti])

            if ti == 0: i = 0
            elif self.status == "open": i = 1
            else: i = -1

            self.f_full.append(self._f_buf[i])
            self.velocity_full.append(self._velocity_buf[i])
            self.source_full.append(self._source_buf[i])
            if self.shape == "rectangle":
                self.df_full.append(self._df_buf[i])
                self.d2f_full.append(self._d2f_buf[i])

        self._pending_snapshots = []