"""
phy_sys_dataset.py
===========

This module implements the dataset containing the informations relative to a physical system
(subcalssing the torch.utils.data.Dataset class).

Class: PhySysDataset
---
**Attributes:**
    - name: the name of the dataset.
    - cols: a dictionary {(col_name: str, col_values: torch.Tensor)}, where each element represent 
            a column of the dataset and the column vaues are all tensors of the same length, 
            which is the number of rows (# data points);
    - length: the length of the dataset, i.e. the number of rows (# data points);
    - subkeys: some columns may be tensors of vector values (e.g. the vector of spatio-temporal 
               coordinates or the vector of parameters of the physical system); any of the entries 
               of these vectors is associated with a subkey, which is a string; subkeys is a dictionary
               {(col_key: str, subkey_list: List[str])}, where col_key is the column and subkey_list is the 
               associated (ordered) list of subkeys. E.g. for the spatio-temporal coordinates
               col_key = "spacetime", subkeys = ["x", "y", "t"].

**Public methods:**
    - `columns`: None -> List[Tensor].
        It returns the list of column tensors.
    - `get_column`: str, Callable[Tensor, bool] -> Tensor.
        It returns the elements of the column identified by the string for which the condition callable returns True.
    - `add_column`: str, Tensor, List[str] -> None.
        It add the column named with the string, valued with the tensor and with subkeys the list of strings.
    - `set_subkeys`: str, List[str] -> None
        It sets the subkeys for the column identifyed by the string.
    - `index`: str, str -> int.
        It returns the index of the column subkey.
    - `subsample`: Tensor -> PhySysDataset.
        It subsamples the rows/points corresponding to the indices in the tensor.
    - `merge`: PhySysDataset -> None.
        It merges the passed dataset with the one of the object. 
    - `copy`: None -> PhySysDataset.
        Shallow copy method: the tensors representing columns are shared.
    - `deep_copy`: None -> PhySysDataset.
        Deep copy method: Nothing shared.
    - `save`: str -> None
        It stores the dataset in the file identified by the path string.
    - `size_mb`: List[str] -> float.
        It returns the size in megabytes of the dataset.

**Class methods:**
    - load: str -> PhySysDataset.
        It load the dataset in the file identified by the string.

Subclass: AdvectionReactionDiffusionDataset
---
**Additional attributes:**
    - `velocity`: Velocity.
        The velocity function of the advection process.
    - `explicit_source`: Source.
        The explicit source function of the reaction process.
    - `implicit_source`: Source.
        The implicit source function of the reaction process.

"""

import torch
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Self, Dict
from AdvectionReactionDiffusion.advection_velocity import Velocity
from AdvectionReactionDiffusion.reaction_source import Source
from AdvectionReactionDiffusion.boundary_condition import BoundaryCondition, RectangularBoundaryCondition, CircularBoundaryCondition
from AdvectionReactionDiffusion.initial_condition import InitialCondition
import copy

# ===================================== PhySysDataset class =====================================
class PhySysDataset(Dataset):
    # ------------ Subclassing methods ------------
    def __init__(
            self,
            name: str,
            cols: List[Tuple[str, list|torch.Tensor]] | Dict[str, torch.Tensor],
            bc: BoundaryCondition | List[BoundaryCondition] = None,
            ic: InitialCondition | List[InitialCondition] = None,
            timeline: List[float] | List[List[float]] = [],
            shape: str | list[str] = ""
    ) -> None:
        self.name = name
        self.cols = {}
        if type(cols) is list:
            if cols == []:
                self.length = 0
            else:
                self.length = len(cols[0][1])
            for col in cols:
                key, items = col[0], col[1]
                if len(items) != self.length:
                    raise ValueError(f"Length of column {key} is {len(items)}, but is expected to be {self.length}).")
            for col in cols:
                key, items = col[0], col[1]
                self.cols[key] = items
        elif type(cols) is dict:
            if cols == {}:
                self.length = 0
            else:
                self.length = len(list(cols.values())[0])
            for key, col in cols.items():
                if len(col) != self.length:
                    raise ValueError(f"Length of column {key} is {len(col)}, but is expected to be {self.length}.")
            self.cols = cols
        self.subkeys = {}
        self.bc = bc
        self.ic = ic
        self.timeline = timeline
        self.shape = shape

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> dict:
        item = torch.tensor([self.cols[key][idx] for key in self.cols.keys()])
        return item
    
    # ------------ Public methods ------------
    def columns(self) -> List[torch.Tensor]:
        """
        Returns the columns of the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        _List_[_torch.Tensor_]
        - The list of columns.
        """
        return [self.cols[key] for key in self.cols.keys()]
    
    def get_column(self, key: str, condition: Callable[[torch.Tensor], bool] = None) -> torch.Tensor:
        """
        Returns the items of the column `key` satisfying the `condition`.

        Parameters
        ----------
        key : str
            Column identifier.
        condition : Callable[[torch.Tensor], bool] = None
            Condition for the points to satisfy in order to be returned in the column tensor.

        Returns
        -------
        _torch.Tensor_
        """
        if key not in self.cols.keys():
            raise ValueError(f"Column of key {key} not in dataset.")
        col = self.cols[key]
        if condition is None:
            return col
        elif type(col) is torch.Tensor:
            return torch.stack([item for item in col if condition(item)])
        else:
            return [item for item in col if condition(item)]
    
    def add_column(self, key: str, col: torch.Tensor, subkeys: List[str] = []) -> None:
        """
        Add the column `col` of key `key` and with subkeys `subkeys`.

        Parameters
        ----------
        key : str
            Column identifier.
        col : torch.Tensor
            Column vector.
        subkeys : List[str] = []
            Identifiers of the column dimentions.

        Returns
        -------
        _None_
        """
        if len(col) != self.length:
            raise ValueError(f"Column of length {len(col)}, but expected of length {self.length}.")
        if key in self.cols.keys():
            raise ValueError(f"Column {key} already present in the dataset.")
        self.cols[key] = col
        if subkeys != []:
            self.subkeys[key] = subkeys
    
    def set_subkeys(self, key: str, subkeys: List[str]) -> None:
        """
        Set the subkeys `subkeys` for column of key `key`.

        Parameters
        ----------
        key : str
            Column identifier.
        subkeys : List[str]
            Subkeys for the column.

        Returns
        -------
        _None_
        """
        if key not in self.cols.keys():
            raise ValueError(f"Column of key {key} not in dataset.")
        if len(subkeys) != len(self.cols[key][0]):
            raise ValueError(f"Items of column {key} have {len(self.cols[key][0])} elements each, but {len(subkeys)} subkeys are passed.")
        self.subkeys[key] = subkeys
    
    def index(self, key: str, subkey: str) -> int:
        """
        Returns the index of the subkey `subkey` of column `key`.

        Parameters
        ----------
        key : str
            The column key.
        subkey : str
            The column subkey.
        Returns
        -------
        _int_
        - The index of the column subkey.
        """
        for i, sk in enumerate(self.subkeys[key]):
            if sk == subkey:
                return i
        raise ValueError(f"{subkey} not in column {key}.")
    
    def subsample(self, indices: torch.Tensor) -> Self:
        """
        Returns a _PhySysDataset_ with subsampled columns. 
        The subsampled rows are the ones of indexes `indices`.

        Parameters
        ----------
        indices : torch.Tensor
            Indices of the points/rows to subsample.

        Returns
        -------
        PhySysDataset
            Dataset with subsampled columns.
        """
        subsampled_cols = {}
        for key, col in self.cols.items():
            subsampled_cols[key] = col[indices]
        subsampled_ds = PhySysDataset(
            name=self.name,
            cols=subsampled_cols,
            bc=self.bc,
            ic=self.ic,
            timeline=self.timeline,
            shape=self.shape
        )
        subsampled_ds.subkeys = self.subkeys.copy()
        return subsampled_ds
    
    def merge(self, dataset: Self, merge_bc: bool = False, merge_ic: bool = False, merge_timeline: bool = False, merge_shape: bool = False) -> None:
        """
        Merge the _PhySysDataset_ `dataset` the current one.

        Parameters
        ----------
        dataset : PhySysDataset
            Dataset to merge with the current one.

        Returns
        -------
        _None_
        """
        if len(dataset.cols.keys()) != len(self.cols.keys()):
            raise ValueError(f"Different columns to merge: {self.cols.keys()} != {dataset.cols.keys()}.")
        for key, col in dataset.cols.items():
            if key not in self.cols.keys():
                raise ValueError(f"Column {key} not in {self.cols.keys()}.")
            self.cols[key] = torch.cat((self.cols[key], col))
        self.length += dataset.length

        if merge_bc:
            if type(self.bc) is not list:
                bc1 = [self.bc]
            else:
                bc1 = self.bc
            if type(dataset.bc) is not list:
                bc2 = [dataset.bc]
            else:
                bc2 = dataset.bc
            self.bc = bc1 + bc2

        if merge_ic:
            if type(self.ic) is not list:
                ic1 = [self.ic]
            else:
                ic1 = self.ic
            if type(dataset.ic) is not list:
                ic2 = [dataset.ic]
            else:
                ic2 = dataset.ic
            self.ic = ic1 + ic2

        if merge_timeline:
            if len(self.timeline) > 0 and type(self.timeline[0]) is not list:
                timeline1 = [self.timeline]
            else:
                timeline1 = self.timeline
            if len(dataset.timeline) > 0 and type(dataset.timeline[0]) is not list:
                timeline2 = [dataset.timeline]
            else:
                timeline2 = dataset.timeline
            self.timeline = timeline1 + timeline2

        if merge_shape:
            if type(self.shape) is not list:
                shape1 = [self.shape]
            else:
                shape1 = self.shape
            if type(dataset.shape) is not list:
                shape2 = [dataset.shape]
            else:
                shape2 = dataset.shape
            self.shape = shape1 + shape2

    def get_keys(self) -> List[str]:
        return list(self.cols.keys())

    def trajectory(self) -> List[Self]:
        trajectory = []
        for t in self.timeline:
            trajectory.append(self.filter_points(ranges={"t": [t, t]}, mode="closed", shape="rectangle"))
        return trajectory

    def boundary(
            self,
            cell_size: float = 0.0,
            center: list = [0.0, 0.0],
            radius: float = 1.0,
            insert_out_normal: bool = True,
            eps: float = 1e-6
    ) -> Self:
        """
        Extract the boundary points from the dataset for a given time instant.

        Parameters
        ----------
        dataset : ConcatDataset|PhySysDataset
        shape : str
            "rectangle" | "circle".
        cell_size : float
        t : int
            Time index.

        Returns
        -------
        PhySysDataset
            The PhySysDataset containing the boundary points at t.
        """
        spatial_keys = [key for key in self.subkeys["spacetime"] if key != "t"]
        if len(spatial_keys) == 0:
            raise ValueError(f"0-dimentional spatial domain.")
        if self.shape == "rectangle":
            ranges = {}
            for key in spatial_keys:
                x = self.cols["spacetime"][:, self.index("spacetime", key)]
                ranges[key] = [x.min(), x.max()]

            boundary = []
            outward_normal_vectors = []
            for key in spatial_keys:
                xmin = ranges[key][0]
                side = copy.deepcopy(ranges)
                side[key] = [xmin, xmin]
                boundary.append(side)
                if insert_out_normal:
                    if len(spatial_keys) == 1:
                        outward_normal_vectors.append(-1.)
                    else:
                        outward_normal_vector = [0. for _ in spatial_keys]
                        outward_normal_vector[self.index("spacetime", key)] = -1.
                        outward_normal_vectors.append(outward_normal_vector)

                xmax = ranges[key][1]
                side = copy.deepcopy(ranges)
                side[key] = [xmax, xmax]
                boundary.append(side)
                if insert_out_normal:
                    if len(spatial_keys) == 1:
                        outward_normal_vectors.append(1.)
                    else:
                        outward_normal_vector = [0. for _ in spatial_keys]
                        outward_normal_vector[self.index("spacetime", key)] = 1.
                        outward_normal_vectors.append(outward_normal_vector)

                # boundary = [
                #     {"x": [xmin, xmin], "y": [ymin, ymax], "z": [zmin, zmax]},
                #     {"x": [xmax, xmax], "y": [ymin, ymax], "z": [zmin, zmax]},
                #     {"x": [xmin, xmax], "y": [ymin, ymin], "z": [zmin, zmax]},
                #     {"x": [xmin, xmax], "y": [ymax, ymax], "z": [zmin, zmax]}
                # ]

                # outward_normal_vectors = [-1., 1.]
                # outward_normal_vectors = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]
                # outward_normal_vectors = [[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]]

            filtered_ds = self.filter_points(ranges=boundary[0], mode="closed", shape=self.shape, eps=eps)
            if insert_out_normal:
                n_col = torch.tensor(outward_normal_vectors[0]).repeat(filtered_ds.length, 1)

            for side, n in zip(boundary[1:], outward_normal_vectors):
                side_ds = self.filter_points(ranges=side, mode="closed", shape=self.shape, eps=eps)
                filtered_ds.merge(self.filter_points(ranges=side, mode="closed", shape=self.shape, eps=eps))
                if insert_out_normal:
                    n_col = torch.cat((n_col, torch.tensor(n).repeat(side_ds.length, 1)))

            if insert_out_normal:
                filtered_ds.add_column(key="n", col=n_col, subkeys=spatial_keys)

        elif self.shape == "circle":
            for key in spatial_keys:
                boundary[key] = center[self.index("spacetime", key)]
            boundary["r"] = [radius-0.5*cell_size, radius+0.5*cell_size] # [radius-cell_size, radius]
            # boundary = {
            #   "x": center[ix], 
            #   "y": center[iy], 
            #   "z": center[iz], 
            #   "r": [radius-0.5*cell_size, radius+0.5*cell_size]
            # }
            filtered_ds = self.filter_points(ranges=boundary, mode="closed", shape=self.shape, eps=eps)
            if insert_out_normal:
                center = torch.tensor(center).repeat(filtered_ds.length, 1)
                spatial_indexes = [self.index("spacetime", key) for key in spatial_keys]
                out_vect = filtered_ds.cols["spacetime"][:, spatial_indexes] - center
                outward_normal_vectors = out_vect / torch.linalg.norm(out_vect, dim=1, keepdim=True)
                filtered_ds.add_column("n", outward_normal_vectors, spatial_keys, spatial_keys)
        else:
            raise ValueError(f"Unrecognized {self.shape} boundary shape.")

        return filtered_ds

    def interior(
            self,
            cell_size: float = 0.0,
            center: list = [0.0, 0.0],
            radius: float = 1.0,
            eps: float = 1e-6
    ) -> Self:
        """
        Extract the interior points from the dataset for a given time instant.

        Parameters
        ----------
        dataset : ConcatDataset|PhySysDataset
        t : int
            Time index.
        shape : str
            "rectangle" | "circle".
        cell_size : float

        Returns
        -------
        PhySysDataset
            The PhySysDataset containing the interior points at t.
        """
        spatial_keys = [key for key in self.subkeys["spacetime"] if key != "t"]
        ranges = {}
        if self.shape == "rectangle":
            for key in spatial_keys:
                x = self.cols["spacetime"][:, self.index("spacetime", key)]
                ranges[key] = [x.min(), x.max()]
            # ranges = {"x": [xmin, xmax], "y": [ymin, ymax]}
        elif self.shape == "circle":
            for key in spatial_keys:
                ranges[key] = center[self.index("spacetime", key)]
            ranges["r"] = [-1.0, radius-0.5*cell_size]
            # ranges = {"x": center[0], "y": center[1], "r": [-1.0, radius-0.5*cell_size]}
        else:
            raise ValueError(f"Unrecognized {self.shape} boundary shape.")
        return self.filter_points(ranges=ranges, mode="open", shape=self.shape, eps=eps)
    
    def copy(self) -> Self:
        """
        Copy method.

        Parameters
        ----------
        _None_

        Returns
        -------
        _PhySysDataset_
        """
        ds_copy = PhySysDataset(
            name=self.name, 
            cols=self.cols.copy(),
            bc=self.bc,
            ic=self.ic,
            timeline=self.timeline,
            shape=self.shape
        )
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy
    
    def deep_copy(self) -> Self:
        """
        Deep copy method.

        Parameters
        ----------
        _None_

        Returns
        -------
        _PhySysDataset_
        """
        new_cols = {key: col.clone() for key, col in self.cols.items()}
        ds_copy = PhySysDataset(
            name=self.name,
            cols=new_cols,
            bc=self.bc,
            ic=self.ic,
            timeline=self.timeline,
            shape=self.shape
        )
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy

    def filter_points(
            self,
            ranges: dict|List[dict], 
            mode: str,
            shape: str = "rectangle",
            eps: float = 1e-6
    ) -> Self:
        """
        Filter columns keeping elements within ranges and return the relative dataset.

        Parameters
        ----------
        columns : PhySysDataset
        ranges : dict|List[dict]
        mode : str
            Closed or open.
        shape : str
            "rectangle"|"circle", default = "rectangle";

            if "rectangle", each key in spatial_ranges is a model a side;

            if "circle", entry of key "r" is the radius and the other keys encode the center coordinates.

        Returns
        -------
        PhySysDataset
            The filtered dataset.
        """
        if type(ranges) is dict:
            ranges = [ranges]
        masks = []
        for subset in ranges:
            mask = torch.ones(self.length, dtype=bool)
            if shape == "rectangle":
                for key in subset.keys():
                    xmin = subset[key][0] - eps
                    xmax = subset[key][1] + eps
                    x = self.cols["spacetime"][:, self.index(key="spacetime", subkey=key)]
                    if mode == "closed":
                        mask = mask & (x >= xmin) & (x <= xmax)
                    elif mode == "open":
                        mask = mask & (x > xmin) & (x < xmax)
                    else:
                        raise ValueError(f"Unrecognized mode {mode}.")
            elif shape == "circle":
                center_coords = [subset[key] for key in subset.keys() if key != "r"]
                coords_indexes = [self.index(key="spacetime", subkey=key) for key in subset.keys() if key != "r"]
                rmin = subset["r"][0] - eps
                rmax = subset["r"][1] + eps
                x = self.cols["spacetime"][:, coords_indexes]
                center = torch.tensor(
                    center_coords,
                    dtype=x.dtype,
                    device=x.device
                )
                if mode == "closed":
                    mask = mask & (torch.linalg.norm(x - center, axis=1) >= rmin) & (torch.linalg.norm(x - center, axis=1) <= rmax)
                elif mode == "open":
                    mask = mask & (torch.linalg.norm(x - center, axis=1) > rmin) & (torch.linalg.norm(x - center, axis=1) < rmax)
                else:
                    raise ValueError(f"Unrecognized mode {mode}.")
            else:
                raise ValueError(f"Unrecognized shape {shape}.")
            masks.append(mask)  
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m

        cols = {}
        for key in self.cols.keys():
            cols[key] = self.cols[key][mask]

        filtered_dataset = PhySysDataset(name=self.name, cols=cols, bc=self.bc, ic=self.ic, timeline=self.timeline, shape=shape)
        filtered_dataset.subkeys = self.subkeys
        return filtered_dataset
    
    def state_dict(self) -> dict:
        if type(self.bc) is list:
            bcs = [bc.state_dict() for bc in self.bc]
        else:
            bcs = self.bc.state_dict()

        if type(self.ic) is list:
            ics = [ic.state_dict() for ic in self.ic]
        else:
            ics = self.ic.state_dict()
            
        return {
            "name": self.name,
            "cols": self.cols,
            "subkeys": self.subkeys,
            "bc": bcs,
            "ic": ics,
            "timeline": self.timeline,
            "shape": self.shape
        }
    
    def load_state(self, state: dict) -> None:
        self.name = state["name"]
        self.cols = state["cols"]
        for key in state["subkeys"]:
            self.set_subkeys(key=key, subkeys=state["subkeys"][key])

        if type(state["bc"]) is not list:
            state["bc"] = [state["bc"]]
        self.bc = []
        for bc_dict in state["bc"]:
            if bc_dict["shape"] == "rectangle":
                self.bc.append(RectangularBoundaryCondition().load_state(bc_dict))
            else:
                self.bc.append(CircularBoundaryCondition().load_state(bc_dict))
        if len(self.bc) == 1:
            self.bc = self.bc[0]

        if type(state["ic"]) is not list:
            state["ic"] = [state["ic"]]
            self.ic = InitialCondition().load_state(state["ic"])
        else:
            self.ic = []
            for ic_dict in state["ic"]:
                self.ic.append(InitialCondition().load_state(ic_dict))

        self.timeline = state["timeline"]
        self.shape = state["shape"]
    
    def save(self, dst_file: str) -> None:
        """
        Save the dataset in `dst_file` as a dictionary
        {"cols": self.cols, "subkeys": self.subkeys}.

        Parameters
        ----------
        dst_file : str
            Filepath where to save the dataset.

        Returns
        -------
        _None_
        """
        d = self.state_dict()
        torch.save(d, dst_file)

    def size_mb(self, col_ids: List[str] = None) -> float:
        """
        Returns the size in Gb of the dataset.

        Parameters
        ----------
        col_ids : List[str]

        Returns
        -------
        _float_
        """
        total_bytes = 0.0
        if col_ids is None:
            col_ids = list(self.cols.keys())
        for col_str in col_ids:
            col = self.cols[col_str]
            total_bytes += col.element_size() * col.numel()
        return total_bytes / (1024 ** 2)

    def __str__(self) -> str:
        string = f"{self.name}\n"
        string += f"- size: {self.size_mb():.2f} MB\n"
        string += f"- n_rows: {self.length}\n"
        string += f"- n_columns: {len(self.cols.keys())}\n"
        string += f"- columns: {self.cols.keys()}\n"
        string += f"\n- subcolumns:\n"
        for key, value in self.subkeys.items():
            string += f"-- {key}: {value}\n"

        if type(self.bc) is list:
            for bc in self.bc:
                string += f"\n{bc.__str__()}"
        elif self.ic is None:
            string += f"\n- Boundary conditions: None"
        else:
            string += f"\n{self.bc.__str__()}"

        if type(self.ic) is list:
            for ic in self.ic:
                string += f"\n{ic.__str__()}"
        elif self.ic is None:
            string += f"\n- Initial conditions: None"
        else:
            string += f"\n{self.ic.__str__()}"

        if len(self.timeline) > 0 and type(self.timeline[0]) is list:
            string += f"\n- timeline:\n"
            for t_list in self.timeline:
                string += f"-- {t_list}\n"
        else:
            string += f"\n- timeline: {self.timeline}\n"

        if type(self.shape) is list:
            string += f"\n- shape:\n"
            for shape in self.shape:
                string += f"\n-- {shape.__str__()}"
        else:
            string += f"\n- shape: {self.shape.__str__()}"
            
        return string

    def __repr__(self) -> str:
        return self.__str__()
    
    # ------------ Class methods ------------
    @classmethod
    def load(cls, src_file: str) -> Self:
        """
        Load the _PhySysDataset_ saved in `src_file`.

        Parameters
        ----------
        src_file : str
            Filepath of the dataset to load.

        Returns
        -------
        _PhySysDataset_
        """
        state = torch.load(src_file, weights_only=False)

        ds = PhySysDataset(name="", cols={}, bc=None, ic=None, timeline=[], shape="")

        ds.load_state(state=state)

        return ds

class AdvectionReactionDiffusionDataset(PhySysDataset):
    def __init__(
            self,
            name: str,
            cols: List[Tuple[str, list|torch.Tensor]] | Dict[str, torch.Tensor],
            bc: BoundaryCondition | List[BoundaryCondition] = None,
            ic: InitialCondition | List[InitialCondition] = None,
            timeline: List[float] | List[List[float]] = [],
            shape: str | List[str] = "",
            diffusion_coefficient: float | List[float] = None,
            velocity: Velocity | List[Velocity] = None,
            explicit_source: Source | List[Source] = None,
            implicit_source: Source | List[Source] = None
    ):
        super().__init__(
            name=name, 
            cols=cols,
            bc=bc,
            ic=ic,
            timeline=timeline,
            shape=shape
        )

        if diffusion_coefficient is None:
            diffusion_coefficient = 0.0
        if velocity is None:
            velocity = Velocity.null_velocity()
        if explicit_source is None:
            explicit_source = Source.null_source()
        if implicit_source is None:
            implicit_source = Source.null_source()

        self.diffusion_coefficient = diffusion_coefficient
        self.velocity = velocity
        self.explicit_source = explicit_source
        self.implicit_source = implicit_source

    def subsample(self, indices: torch.Tensor) -> Self:
        """
        Returns a _AdvectionReactionDiffusionDataset_ with subsampled columns. 
        The subsampled rows are the ones of indexes `indices`.

        Parameters
        ----------
        indices : torch.Tensor
            Indices of the points/rows to subsample.

        Returns
        -------
        AdvectionReactionDiffusionDataset
            Dataset with subsampled columns.
        """
        subsampled_cols = {}
        for key, col in self.cols.items():
            subsampled_cols[key] = col[indices]
        subsampled_ds = AdvectionReactionDiffusionDataset(
            name=self.name, 
            cols=subsampled_cols, 
            bc=self.bc,
            ic=self.ic,
            timeline=self.timeline,
            shape=self.shape,
            diffusion_coefficient=self.diffusion_coefficient,
            velocity=self.velocity, 
            explicit_source=self.explicit_source, 
            implicit_source=self.implicit_source
        )
        subsampled_ds.subkeys = self.subkeys.copy()
        return subsampled_ds
        
    def merge(
            self, 
            dataset: Self,
            merge_bc: bool = False,
            merge_ic: bool = False,
            merge_timeline: bool = False,
            merge_shape: bool = False,
            merge_diffusion_coefficient: bool = False,
            merge_velocity: bool = False,
            merge_explicit_source: bool = False,
            merge_implicit_source: bool = False
    ) -> None:
        """
        Merge the _AdvectionReactionDiffusionDataset_ `dataset` the current one.

        Parameters
        ----------
        dataset : AdvectionReactionDiffusionDataset
            Dataset to merge with the current one.

        Returns
        -------
        _None_
        """
        #if len(dataset.cols.keys()) != len(self.cols.keys()):
        #    raise ValueError(f"Different columns to merge: {self.cols.keys()} != {dataset.cols.keys()}.")
        #for key, col in dataset.cols.items():
        #    if key not in self.cols.keys():
        #        raise ValueError(f"Column {key} not in {self.cols.keys()}.")
        #    self.cols[key] = torch.cat((self.cols[key], col))
        #self.length += dataset.length
#
        #if merge_bc:
        #    if type(self.bc) is not list:
        #        bc1 = [self.bc]
        #    else:
        #        bc1 = self.bc
        #    if type(dataset.bc) is not list:
        #        bc2 = [dataset.bc]
        #    else:
        #        bc2 = dataset.bc
        #    self.bc = bc1 + bc2
#
        #if merge_ic:
        #    if type(self.ic) is not list:
        #        ic1 = [self.ic]
        #    else:
        #        ic1 = self.ic
        #    if type(dataset.ic) is not list:
        #        ic2 = [dataset.ic]
        #    else:
        #        ic2 = dataset.ic
        #    self.ic = ic1 + ic2
#
        #if merge_timeline:
        #    if len(self.timeline) > 0 and type(self.timeline[0]) is not list:
        #        timeline1 = [self.timeline]
        #    else:
        #        timeline1 = self.timeline
        #    if len(dataset.timeline) > 0 and type(dataset.timeline[0]) is not list:
        #        timeline2 = [dataset.timeline]
        #    else:
        #        timeline2 = dataset.timeline
        #    self.timeline = timeline1 + timeline2
        super().merge(
            dataset=dataset,
            merge_bc=merge_bc,
            merge_ic=merge_ic,
            merge_timeline=merge_timeline,
            merge_shape=merge_shape
        )

        if merge_diffusion_coefficient:
            if type(self.diffusion_coefficient) is not list:
                diffusion_coefficient1 = [self.diffusion_coefficient]
            else:
                diffusion_coefficient1 = self.diffusion_coefficient
            if type(dataset.diffusion_coefficient) is not list:
                diffusion_coefficient2 = [dataset.diffusion_coefficient]
            else:
                diffusion_coefficient2 = dataset.diffusion_coefficient
            self.diffusion_coefficient = diffusion_coefficient1 + diffusion_coefficient2

        if merge_velocity:
            if type(self.velocity) is not list:
                velocity1 = [self.velocity]
            else:
                velocity1 = self.velocity
            if type(dataset.velocity) is not list:
                velocity2 = [dataset.velocity]
            else:
                velocity2 = dataset.velocity
            self.velocity = velocity1 + velocity2

        if merge_explicit_source:
            if type(self.explicit_source) is not list:
                explicit_source1 = [self.explicit_source]
            else:
                explicit_source1 = self.explicit_source
            if type(dataset.explicit_source) is not list:
                explicit_source2 = [dataset.explicit_source]
            else:
                explicit_source2 = dataset.explicit_source
            self.explicit_source = explicit_source1 + explicit_source2

        if merge_implicit_source:
            if type(self.implicit_source) is not list:
                implicit_source1 = [self.implicit_source]
            else:
                implicit_source1 = self.implicit_source
            if type(dataset.implicit_source) is not list:
                implicit_source2 = [dataset.implicit_source]
            else:
                implicit_source2 = dataset.implicit_source
            self.implicit_source = implicit_source1 + implicit_source2

    def filter_points(
            self,
            ranges: dict|List[dict],
            mode: str,
            shape: str = "rectangle",
            eps: float = 1e-6
    ) -> Self:
        ds = super().filter_points(
            ranges=ranges,
            mode=mode,
            shape=shape,
            eps=eps
        )
        return AdvectionReactionDiffusionDataset(
            name=ds.name,
            cols=ds.cols,
            bc=ds.bc,
            ic = ds.ic,
            timeline=ds.timeline,
            shape=ds.shape,
            diffusion_coefficient=self.diffusion_coefficient,
            velocity=self.velocity,
            explicit_source=self.explicit_source,
            implicit_source=self.implicit_source
        )

    def boundary(
            self,
            cell_size: float = 0.0,
            center: list = [0.0, 0.0],
            radius: float = 1.0,
            insert_out_normal: bool = True,
            eps: float = 1e-6
    ) -> Self:
        ds = super().boundary(
            cell_size=cell_size,
            center=center,
            radius=radius,
            insert_out_normal=insert_out_normal,
            eps=eps
        )
        return AdvectionReactionDiffusionDataset(
            name=ds.name,
            cols=ds.cols,
            bc=ds.bc,
            ic = ds.ic,
            timeline=ds.timeline,
            shape=ds.shape,
            diffusion_coefficient=self.diffusion_coefficient,
            velocity=self.velocity,
            explicit_source=self.explicit_source,
            implicit_source=self.implicit_source
        )

    def interior(
            self,
            cell_size: float = 0.0,
            center: list = [0.0, 0.0],
            radius: float = 1.0,
            eps: float = 1e-6
    ) -> Self:
        ds = super().interior(
            cell_size=cell_size,
            center=center,
            radius=radius,
            eps=eps
        )
        return AdvectionReactionDiffusionDataset(
            name=ds.name,
            cols=ds.cols,
            bc=ds.bc,
            ic = ds.ic,
            timeline=ds.timeline,
            shape=ds.shape,
            diffusion_coefficient=self.diffusion_coefficient,
            velocity=self.velocity,
            explicit_source=self.explicit_source,
            implicit_source=self.implicit_source
        )
        
    def copy(self) -> Self:
        """
        Copy method.

        Parameters
        ----------
        _None_

        Returns
        -------
        _AdvectionReactionDiffusionDataset_
        """
        ds_copy = AdvectionReactionDiffusionDataset(
            name=self.name, 
            cols=self.cols.copy(), 
            bc=self.bc,
            ic=self.ic,
            timeline=self.timeline,
            shape=self.shape,
            diffusion_coefficient=self.diffusion_coefficient, 
            velocity=self.velocity, 
            explicit_source=self.explicit_source, 
            implicit_source=self.implicit_source
            )
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy
        
    def deep_copy(self) -> Self:
        """
        Deep copy method.

        Parameters
        ----------
        _None_

        Returns
        -------
        _AdvectionReactionDiffusionDataset_
        """
        new_cols = {key: col.clone() for key, col in self.cols.items()}
        ds_copy = AdvectionReactionDiffusionDataset(
            name=self.name, 
            cols=new_cols, 
            bc=self.bc,
            ic=self.ic,
            timeline=self.timeline,
            shape=self.shape,
            diffusion_coefficient=self.diffusion_coefficient, 
            velocity=self.velocity, 
            explicit_source=self.explicit_source, 
            implicit_source=self.implicit_source
        )
        ds_copy.subkeys = self.subkeys.copy()
        return ds_copy

    def state_dict(self) -> dict:
        if type(self.velocity) is list:
            velocity_entry = [v.state_dict() for v in self.velocity]
        else:
            velocity_entry = self.velocity.state_dict()

        if type(self.explicit_source) is list:
            explicit_source_entry = [s.state_dict() for s in self.explicit_source]
        else:
            explicit_source_entry = self.explicit_source.state_dict()

        if type(self.implicit_source) is list:
            implicit_source_entry = [s.state_dict() for s in self.implicit_source]
        else:
            implicit_source_entry = self.implicit_source.state_dict()

        extra_state = {
            "diffusion_coefficient": self.diffusion_coefficient,
            "velocity": velocity_entry,
            "explicit_source": explicit_source_entry,
            "implicit_source": implicit_source_entry
        }

        return super().state_dict() | extra_state

    def load_state(self, state: dict) -> None:
        super().load_state(state)

        if type(state["velocity"]) is not list:
            state["velocity"] = [state["velocity"]]
            self.velocity = Velocity.null_velocity().load_state(state["velocity"])
        else:
            self.velocity = []
            for velocity_dict in state["velocity"]:
                self.velocity.append(Velocity.null_velocity().load_state(velocity_dict))

        if type(state["explicit_source"]) is not list:
            state["explicit_source"] = [state["explicit_source"]]
            self.explicit_source = Source.null_source().load_state(state["explicit_source"])
        else:
            self.explicit_source = []
            for explicit_source_dict in state["explicit_source"]:
                self.explicit_source.append(Source.null_source().load_state(explicit_source_dict))

        if type(state["implicit_source"]) is not list:
            state["implicit_source"] = [state["implicit_source"]]
            self.implicit_source = Source.null_source().load_state(state["implicit_source"])
        else:
            self.implicit_source = []
            for implicit_source_dict in state["implicit_source"]:
                self.implicit_source.append(Source.null_source().load_state(implicit_source_dict))

    def save(self, dst_file: str) -> None:
        """
        Save the dataset in `dst_file` as a dictionary
        {"cols": self.cols, "subkeys": self.subkeys}.

        Parameters
        ----------
        dst_file : str
            Filepath where to save the dataset.

        Returns
        -------
        _None_
        """
        d = self.state_dict()
        torch.save(d, dst_file)

    def __str__(self) -> str:
        string = super().__str__()
        string += f"\n- Diffusion coefficient: {self.diffusion_coefficient}\n"
        string += f"\n{self.velocity.__str__()}\n"
        string += f"{self.explicit_source.__str__()}\n"
        string += f"{self.implicit_source.__str__()}\n"
        return string

    def __repr__(self) -> str:
        return self.__str__()

    # ------------ Class methods ------------
    @classmethod
    def load(cls, src_file: str) -> Self:
        """
        Load the _AdvectionReactionDiffusionDataset_ saved in `src_file`.

        Parameters
        ----------
        src_file : str
            Filepath of the dataset to load.

        Returns
        -------
        _AdvectionReactionDiffusionDataset_
        """
        state = torch.load(src_file, weights_only=False)
        ds = AdvectionReactionDiffusionDataset(name="", cols={}, bc=None, ic=None, timeline=[], shape="")
        ds.load_state(state)
        return ds