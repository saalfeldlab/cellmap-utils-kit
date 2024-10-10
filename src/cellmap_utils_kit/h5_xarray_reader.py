"""
Module for reading HDF5 datasets and groups as xarrays.

This module provides tools for interacting with HDF5 datasets and groups that store
multiscale metadata in an h5ified version of the OME-NGFF version 0.4 specification.

Key Functions:
--------------
- `read_any_xarray`:
    Closely mimicks fibsem_tools.read_xarray but additionally allows for reading of HDF5
    datasets.
"""

import json
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import h5py
import numpy as np
from dask.array.core import Array as DaskArray
from dask.base import tokenize
from datatree import DataTree
from fibsem_tools import read, read_xarray
from fibsem_tools.type import PathLike
from pydantic_ome_ngff.v04.multiscale import (
    ArraySpec,
    GroupSpec,
    MultiscaleGroup,
    MultiscaleGroupAttrs,
)
from xarray import DataArray, Dataset
from xarray_ome_ngff.array_wrap import BaseArrayWrapper
from xarray_ome_ngff.v04.multiscale import coords_from_transforms, normalize_transforms


# adapted from pydantic_zarr.v2.GroupSpec
class HDFGroupSpec(GroupSpec):
    """A model of a HDF Group."""

    @classmethod
    def from_hdf(cls, group: h5py.Group, *, depth: int = -1) -> "HDFGroupSpec":
        """
        Create a HDFGroupSpec from an instance of a `h5py.Group`. Subgroups and arrays
        contained in the Zarr group will be converted to instances of `GroupSpec` and
        `ArraySpec`, respectively, and these spec instances will be stored in the
        `members` attribute of the parent `GroupSpec`.

        This is a recursive function, and the depth of the recursion can be controlled
        by the `depth` keyword argument. The default value for `depth` is -`, which
        directs this function to traverse the entirety of a `zarr.Group`. This may be
        slow for large hierarchies, in which case setting `depth` to a positive integer
        can limit how deep into the hierarchy the recursion goes.

        Args:
            group (h5py.Group): The HDF5 Group to model.
            depth (int, optional): An integer which may be no lower than -1. Determines
                how far into the tree to parse. Defaults to -1.

        Raises:
            ValueError: If depth is smaller than -1.
            ValueError: If group contains elements that are unparsable.

        Returns:
            HDFGroupSpec: Instance of `HDFGroupSpec` describing the HDF5 group's
                hierarchy.

        """
        attributes = {}
        for key, value in group.attrs.items():
            attributes[key] = json.loads(value)
        members = {}
        if depth == 0:
            return cls(attributes=attributes, members=None)
        if depth < -1:
            msg = (
                f"Invalid value for depth. Got {depth}, expected an integer "
                "greater than or equal to -1."
            )
            raise ValueError(msg)
        new_depth = max(depth - 1, -1)
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                item_out = ArraySpec.from_array(item).model_dump()
            elif isinstance(item, h5py.Group):
                item_out = HDFGroupSpec.from_hdf(item, depth=new_depth).model_dump()
            else:
                msg = (
                    f"Unparseable object encountered: {type(item)}. Expected "
                    "`h5py.Dataset` or `h5py.Group`."
                )
                raise ValueError(msg)
            members[name] = item_out
        result = cls(attributes=attributes, members=members)
        return result


# adapted form pydantic_ome_ngff.v04.multiscale.MultiscaleGroup
class HDFMultiscaleGroup(MultiscaleGroup, HDFGroupSpec):
    """A model of an HDF5 group that implements the h5ified OME-NGFF multiscales
    metadata.

    Attributes:
        attributes (GroupAttrs): The attributes of this HDF5 group, which should
            contain valid `GroupAttrs`.
        members (dict[str, ArraySpec | HDFGroupSpec]): The members of this HDF5 group.
            Should be instances of `HDFGroupSpec` or `ArraySpec`.

    """

    @classmethod
    def from_hdf(cls, node: h5py.Group, *, depth: int = 0) -> "HDFMultiscaleGroup":  # noqa: ARG003
        """Create an instance of `HDFMultiscaleGroup` from a `h5py.Group`. This method
        discovers h5py datasets in the group by inspecting the OME-NGFF multiscales
        metadata.

        Args:
            node (h5py.Group): A HDFGroup that has valid h5ified OME-NGFF multiscale
                metadata.
            depth (int, optional): This parameter is not actually used and only here to
                match the signature of the superclass.

        Returns:
            HDFMultiscaleGroup: An instance of `HDFMultiscaleGroup` that describes the
                multiscale hierarchy.

        """
        guess = HDFGroupSpec.from_hdf(node, depth=0)
        multi_meta = MultiscaleGroupAttrs(
            multiscales=json.loads(node.attrs["multiscales"])
        )
        members_tree_flat = {}
        for multiscale in multi_meta.multiscales:
            for dataset in multiscale.datasets:
                array = node[dataset.path]
                array_spec = ArraySpec.from_array(array)
                members_tree_flat["/" + dataset.path] = array_spec
        members_normalized = GroupSpec.from_flat(members_tree_flat)
        guess_inferred_members = guess.model_copy(
            update={"members": members_normalized.members}
        )
        return cls(**guess_inferred_members.model_dump())


def read_any_xarray(
    path: str | Path,
    *,
    chunks: Literal["auto"] | tuple[int, ...] = "auto",
    coords: Literal["auto"] | dict[Hashable, Any] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> DataTree | DataArray:
    """Extension of fibsem_tools.read_xarray that can also interpret hdf5 data.

    For HDF5 data the OME-NGFF metadata is used but encoded as a single json string in
    the HDF5 dataset's `multiscales` attributes.

    path (str | Path): The path to the array to load.
    chunks (Literal["auto"] | tuple[int, ...], optional): The chunks to use for the
        arrays in the tree. Ignored if `use_dask` is `False`. Defaults to "auto".
    coords (Any, optional): If set to "auto" assumes all datasets to be part of
        h5ified OME-NGFF hierarchy and infers xarray coordinates from that.
        Otherwise, needs to be parasable as `coords` kwarg for DataArray and all
        arrays will have the same coordinates. Defaults to "auto".
    use_dask (bool, optional): Whether to wrap the DataArrays in dask arrays.
        Defaults to True.
    attrs (dict[str, Any] | None, optional): Attributes to add to the `element`'s
        node in the tree. Defaults to None in which case it will be read and decoded
        from the HDF5 attributes.
    name (str | None, optional): Name of this node in the tree. Defaults to None.
    kwargs (Any): Additional keyword arguments passed to `read` and
        `create_dataelement`.

    Raises:
        ValueError: If multiscales is not supported for the given path.

    Returns:
        DataTree | DataArray: If path points to an array, it will be returned as an
        `xarray.DataArray`. If it is a collection of arrays, it will be returned as a
        `DataTree`.

    """
    try:
        return read_xarray(
            path,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
            **kwargs,
        )
    except ValueError as e:
        err_str = str(e)
        if "h5" in err_str:
            return read_h5_xarray(
                path,
                chunks=chunks,
                coords=coords,
                use_dask=use_dask,
                attrs=attrs,
                name=name,
                **kwargs,
            )
        else:
            raise e


def get_url(node: h5py.Dataset) -> str:
    """Make url style representation of h5py.Dataset.

    Args:
        node (h5py.Dataset): The h5py.Dataset

    Returns:
        str: URL representation of h5py.Dataset

    """
    protocol = "file"
    store_path = node.file.filename + node.name
    return f"{protocol}://{store_path}"


# adapted from fibsem_tools.io.zarr.core.to_dask
def to_dask(
    arr: h5py.Dataset,
    *,
    chunks: Literal["auto", "inherit"] | tuple[int, ...] = "auto",
    inline_array: bool = True,
    **kwargs: Any,
) -> da.Array:
    """
    Create a dask array from a HDF5 dataset. This is a very thin wrapper around
    `dask.array.from_array`.

    Args:
        arr (h5py.Dataset): HDF5 dataset to turn into dask array.
        chunks (Literal["auto", "inherit"] | tuple[int, ...], optional): The chunks to
            use for the output Dask array. "inherit" allows to use the chunk size of the
            input array for this parameter, but be advised that Dask performance suffers
            when arrays have too many chunks, and h5py Datasets routinely have too many
            chunks by this definition. Defaults to "auto".
        inline_array (bool, optional): Whether the h5py.Dataset should be inlined in the
            Dask compute graph. See documentation for `dask.array.from_array` for
            details. Defaults to True.
        **kwargs (Any): Additional keyword arguments to `dask.array.from_array`

    Returns:
        da.Array: The dask array of the HDF5 dataset.

    """
    if kwargs.get("name") is None:
        kwargs["name"] = f"{get_url(arr)}-{tokenize(arr)}"
    if chunks == "inherit":
        chunks = arr.chunks
    return da.from_array(arr, chunks=chunks, inline_array=inline_array, **kwargs)


# adapted from xarray_ome_ngff.array_wrap.ZarrArrayWrapper
@dataclass
class HDFArrayWrapper(BaseArrayWrapper):
    """An Array wrapper that passes `h5py.Dataset` instances through unchanged."""

    def wrap(self, data: h5py.Dataset) -> np.ndarray:
        """Read the HDF5 Dataset.

        Args:
            data (h5py.Dataset): Data to be read

        Returns:
            np.ndarray: numpy array of `data`

        """
        return data[:]


# adapted from xarray_ome_ngff.array_wrap.ZarrDaskArrayWrapper
@dataclass
class HDFDaskArrayWrapper(BaseArrayWrapper):
    """An Array wrapper that wraps `h5py.Dataset` in a dask array using
    `dask.array.from_array`. The attributes of this class are a subset of the keyword
    arguments ot `dask.array.from_array`; specifically, those keyword arguments that
    make sense when input to `from_array` is a `h5py.Dataset`.
    """

    chunks: str | int | tuple[int, ...] | tuple[tuple[int, ...], ...] = "auto"
    meta: Any = None
    inline_array: bool = True
    naming: Literal["auto", "array_url"] = "array_url"

    def wrap(self, data: h5py.Dataset) -> DaskArray:
        """Wrap the HDF5 Dataset in a dask array.

        Args:
            data (h5py.Dataset): Data to be read

        Returns:
            DaskArray: Dask Array of `data`.

        """
        if self.naming == "auto":
            name = None
        elif self.naming == "array_url":
            name = f"{get_url(data)}"
        return da.from_array(
            data,
            chunks=self.chunks,
            inline_array=self.inline_array,
            meta=self.meta,
            name=name,
        )


# adapted from xarray_ome_ngff.v04.multiscale
def read_multiscale_array(
    array: h5py.Dataset,
    array_wrapper: HDFArrayWrapper | HDFDaskArrayWrapper | None = None,
) -> DataArray:
    """Read a single HDF5 dataset as an `xarray.DataArray`, using a h5ification of
    version 0.4 OME-NGFF multiscale metadata.

    The information necessary for creating the coordinates of the `DataArray` are not
    stored in the attributes of the HDF5 dataset given to this function. Instead, the
    coordinates must be inferred by walking up the HDF5 hierarchy, group by group, until
    a HDF5 group with attributes containing OME-NGFF multiscales metadata is found; then
    that metadata is parsed to determine whether the metadata references the provided
    dataset. Once the correct multiscales metadata is found, the coordinates can be
    constructed correctly.

    Args:
        array (h5py.Dataset): A HDF5 dataset that is part of a h5ified 0.4 OME-NGFF
            multiscale image.
        array_wrapper (HDFArrayWrapper | HDFDaskArrayWrapper | None, optional): The
            array wrapper class to use when converting the HDF5 dataset to an
            `xarray.DataArray`. Defaults to None, which then uses a default
            `HDFArrayWrapper`.

    Raises:
        FileNotFoundError: If no h5ified 0.4 OME-NGFF multiscale metadata is found.

    Returns:
        DataArray: The data from the HDF5 dataset as an xarray.DataArray with
            coordinates inferred from metadata.

    """
    if array_wrapper is None:
        array_wrapper = HDFArrayWrapper()
    node = array
    for _ in range(array.name.count("/") + 1):
        try:
            model = HDFMultiscaleGroup.from_hdf(node.parent)
            for multi in model.attributes.multiscales:
                multi_tx = multi.coordinateTransformations
                for dset in multi.datasets:
                    if dset.path == array.name.split("/")[-1]:
                        tx_fused = normalize_transforms(
                            multi_tx, dset.coordinateTransformations
                        )
                        coords = coords_from_transforms(
                            axes=multi.axes, transforms=tx_fused, shape=array.shape
                        )
                        array = array_wrapper.wrap(array)

                        return DataArray(array, coords=coords)
        except KeyError:
            node = node.parent
    msg = (
        "Could not find version 0.4 OME-NGFF multiscale metadata in any HDF group"
        f"ancestral to the array at {array.name}"
    )
    raise FileNotFoundError(msg)


# adapted from fibsem_tools.io.zarr.hierarchy.ome_ngff.create_dataarray
def create_inferred_dataarray(
    element: h5py.Dataset,
    *,
    use_dask: bool = True,
    chunks: tuple[int, ...] | Literal["auto", "inherit"] = "auto",
    name: str | None = None,
) -> DataArray:
    """Create a DataArray from a HDF5 dataset with h5ified OME-NGFF version 0.4
    metadata.

    Args:
        element (h5py.Dataset): The HDF5 dataset
        use_dask (bool, optional): Whether to wrap the result in a dask array. Defaults
            to True.
        chunks (tuple[int, ...] | Literal["auto", "inherit"], optional): The chunks to
            use for the returned array.. Ignored if `use_dask` is `False`. Defaults to
            "auto".
        name (str | None, optional): The name of the resulting array. Defaults to None,
            in which case it gets set automagically.

    Returns:
        DataArray: The resulting xarray.DataArray

    """
    if use_dask:
        wrapper = HDFDaskArrayWrapper(chunks=chunks)
    else:
        wrapper = HDFArrayWrapper()

    result = read_multiscale_array(array=element, array_wrapper=wrapper)
    if name is not None:
        result.name = name
    return result


# adapted from fibsem_tools.io.zarr.core.create_dataarray
def create_dataarray(
    element: h5py.Dataset,
    *,
    chunks: tuple[int, ...] | Literal["auto", "inherit"] = "auto",
    coords: Literal["auto"] | Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataArray:
    """Create an xarray.DataArray from a HDF5 Dataset.

    Args:
        element (h5py.Dataset): The HDF5 Dataset
        chunks (tuple[int, ...] | Literal["auto", "inherit"], optional): The chunks to
            use for the returned array. Ignored if `use_dask` is `False`. Defaults to
            "auto".
        coords (Any, optional): If set to "auto" assumes dataset to be part of a h5ified
            OME-NGFF hierarchy and infers xarray coordinates from that. Otherwise, needs
            to be parasable as `coords` kwarg for DataArray. Defaults to "auto".
        use_dask (bool, optional): Whether to wrap the result in a dask array. Defaults
            to True.
        attrs (dict[str, Any] | None, optional): Attributes to add to the DataArray.
            Defaults to None.
        name (str | None, optional): The name of the resulting array. Defaults to None,
            in which case it gets set automagically.

    Returns:
        DataArray: The resulting xarray.DataArray

    """
    if coords == "auto":
        return create_inferred_dataarray(
            element, use_dask=use_dask, chunks=chunks, name=name
        )

    wrapped = to_dask(element, chunks=chunks) if use_dask else element
    return DataArray(wrapped, coords=coords, attrs=attrs, name=name)


# adapted from fibsem_tools.io.zarr.core.create_datatree
def create_datatree(
    element: h5py.Group,
    *,
    chunks: Literal["auto", "inherit"] | tuple[int, ...] = "auto",
    coords: Any = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataTree:
    """Create a DataTree from a HDF5 Group.

    Args:
        element (h5py.Group): The HDF5 Group
        chunks (Literal["auto", "inherit"] | tuple[int, ...], optional): The chunks to
            use for the arrays in the tree. Ignored if `use_dask` is `False`. Defaults
            to "auto".
        coords (Any, optional): If set to "auto" assumes all datasets to be part of
            h5ified OME-NGFF hierarchy and infers xarray coordinates from that.
            Otherwise, needs to be parasable as `coords` kwarg for DataArray and all
            arrays will have the same coordinates. Defaults to "auto".
        use_dask (bool, optional): Whether to wrap the DataArrays in dask arrays.
            Defaults to True.
        attrs (dict[str, Any] | None, optional): Attributes to add to the `element`'s
            node in the tree. Defaults to None in which case it will be read and decoded
            from the HDF5 attributes.
        name (str | None, optional): Name of this node in the tree. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        DataTree: _description_

    """
    if coords != "auto":
        msg = (
            "This function does not support values of `coords` other than `auto`. "
            f"Got {coords=}."
        )
        raise NotImplementedError(msg)
    if name is None:
        name = Path(element.name).name
    if name == "":
        name = None
    # for child in element.values():
    #    print(Path(child.name).name)
    nodes = {
        Path(child.name).name: create_dataelement(
            child,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=None,
            name="data",
        )
        for child in element.values()
    }
    if attrs is None:
        attrs = {}
        for attr, val in element.attrs.items():
            attrs[attr] = json.loads(val)
    nodes["/"] = Dataset(attrs=attrs)

    return DataTree.from_dict(nodes, name=name)


# adapted from fibsem_tools.io.zarr.core.to_xarray
def create_dataelement(
    element: h5py.Dataset | h5py.Group,
    *,
    chunks: Literal["auto", "inherit"] | tuple[int, ...] = "auto",
    coords: Literal["auto", "inherit"] | dict[Hashable, Any] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
) -> DataTree | DataArray:
    """Reads HDF5 Dataset and HDF5 Groups as a DataArray and a DataTree, respectively.

    Args:
        element (h5py.Dataset | h5py.Group): The HDF5 Dataset or HDF5 Group
        chunks (Literal["auto", "inherit"] | tuple[int, ...], optional): The chunks to
            use for the arrays in the tree. Ignored if `use_dask` is `False`. Defaults
            to "auto".
        coords (Any, optional): If set to "auto" assumes all datasets to be part of
            h5ified OME-NGFF hierarchy and infers xarray coordinates from that.
            Otherwise, needs to be parasable as `coords` kwarg for DataArray and all
            arrays will have the same coordinates. Defaults to "auto".
        use_dask (bool, optional): Whether to wrap the DataArrays in dask arrays.
            Defaults to True.
        attrs (dict[str, Any] | None, optional): Attributes to add to the `element`'s
            node in the tree. Defaults to None in which case it will be read and decoded
            from the HDF5 attributes.
        name (str | None, optional): Name of this node in the tree. Defaults to None.

    Returns:
        DataTree | DataArray: The resulting DataArray or DataTree (with leaf nodes that
            are `DataArray`s)

    """
    if isinstance(element, h5py.Group):
        return create_datatree(
            element,
            chunks=chunks,
            coords=coords,
            use_dask=use_dask,
            attrs=attrs,
            name=name,
        )
    return create_dataarray(
        element, chunks=chunks, coords=coords, use_dask=use_dask, attrs=attrs, name=name
    )


def read_h5_xarray(
    path: PathLike,
    *,
    chunks: Literal["auto", "inherit"] | tuple[int, ...] = "auto",
    coords: Literal["auto", "inherit"] | dict[Hashable, Any] = "auto",
    use_dask: bool = True,
    attrs: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> DataTree | DataArray:
    """Imitates fibsem_tools.read_xarray specifically for HDF5 datasets or groups.

    Args:
        path (str): Path pointing to a HDF5 Dataset or Group
        chunks (Literal["auto"] | tuple[int, ...], optional): The chunks to use for the
            arrays in the tree. Ignored if `use_dask` is `False`. Defaults to "auto".
        coords (Any, optional): If set to "auto" assumes all datasets to be part of
            h5ified OME-NGFF hierarchy and infers xarray coordinates from that.
            Otherwise, needs to be parasable as `coords` kwarg for DataArray and all
            arrays will have the same coordinates. Defaults to "auto".
        use_dask (bool, optional): Whether to wrap the DataArrays in dask arrays.
            Defaults to True.
        attrs (dict[str, Any] | None, optional): Attributes to add to the `element`'s
            node in the tree. Defaults to None in which case it will be read and decoded
            from the HDF5 attributes.
        name (str | None, optional): Name of this node in the tree. Defaults to None.
        kwargs (Any) : Additional keyword arguments passed on to `fibsem_tools.read`.

    Returns:
        DataTree | DataArray: Resulting DataArray (for HDF5 Dataset) or DataTree (for
            HDF5 Group)

    """
    element = read(path, **kwargs)
    return create_dataelement(
        element,
        chunks=chunks,
        coords=coords,
        use_dask=use_dask,
        attrs=attrs,
        name=name,
    )
