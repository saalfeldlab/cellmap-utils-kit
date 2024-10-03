"""
Provide a command-line interface (CLI) for data preparation for cellmap
data. The CLI allows for various operations such as copying crops, checking YAML data
configurations, generating multiscale pyramids, exporting data, filtering crops, and
correcting attributes. The commands available in the CLI are intended to streamline and
automate data preprocessing workflows.

Commands:
    - copy-crops: Copy crops specified in a data YAML file to a new destination.
    - check-data-yaml: Validate data configuration YAML for correctness.
    - multiscale-labels: Generate a multiscale pyramid for label data.
    - add-raw: Add cropped raw data to crops.
    - multiscale-raw: Generate a multiscale pyramid for raw data.
    - h5-export: Export crops from Zarr to HDF5 format.
    - correct-attrs: Correct label attributes within the data configuration.
    - filter-yaml: Filter the list of crops based on various criteria.

The package provides this cli under data-prep. Each command can be accessed through the
main script. Run the script + command with the `--help` option to view detailed usage
for each command, e.g. `data-prep copy-crops --help`
"""

import logging
from typing import Sequence

import click

from cellmap_utils_kit.add_raw import add_raw_main
from cellmap_utils_kit.check_yaml import check_data_yaml_main
from cellmap_utils_kit.copy_crops import copy_crops_main
from cellmap_utils_kit.correct_attrs import correct_label_attrs_main
from cellmap_utils_kit.filter_yaml import filter_yaml_main
from cellmap_utils_kit.h5_export import h5_export_main
from cellmap_utils_kit.multiscale import (
    smooth_multiscale_labels_main,
    smooth_multiscale_raw_main,
)


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity level (-v for INFO, -vv for DEBUG).",
)
def cli(verbose: int) -> None:
    """Data preparation CLI."""
    # Set the logging level based on the verbosity
    if verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


@click.command(name="copy-crops")
@click.argument("data-yaml")
@click.argument("destination")
def copy_crops_cli(data_yaml: str, destination: str) -> None:
    """Copy crops specified in `data_yaml` to a new `destination`.

    Args:
        data_yaml (str): Path to data configuration yaml
        destination (str): Folder to which crops should be copied

    """
    copy_crops_main(data_yaml, destination)


@click.command(name="check-data-yaml")
@click.argument("data-yaml")
@click.option(
    "--label-scalelevels",
    type=str,
    multiple=True,
    help="Scale levels for verifying labels.",
)
@click.option(
    "--raw-scalelevels",
    type=str,
    multiple=True,
    help="Scale levels for verifying raw data.",
)
def check_data_yaml_cli(
    data_yaml: str,
    label_scalelevels: tuple[str, ...] = (),
    raw_scalelevels: tuple[str, ...] = (),
) -> None:
    """Check that data specified in a data configuration yaml is valid.

    Args:
        data_yaml (str): Path to data configuration yaml
        label_scalelevels (tuple[str]): Scale levels for which to verify that they're
            openable for labels. Defaults to empty tuple.
        raw_scalelevels (tuple[str]): Scale levels for which to verify that they're
            openable for raw data. Defaults. to empty tuple.

    """
    check_data_yaml_main(
        data_yaml, label_scalelevels=label_scalelevels, raw_scalelevels=raw_scalelevels
    )


@click.command(name="multiscale-labels")
@click.argument("data-yaml")
@click.option(
    "--num-scales", type=int, help="desired number of scale levels.", default=4
)
def smooth_multiscale_labels_cli(data_yaml: str, num_scales: int = 4) -> None:
    """Generate multiscale pyramid for labels. Results in label smoothing.

    Args:
        data_yaml (str): Path to data configuration yaml
        num_scales (int, optional): Desired number of scale levels. Defaults to 4.

    """
    smooth_multiscale_labels_main(data_yaml, num_scales=num_scales)


@click.command(name="add-raw")
@click.argument("data-yaml")
def add_raw_cli(data_yaml: str) -> None:
    """Add cropped raw to crop.

    Args:
        data_yaml (str): Path to data configuration yaml

    """
    add_raw_main(data_yaml)


@click.command(name="multiscale-raw")
@click.argument("data-yaml")
@click.option(
    "--num-scales", type=int, help="desired number of scale levels.", default=4
)
def smooth_multiscale_raw_cli(data_yaml: str, num_scales: int = 4) -> None:
    """Generate multiscale pyramid for raw. Results in label smoothing.

    Args:
        data_yaml (str): Path to data configuration yaml
        num_scales (int, optional): Desired number of scale levels. Defaults to 4.

    """
    smooth_multiscale_raw_main(data_yaml, num_scales=num_scales)


@click.command(name="h5-export")
@click.argument("data-yaml")
@click.argument("destination")
def h5_export_cli(data_yaml: str, destination: str) -> None:
    """Export crops from zarr to h5.

    Args:
        data_yaml (str): Path to data configuration yaml
        destination (str): Folder to which crops should be copied

    """
    h5_export_main(data_yaml, destination)


@click.command(name="correct-attrs")
@click.argument("data-yaml")
def correct_attrs_cli(data_yaml: str) -> None:
    """Correct the complement counts in the label attributes for crops.

    Crops are specified in a data configuration yaml. Attributes will be edited in
    place.

    Args:
        data_yaml (str): Path to data configuration yaml.

    """
    correct_label_attrs_main(data_yaml)


@click.command(name="filter-yaml")
@click.argument("data-yaml")
@click.argument("data-yaml-filtered")
@click.option(
    "--scale",
    type=(float, float, float),
    default=None,
    help=(
        "Only keep labels that have a scale level matching this scale. If not given do"
        "not filter for scale."
    ),
)
@click.option(
    "--min-size",
    type=(int, int, int),
    default=None,
    help=(
        "Only keep labels that have at least this shape. If scale is given this "
        "refers to the label at that scale, otherwise it refers to `s0`. If not "
        "given do not filter for shape."
    ),
)
@click.option(
    "--min-frac-annotated",
    type=float,
    default=None,
    help=(
        "Minimum percent of voxels that should be annotated for a crop to be "
        "considered"
    ),
)
@click.option(
    "--labels",
    type=str,
    multiple=True,
    help="Labels to consider for checking the minimum percentage of annotations",
)
def filter_yaml_cli(
    data_yaml: str,
    data_yaml_filtered: str,
    scale: Sequence[float] | None = None,
    min_size: Sequence[int] | None = None,
    min_frac_annotated: float | None = None,
    labels: Sequence[str] = (),
) -> None:
    """Filter the list of crops in a data configuration yaml.

    Args:
        data_yaml (str): Path to data configuration yaml
        data_yaml_filtered (str): Path to where filtered data configuration yaml should
            be saved.
        scale (Sequence[float] | None, optional): Scale that needs to exist for labels
            in a crop to keep it. If None, scales are not considered for filtering.
            Other filters will be done at the given scale, and at s0 if None. Defaults
            to None.
        min_size (Sequence[int] | None, optional): Minimum size of a crop to keep it. If
            None, size is not considered for filtering. Defaults to None.
        min_frac_annotated (float | None, optional): Minimum fraction of a crop that
            needs to be annotated to keep it. This is checked for the labels specified
            in `labels`. If None, annotaton will not be considred for filtering.
            Defaults to None.
        labels (Sequence[str], optional): List of labels for which to check annotation
            fraction. Defaults to ().

    """
    filter_yaml_main(
        data_yaml,
        data_yaml_filtered,
        scale=scale,
        min_size=min_size,
        min_frac_annotated=min_frac_annotated,
        labels=labels,
    )


cli.add_command(copy_crops_cli)
cli.add_command(check_data_yaml_cli)
cli.add_command(smooth_multiscale_labels_cli)
cli.add_command(add_raw_cli)
cli.add_command(smooth_multiscale_raw_cli)
cli.add_command(h5_export_cli)
cli.add_command(correct_attrs_cli)
cli.add_command(filter_yaml_cli)


if __name__ == "__main__":
    cli()
