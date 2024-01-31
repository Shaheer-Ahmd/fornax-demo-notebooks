import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

import yaml
from astropy.table import Table

BULK_RUN_DIR = Path(__file__).parent
# print("mypy", BULK_RUN_DIR)
sys.path.append(str(BULK_RUN_DIR.parent))

# Lazy-load all other imports to avoid depending on modules that will not actually be used.


# load default kwargs from yaml files
def _load_yaml(yaml_file):
    with open(yaml_file, "r") as fin:
        yaml_dict = yaml.safe_load(fin)
    return yaml_dict


KWARG_DEFAULTS = _load_yaml(BULK_RUN_DIR / "helper_kwargs_defaults.yml")
KWARG_DEFAULTS_BAG = KWARG_DEFAULTS.pop("bag")


def run(*, build, kwargs_yaml=None, **kwargs_dict):
    my_kwargs_dict = _construct_kwargs_dict(kwargs_yaml=kwargs_yaml, kwargs_dict=kwargs_dict)

    if build == "sample":
        return _build_sample(**my_kwargs_dict)

    if build == "lightcurves":
        return _build_lightcurves(**my_kwargs_dict)

    return _build_other(keyword=build, **my_kwargs_dict)


# ---- build functions ----


def _build_sample(
    *,
    literature_names,
    consolidate_nearby_objects,
    sample_filepath,
    overwrite_existing_sample,
    **get_sample_kwargs,
):
    """Build an AGN sample using coordinates from different papers.

    Parameters
    ----------
    sample_filename : str
        Name of the file to write the sample objects to.
    literature_names : list[str]
        Names of papers to get sample objects from. Case-sensitive.
        This will call the function `get_{name}_sample` for every name in literature_names.)
    kwargs : dict[str: dict[str: any]]
        Dict key should be one of literature_names, value should be a dict of keyword arguments
        for the get-sample function.
    base_dir : str
        Base directory for the sample file.
    sample_filename : str
        Name of file to write the sample to.
    overwrite_existing_sample : bool
        Whether to overwrite a preexisting sample file (True) or skip fetching a new sample and return
        the sample on file (False). Has no effect if there is no preexisting file at
        `{base_dir}/{sample_filename}`.

    Returns
    -------
    sample_table : `~astropy.table.Table`
        Coordinates and labels for objects in the sample.
        This function also writes the sample to an ascii.ecsv file.
    """

    import sample_selection

    _init_worker(job_name="build=sample")

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if sample_filepath.is_file() and not overwrite_existing_sample:
        print(f"Using existing object sample at: {sample_filepath}", flush=True)
        return Table.read(sample_filepath, format="ascii.ecsv")

    # else continue fetching the sample
    print(f"Building object sample from literature: {literature_names}", flush=True)

    # create a list of tuples for the get_sample functions and their kwargs
    get_sample_functions = [
        # tuple: (get-sample function, kwargs dict)
        (getattr(sample_selection, name), get_sample_kwargs.get(f"{name}_kwargs", {}))
        for name in [f"get_{name}_sample" for name in literature_names]
    ]

    # iterate over the functions and get the samples
    coords, labels = [], []
    for get_sample_fnc, kwargs in get_sample_functions:
        get_sample_fnc(coords, labels, **kwargs)

    # create an astropy Table of objects
    sample_table = sample_selection.clean_sample(
        coords, labels, consolidate_nearby_objects=consolidate_nearby_objects
    )

    # save and return the Table
    sample_table.write(sample_filepath, format="ascii.ecsv", overwrite=True)
    print(f"Object sample saved to: {sample_filepath}", flush=True)
    return sample_table


def _build_lightcurves(
    *,
    mission,
    mission_kwargs,
    sample_filepath,
    parquet_dirpath,
    overwrite_existing_data,
    **extra_kwargs,
):
    """Fetch data from the mission's archive and build light curves for objects in sample_filename.

    Parameters
    ----------
    mission : str
        Name of the mission to query for light curves. Case-insensitive.
        (This will call the function `{mission}_get_lightcurve`.)
    base_dir : str
        Base directory for the sample file and parquet dataset.
    sample_filename : str
        Name of the file containing the sample objects.
    parquet_dataset_name : str
        Name of the parquet dataset to write light curves to.
    overwrite_existing_data : bool
        Whether to overwrite an existing data file (True) or skip building light curves if a file
        exists (False). Has no effect if there is no preexisting data file for this mission.

    Returns
    -------
    lightcurve_df : MultiIndexDFObject
        Light curves. This function also writes the light curves to a Parquet file.
    """

    _init_worker(job_name=f"build=lightcurves, mission={mission}")
    parquet_filepath = parquet_dirpath / f"mission={mission}/part0.snappy.parquet"
    parquet_filepath.parent.mkdir(parents=True, exist_ok=True)

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if parquet_filepath.is_file() and not overwrite_existing_data:
        import pandas as pd
        from data_structures import MultiIndexDFObject

        print(f"Using existing light curve data at: {parquet_filepath}", flush=True)
        return MultiIndexDFObject(data=pd.read_parquet(parquet_filepath))

    # else load the sample and fetch the light curves
    sample_table = Table.read(sample_filepath, format="ascii.ecsv")

    # Query the mission's archive and load light curves.
    # [TODO] uniformize module and function names so that we can do this with getattr (like _build_sample)
    # instead of checking for every mission individually.
    if mission.lower() == "gaia":
        from gaia_functions import Gaia_get_lightcurve

        lightcurve_df = Gaia_get_lightcurve(sample_table, **mission_kwargs)

    elif mission.lower() == "heasarc":
        from heasarc_functions import HEASARC_get_lightcurves

        lightcurve_df = HEASARC_get_lightcurves(sample_table, **mission_kwargs)

    elif mission.lower() == "hcv":
        from HCV_functions import HCV_get_lightcurves

        lightcurve_df = HCV_get_lightcurves(sample_table, **mission_kwargs)

    elif mission.lower() == "icecube":
        from icecube_functions import Icecube_get_lightcurve

        lightcurve_df = Icecube_get_lightcurve(sample_table, **mission_kwargs)

    elif mission.lower() == "panstarrs":
        from panstarrs import Panstarrs_get_lightcurves

        lightcurve_df = Panstarrs_get_lightcurves(sample_table, **mission_kwargs)

    elif mission.lower() == "tess_kepler":
        from TESS_Kepler_functions import TESS_Kepler_get_lightcurves

        lightcurve_df = TESS_Kepler_get_lightcurves(sample_table, **mission_kwargs)

    elif mission.lower() == "wise":
        from WISE_functions import WISE_get_lightcurves

        lightcurve_df = WISE_get_lightcurves(sample_table, **mission_kwargs)

    elif mission.lower() == "ztf":
        from ztf_functions import ZTF_get_lightcurve

        lightcurve_df = ZTF_get_lightcurve(sample_table, **mission_kwargs)

    else:
        raise ValueError(f"Unknown mission '{mission}'")

    # save and return the light curve data
    lightcurve_df.data.to_parquet(parquet_filepath)
    print(f"Light curves saved to: {parquet_filepath}", flush=True)
    return lightcurve_df


def _build_other(keyword, **kwargs_dict):
    # if this was called from the command line, we need to print the value so it can be captured by the script
    # this is indicated by a "+" flag appended to the keyword
    print_scalar, print_list = keyword.endswith("+"), keyword.endswith("+l")
    keyword = keyword.strip("+l").strip("+")

    # if keyword == "sample_filepath":
    #     other = my_kwargs_dict["base_dir"] + "/" + my_kwargs_dict['sample_filename']
    # elif keyword == "parquet_dirpath":
    #     other = my_kwargs_dict["base_dir"] + "/" + my_kwargs_dict["parquet_dataset_name"]
    # else:
    #     other = my_kwargs_dict.get(keyword)

    value = kwargs_dict.get(keyword, KWARG_DEFAULTS_BAG.get(keyword))

    if print_scalar:
        print(value)
    if print_list:
        print(" ".join(value))

    return value


# ---- utils ----


def _construct_kwargs_dict(*, kwargs_yaml=None, kwargs_dict=dict()):
    """Construct a complete kwargs dict by combining `kwargs_dict`, `kwargs_yaml`, and `KWARG_DEFAULTS`
    (listed in order of precedence).
    """
    # make a copy of the defaults
    my_kwargs_dict = dict(**KWARG_DEFAULTS)
    # update with kwargs from yaml file
    my_kwargs_dict.update(_load_yaml(kwargs_yaml) if kwargs_yaml else {})
    # update with kwargs from dict
    my_kwargs_dict.update(kwargs_dict)

    # handle sample and mission kwargs
    my_kwargs_dict.update(_construct_sample_kwargs(my_kwargs_dict))
    my_kwargs_dict.update(_construct_mission_kwargs(my_kwargs_dict))

    # construct and add paths
    base_dir = BULK_RUN_DIR.parent.parent / f"output/lightcurves-{my_kwargs_dict['run_id']}"
    base_dir.mkdir(parents=True, exist_ok=True)
    my_kwargs_dict["base_dir"] = base_dir
    my_kwargs_dict["sample_filepath"] = base_dir / my_kwargs_dict["sample_filename"]
    my_kwargs_dict["parquet_dirpath"] = base_dir / my_kwargs_dict["parquet_dataset_name"]

    # sort the kwargs by key and return
    return {key: my_kwargs_dict[key] for key in sorted(my_kwargs_dict)}


def _construct_sample_kwargs(kwargs_dict):
    """Construct sample kwargs from kwargs_dict plus defaults."""
    # make a copy of the defaults for get_*_sample functions
    my_sample_kwargs = {**KWARG_DEFAULTS_BAG["get_sample_kwargs_all"]}
    # update with passed-in dict
    my_sample_kwargs.update(kwargs_dict)

    # expand a literature_names shortcut
    if my_sample_kwargs["literature_names"] == "all":
        my_sample_kwargs["literature_names"] = [*KWARG_DEFAULTS_BAG["literature_names_all"]]

    return my_sample_kwargs


def _construct_mission_kwargs(kwargs_dict):
    """Construct mission_kwargs from kwargs_dict plus defaults."""
    mission = kwargs_dict.get("mission", "").lower()
    # make a copy of default mission_kwargs
    default_mission_kwargs = {**KWARG_DEFAULTS_BAG["mission_kwargs_all"].get(mission, {})}
    # update with passed-in values
    default_mission_kwargs.update(kwargs_dict.get("mission_kwargs", {}))

    # convert radius to a float
    mission_kwargs = {
        key: (float(Fraction(val)) if key.endswith("radius") else val)
        for key, val in default_mission_kwargs.items()
    }

    # convert radius to an astropy Quantity if needed
    if mission in ["wise", "ztf"]:
        import astropy.units as u

        radius, unit = tuple(["radius", u.arcsec] if mission == "wise" else ["match_radius", u.deg])
        mission_kwargs[radius] = mission_kwargs[radius] * unit

    return {"mission_kwargs": mission_kwargs}


def _init_worker(job_name="worker"):
    """Run generic start-up tasks for a job."""
    # print the Process ID for the current worker so it can be killed if needed
    print(f"[pid={os.getpid()}] Starting {job_name}", flush=True)


# ---- helpers for __name__ == "__main__" ----


def _run_main(args_list):
    """Run the function to build either the object sample or the light curves.

    Parameters
    ----------
    args_list : list
        Arguments submitted from the command line.
    """
    args = _parse_args(args_list)
    run(build=args.build, kwargs_yaml=args.kwargs_yaml, kwargs_dict=args.extra_kwargs)


def _parse_args(args_list):
    # define the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        type=str,
        default="sample",
        help="Either 'sample', 'lightcurves', or a kwargs key.",
    )
    parser.add_argument(
        "--kwargs_yaml",
        type=str,
        default=None,
        help="Path to a yaml file containing the function keyword arguments to be used.",
    )
    parser.add_argument(
        "--extra_kwargs",
        type=str,
        default=list(),
        nargs="*",
        help="Kwargs to be added to kwargs_yaml. If the same key is provided both places, this takes precedence.",
    )
    parser.add_argument(
        # this is separate for convenience and will be added to extra_kwargs if provided
        "--mission",
        type=str,
        default=None,
        help="Mission name to query for light curves.",
    )

    # parse and return the script arguments
    args = parser.parse_args(args_list)
    args.extra_kwargs = _parse_extra_kwargs(args)
    # if kwargs_yaml was sent in extra_kwargs pop it out, but don't overwrite existing
    args.kwargs_yaml = args.kwargs_yaml or args.extra_kwargs.pop("kwargs_yaml", None)
    return args


def _parse_extra_kwargs(args):
    # parse extra kwargs into a dict
    extra_kwargs_tmp = {kwarg.split("=")[0]: kwarg.split("=")[1] for kwarg in args.extra_kwargs}
    # convert true/false to bool
    extra_kwargs = {
        key: (bool(val) if val.lower() in ["true", "false"] else val) for (key, val) in extra_kwargs_tmp.items()
    }
    # add the mission, if provided
    if args.mission:
        extra_kwargs["mission"] = args.mission
    return extra_kwargs


if __name__ == "__main__":
    _run_main(sys.argv[1:])
