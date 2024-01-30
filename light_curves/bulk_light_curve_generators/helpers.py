import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

import yaml
from astropy.table import Table

sys.path.append("../code_src/")
# Lazy-load all other imports to avoid depending on modules that will not actually be used.

KWARG_DEFAULTS_YAML = "../bulk_light_curve_generators/helpers_kwargs_defaults.yml"


def run(build, *, kwargs_yaml=None, **kwargs_dict):
    my_kwargs_dict = _construct_kwargs_dict(kwargs_yaml=kwargs_yaml, kwargs_dict=kwargs_dict)

    if build == "sample":
        return _build_sample(**my_kwargs_dict)

    if build == "lightcurves":
        return _build_lightcurves(**my_kwargs_dict)

    return _build_other(keyword=build, kwargs_dict=my_kwargs_dict)


# ---- build functions ----


def _build_sample(
    *,
    literature_names,
    consolidate_nearby_objects,
    get_sample_kwargs,
    sample_filepath,
    overwrite_existing_sample,
    **extra_kwargs,
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

    _init_worker(job_name="_build_sample")
    sample_filepath.parent.mkdir(parents=True, exist_ok=True)

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if sample_filepath.is_file() and not overwrite_existing_sample:
        print(f"Using existing object sample at: {sample_filepath}", flush=True)
        return Table.read(sample_filepath, format="ascii.ecsv")

    # else continue fetching the sample
    print(f"Building object sample from literature: {literature_names}", flush=True)

    # create list of tuples: (get-sample function, kwargs dict)
    get_sample_functions = [
        (getattr(sample_selection, f"get_{name}_sample"), get_sample_kwargs.get(name, {}))
        for name in literature_names
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
    print(f"object sample saved to: {sample_filepath}", flush=True)
    return sample_table


def _build_lightcurves(
    *,
    mission,
    mission_kwargs,
    sample_filepath,
    parquet_dir,
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

    _init_worker(job_name=f"_build_lightcurves: {mission}")
    parquet_filepath = Path(f"{parquet_dir}/mission={mission}/part0.snappy.parquet")
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
    # elif keyword == "parquet_dir":
    #     other = my_kwargs_dict["base_dir"] + "/" + my_kwargs_dict["parquet_dataset_name"]
    # else:
    #     other = my_kwargs_dict.get(keyword)

    value = kwargs_dict.get(keyword)

    if print_scalar:
        print(value)
    if print_list:
        print(" ".join(value))

    return value


# ---- utils ----


def _load_yaml(fyaml):
    with open(fyaml, "r") as fin:
        kwargs = yaml.safe_load(fin)
    return kwargs


def _construct_kwargs_dict(*, kwargs_yaml=None, kwargs_dict=dict()):
    """Construct a complete kwargs dict by combining `kwargs_dict`, `kwargs_yaml`, and `KWARG_DEFAULTS_YAML`
    (listed in order of precedence).
    """
    # load defaults
    my_kwargs_dict = _load_yaml(KWARG_DEFAULTS_YAML)
    # load and add kwargs from yaml file
    my_kwargs_dict.update(_load_yaml(kwargs_yaml) if kwargs_yaml else {})
    # add kwargs from dict
    my_kwargs_dict.update(kwargs_dict)

    # expand a literature_names shortcut
    if my_kwargs_dict['literature_names'] == "core":
        my_kwargs_dict['literature_names'] = my_kwargs_dict["literature_names_core"]

    # construct and add base_dir, sample_filepath, and parquet_dir
    base_dir = my_kwargs_dict["base_dir_stub"] + "-" + my_kwargs_dict["run_id"]
    my_kwargs_dict["base_dir"] = base_dir
    my_kwargs_dict['sample_filepath'] = Path(base_dir + "/" + my_kwargs_dict['sample_filename'])
    my_kwargs_dict["parquet_dir"] = Path(base_dir + "/" + my_kwargs_dict["parquet_dataset_name"])

    # handle mission_kwargs
    my_kwargs_dict["mission_kwargs"] = _construct_mission_kwargs(my_kwargs_dict)

    return {key: my_kwargs_dict[key] for key in sorted(my_kwargs_dict)}


def _construct_mission_kwargs(kwargs_dict):
    """Construct mission_kwargs from kwargs_dict."""
    mission = kwargs_dict.get("mission", "").lower()
    # get default mission_kwargs
    default_mission_kwargs = kwargs_dict["mission_kwargs_all"].get(mission, {})
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

    return mission_kwargs


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
    run(args.build, kwargs_yaml=args.kwargs_yaml, kwargs_dict=args.extra_kwargs)


def _parse_args(args_list):
    # define the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        type=str,
        default="sample",
        help="Either 'sample', 'lightcurves', or a key in _helpers_kwargs_options.yml",
    )
    parser.add_argument(
        "--kwargs_yaml",
        type=str,
        default="../bulk_light_curve_generators/helpers_kwargs_defaults.yml",
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
