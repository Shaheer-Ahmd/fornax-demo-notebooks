import argparse
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from astropy.table import Table

# Lazy-load all other imports to avoid depending on modules that will not actually be used.

BULK_RUN_DIR = Path(__file__).parent
sys.path.append(str(BULK_RUN_DIR.parent))  # put code_src dir on the path

ARCHIVE_NAMES = {
    "all": ["Gaia", "HCV", "HEASARC", "IceCube", "PanSTARRS", "TESS_Kepler", "WISE", "ZTF"],
    "core": ["Gaia", "HEASARC", "IceCube", "WISE", "ZTF"],
}
DEFAULTS = {
    "run_id": "my-run",
    "get_samples": ["Yang"],
    "consolidate_nearby_objects": True,
    "overwrite_existing_sample": True,
    "archives": ARCHIVE_NAMES["all"],
    "overwrite_existing_lightcurves": True,
    "use_yaml": False,
    "yaml_filename": "kwargs.yml",
    "sample_filename": "object_sample.ecsv",
    "parquet_dataset_name": "lightcurves.parquet",
}


def run(*, build, **kwargs_dict):
    my_kwargs_dict = _construct_kwargs_dict(**kwargs_dict)

    if build == "sample":
        return _build_sample(**my_kwargs_dict)

    if build == "lightcurves":
        return _build_lightcurves(**my_kwargs_dict)

    return _build_other(keyword=build, **my_kwargs_dict)


# ---- build functions ---- #


def _build_sample(*, get_samples, consolidate_nearby_objects, sample_file, overwrite_existing_sample, **_):
    """Build an AGN sample using coordinates from different papers.

    Parameters
    ----------
    get_samples : dict
        Dict keys should be sample names from sample_selection.py -- the function
        `get_{key}_sample` will be called for every key.
        Dict values should be dicts of keyword arguments for that function.
    sample_filename : str
        Name of the file to write the sample objects to.
    kwargs : dict[str: dict[str: any]]
        Dict key should be one of get_samples, value should be a dict of keyword arguments
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
    _init_worker(job_name="build=sample")

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if sample_file.is_file() and not overwrite_existing_sample:
        print(f"Using existing object sample at: {sample_file}", flush=True)
        return Table.read(sample_file, format="ascii.ecsv")

    # else continue fetching the sample
    print(f"Building object sample from: {list(get_samples.keys())}", flush=True)

    import sample_selection

    # list of tuples. tuple contains: (get-sample function, kwargs dict for that function)
    get_sample_functions = [
        (getattr(sample_selection, f"get_{name}_sample"), kwargs) for name, kwargs in get_samples.items()
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
    sample_table.write(sample_file, format="ascii.ecsv", overwrite=True)
    print(f"Object sample saved to: {sample_file}")
    print(_now(), flush=True)
    return sample_table


def _build_lightcurves(
    *, archive, archive_kwargs, sample_file, parquet_dir, overwrite_existing_lightcurves, **_
):
    """Fetch data from the archive and build light curves for objects in sample_filename.

    Parameters
    ----------
    archive : str
        Name of the archive to query for light curves. Case-insensitive.
        (This will call the function `{archive}_get_lightcurve`.)
    base_dir : str
        Base directory for the sample file and parquet dataset.
    sample_filename : str
        Name of the file containing the sample objects.
    parquet_dataset_name : str
        Name of the parquet dataset to write light curves to.
    overwrite_existing_lightcurves : bool
        Whether to overwrite an existing data file (True) or skip building light curves if a file
        exists (False). Has no effect if there is no preexisting data file for this archive.

    Returns
    -------
    lightcurve_df : MultiIndexDFObject
        Light curves. This function also writes the light curves to a Parquet file.
    """
    _init_worker(job_name=f"build=lightcurves, archive={archive}")
    parquet_filepath = parquet_dir / f"archive={archive}" / "part0.snappy.parquet"

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if parquet_filepath.is_file() and not overwrite_existing_lightcurves:
        from data_structures import MultiIndexDFObject

        print(f"Using existing light curve data at: {parquet_filepath}", flush=True)
        return MultiIndexDFObject(data=pd.read_parquet(parquet_filepath))

    # Load the sample.
    sample_table = Table.read(sample_file, format="ascii.ecsv")

    # Import only the archive module that will actually be used to avoid unnecessary dependencies.
    archive_functions = importlib.import_module(f"{archive}_functions")
    get_lightcurves_fnc = getattr(archive_functions, f"{archive}_get_lightcurves")
    # archive_kwargs = archive_kwargs.get(f"{archive}_get_lightcurves", {})

    # Query the archive and load light curves.
    lightcurve_df = get_lightcurves_fnc(sample_table, **archive_kwargs)

    # Save and return the light curve data.
    parquet_filepath.parent.mkdir(parents=True, exist_ok=True)
    lightcurve_df.data.to_parquet(parquet_filepath)
    print(f"Light curves saved to:\n\tparquet_dir={parquet_dir}\n\tfile={parquet_filepath.relative_to(parquet_dir)}")
    print(_now(), flush=True)
    return lightcurve_df


def _build_other(keyword, **kwargs_dict):
    if keyword == "kwargs":
        return kwargs_dict

    # if this was called from the command line, we need to print the value so it can be captured by the script
    # this is indicated by a "+" flag appended to the keyword
    print_scalar, print_list = keyword.endswith("+"), keyword.endswith("+l")
    my_keyword = keyword.removesuffix("+l").removesuffix("+")

    # get the keyword value
    if my_keyword in ["archive_names_all", "archive_names_core"]:
        value = ARCHIVE_NAMES[my_keyword.split("_")[-1]]
    else:
        value = kwargs_dict[my_keyword]

    if print_scalar:
        print(value)
    if print_list:
        print(" ".join(value))

    return value


# ---- construct kwargs ---- #


def _construct_kwargs_dict(**kwargs_dict):
    """Construct a complete kwargs dict by combining defaults, yaml (if requested), and `kwargs_dict`
    (listed in order of increasing precedence).
    """
    run_id = kwargs_dict.get("run_id", DEFAULTS["run_id"])
    base_dir = BULK_RUN_DIR.parent.parent / f"output/lightcurves-{run_id}"

    # load kwargs from yaml if requested
    yaml_file = base_dir / kwargs_dict.get("yaml_filename", DEFAULTS["yaml_filename"])
    my_kwargs = _load_yaml(yaml_file) if kwargs_dict.get("use_yaml", DEFAULTS["use_yaml"]) else {}

    # update with kwargs_dict. both may contain dict values for the following keys, which need deep updates.
    for key in ["get_samples", "archives"]:
        my_kwargs[key] = _deep_update_kwargs_group(key, my_kwargs.pop(key, []), kwargs_dict.pop(key, []))
    my_kwargs.update(kwargs_dict)  # update with any additional keys in kwargs_dict

    # add any defaults that are still missing
    for key in set(DEFAULTS) - set(my_kwargs):
        my_kwargs[key] = DEFAULTS[key]

    # if a single archive is requested, lower-case it and pull out the right set of kwargs for _build_lightcurves
    if my_kwargs.get("archive"):
        my_kwargs["archive"] = my_kwargs["archive"].lower()
        my_kwargs["archive_kwargs"] = my_kwargs["archives"].get(my_kwargs["archive"], {})

    # set path kwargs and make the base dir if needed
    my_kwargs["run_id"] = run_id
    my_kwargs["base_dir"] = base_dir
    my_kwargs["logs_dir"] = base_dir / "logs"
    my_kwargs["sample_file"] = base_dir / my_kwargs["sample_filename"]
    my_kwargs["parquet_dir"] = base_dir / my_kwargs["parquet_dataset_name"]
    my_kwargs["yaml_file"] = yaml_file
    base_dir.mkdir(parents=True, exist_ok=True)

    # sort by key and return
    return {key: my_kwargs[key] for key in sorted(my_kwargs)}


def _deep_update_kwargs_group(key, group_a, group_b):
    # if both groups are empty, just return defaults
    if len(group_a) == 0 and len(group_b) == 0:
        return _kwargs_list_to_dict(DEFAULTS[key])

    # these groups may be either lists or dicts.
    # turn them both into dicts with key = name, value = dict of kwargs for name (empty dict if none supplied)
    my_group_a, my_group_b = _kwargs_list_to_dict(group_a), _kwargs_list_to_dict(group_b)

    # update a with b. first, descend one level to update individual name/kwarg pairs.
    for name in my_group_a:
        my_group_a[name].update(my_group_b.pop(name, {}))
    # add any keys from b that were not in a
    my_group_a.update(my_group_b)

    return my_group_a


def _kwargs_list_to_dict(list_or_dict):
    if isinstance(list_or_dict, list):
        return {name.lower(): {} for name in list_or_dict}
    return {name.lower(): kwargs for name, kwargs in list_or_dict.items()}


# ---- other utils ---- #


def _init_worker(job_name="worker"):
    """Run generic start-up tasks for a job."""
    # print the Process ID for the current worker so it can be killed if needed
    print(f"{_now()} | [pid={os.getpid()}] Starting {job_name}", flush=True)


def _now():
    # parse with:
    # strtime = '2024/01/31 12:40:29 UTC'
    # datetime.strptime(strtime, "%Y/%m/%d %H:%M:%S %Z")
    date_format = "%Y/%m/%d %H:%M:%S %Z"
    return datetime.now(timezone.utc).strftime(date_format)


def load_top_log(run_id):
    logs_dir = run(build="logs_dir", run_id=run_id)

    # map PIDs to names by scraping log files
    pid_map = {}
    for logfile in logs_dir.iterdir():
        if not logfile.is_file() or logfile.name.endswith(".sh.log"):
            continue
        pid_map.update(_regex_log_for_pid_names(logfile))

    summary_df, pid_df = _load_toptxt(logs_dir / "top.txt", pid_map)
    
    return summary_df, pid_df


def _load_toptxt(ftop, pid_map=dict()):
    # load top.txt to a dataframe
    line_batch, df_list = [], []
    with open(ftop, "r") as fin:
        for line in fin:
            if line.startswith("----") and len(line_batch) > 0:
                df_list.append(_regex_top_line_batch(line_batch, pid_map))
                line_batch = []
            line_batch.append(line.removeprefix("----").rstrip("\n").strip())
        df_list.append(_regex_top_line_batch(line_batch, pid_map))

    # summary_df = pd.concat([df_tuple[0] for df_tuple in df_list], ignore_index=True)
    # pid_df = pd.concat([df_tuple[1] for df_tuple in df_list], ignore_index=True)
    # return summary_df, pid_df
    summary_dfs, pid_dfs = zip(*df_list)
    return pd.concat(summary_dfs, ignore_index=True), pd.concat(pid_dfs, ignore_index=True)


def _regex_log_for_pid_names(logfile):
    pid_map = {}
    pid_re = r"\[pid=([0-9]+)\]"
    archive_re = r"archive=([a-zA-Z0-9_-]+)"
    build_re = r"build=([a-zA-Z0-9_-]+)"

    with open(logfile) as fin:
        for line in fin:
            pid_match = re.search(pid_re, line)
            if not pid_match:
                continue
            pid = pid_match.group(1)

            name_match = re.search(archive_re, line) or re.search(build_re, line)
            try:
                pid_map[pid] = name_match.group(1)
            except AttributeError:
                if "worker" not in line:
                    raise NotImplementedError(f"unable to identify a name for [pid={pid}]")
                pid_map[pid] = logfile.stem + "-worker"

    return pid_map


def _regex_top_line_batch(line_batch, pid_map=dict()):
    # line 0
    batch_tag = line_batch[0]
    
    # line 1
    date_format = "%Y/%m/%d %H:%M:%S %Z"
    batch_time = datetime.strptime(line_batch[1], date_format)

    # line 2
    load_match = re.search(r"(load average: )(.+)$", line_batch[2])
    load_avg = tuple(float(num) for num in load_match.group(2).split(", "))

    # lines 5-6
    ram_units = re.search(r"^(.+)( Mem )", line_batch[5]).group(1)
    mem_free = float(re.search(r"([0-9]+\.[0-9]+)( free)", line_batch[5]).group(1))
    mem_used = float(re.search(r"([0-9]+\.[0-9]+)( used)", line_batch[5]).group(1))
    mem_avail = float(re.search(r"([0-9]+\.[0-9]+)( avail Mem)$", line_batch[6]).group(1))

    # summary df
    data = [(batch_time, batch_tag, *load_avg, mem_free, mem_used, mem_avail)]
    load_names = [f"load_avg_{t}" for t in ["1m","5m","15m"]]
    mem_names = [f"{s}_{ram_units}" for s in ["free", "used", "avail"]]
    summary_df = pd.DataFrame(data=data, columns=["time", "tag", *load_names, *mem_names])

    # pid df, lines 8+
    pid_df = pd.DataFrame(data=[line.split() for line in line_batch[9:]], columns=line_batch[8].split())
    pid_df["time"] = batch_time
    pid_df["job"] = pid_df.PID.map(pid_map)
    pid_df = pid_df[["time", "job", *line_batch[8].split()]]

    return summary_df, pid_df


def _load_yaml(yaml_file):
    with open(yaml_file, "r") as fin:
        yaml_dict = yaml.safe_load(fin)
    return yaml_dict


def write_kwargs_to_yaml(**kwargs_dict):
    yaml_path = run(build="yaml_file", **kwargs_dict)
    with open(yaml_path, "w") as fout:
        yaml.safe_dump(kwargs_dict, fout)
    print(f"kwargs written to {yaml_path}")


# ---- helpers for __name__ == "__main__" ---- #


def __argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        type=str,
        default="sample",
        help="Either 'sample', 'lightcurves', or a kwargs key.",
    )
    parser.add_argument(
        "--kwargs_dict",
        type=str,
        default=list(),
        nargs="*",
        help="Keyword arguments for the run. Input as a list of strings 'key=value'.",
    )
    parser.add_argument(
        "--kwargs_json",
        type=json.loads,
        default=r"{}",
        help="Kwargs as a json string, to be added to kwargs_dict.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Archive name to query for light curves, to be added to kwargs_dict.",
    )
    return parser


def __parse_args(args_list):
    args = __argparser().parse_args(args_list)

    # start with kwargs_json, then update
    my_kwargs_dict = args.kwargs_json

    # parse args.kwargs_dict 'key=value' pairs into a proper dict and convert true/false to bool
    bool_map = {"true": True, "false": False}
    args_kwargs_dict = {
        key: bool_map.get(val.lower(), val)
        for (key, val) in dict(kwarg.split("=") for kwarg in args.kwargs_dict).items()
        # for (key, val) in {kwarg.split("=")[0]: kwarg.split("=")[1] for kwarg in args.kwargs_dict}.items()
    }

    # update kwargs_dict with args_kwargs_dict
    # both may contain dict values for the following keys, which need deep updates.
    for key in ["get_samples", "archives"]:
        my_kwargs_dict[key] = _deep_update_kwargs_group(
            key, my_kwargs_dict.pop(key, []), args_kwargs_dict.pop(key, [])
        )
    my_kwargs_dict.update(args_kwargs_dict)  # update with any additional keys in args_kwargs_dict
    # kwargs_dict.update(args.kwargs_json)
    # pop out the kwargs_json, parse into a dict, and add them back in
    # kwargs_json = args.kwargs_dict.pop('kwargs_json', r'{}')

    # add the archive, if provided
    if args.archive:
        my_kwargs_dict["archive"] = args.archive

    return args.build, my_kwargs_dict


if __name__ == "__main__":
    build, kwargs_dict = __parse_args(sys.argv[1:])
    run(build=build, **kwargs_dict)
