---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: science_demo
  language: python
  name: conda-env-science_demo-py
---

# Make Multi-Wavelength Light Curves Using Archival Data
***

## Learning Goals    
By the end of this tutorial, you will be able to:  
  &bull; Automatically load a catalog of sources  
  &bull; Automatically & efficiently search NASA and non-NASA resources for light curves at scale  
  &bull; Store & manipulate light curves in a Pandas MultiIndex dataframe  
  &bull; Plot all light curves on the same plot
 
 
## Introduction:  
 &bull; A user has a sample of interesting targets for which they would like to see a plot of available archival light curves.  We start with a small set of changing look AGN from Yang et al., 2018, which are automatically downloaded. Changing look AGN are cases where the broad emission lines appear or disappear (and not just that the flux is variable). 
 
 &bull; We model light curve plots after van Velzen et al. 2021.  We search through a curated list of time-domain NASA holdings as well as non-NASA sources.  HEASARC catalogs used are Fermi and Beppo-Sax, IRSA catalogs used are ZTF and WISE, and MAST catalogs used are Pan-STARRS, TESS, Kepler, and K2.  Non-NASA sources are Gaia and IceCube. This list is generalized enough to include many types of targets to make this notebook interesting for many types of science.  All of these time-domain archives are searched in an automated and efficient fashion using astroquery, pyvo, pyarrow or APIs.
 
 &bull; Light curve data storage is a tricky problem.  Currently we are using a MultiIndex Pandas dataframe, as the best existing choice for right now.  One downside is that we need to manually track the units of flux and time instead of relying on an astropy storage scheme which would be able to do some of the units worrying for us (even astropy can't do all magnitude to flux conversions).  Astropy does not currently have a good option for multi-band light curve storage.
 
 &bull; ML work using these time-series light curves is in two neighboring notebooks: ML_AGNzoo and lc_classifier.
 
## Input:
 &bull; choose from a list of known changing look AGN from the literature  
  OR -    
 &bull; input your own sample

## Output:
 &bull; an archival optical + IR + neutrino light curve  
 
## Imports:
 &bull; `acstools` to work with HST magnitude to flux conversion  
 &bull; `astropy` to work with coordinates/units and data structures  
 &bull; `astroquery` to interface with archives APIs  
 &bull; `hpgeom` to locate coordinates in HEALPix space  
 &bull; `lightkurve` to search TESS, Kepler, and K2 archives  
 &bull; `matplotlib` for plotting  
 &bull; `multiprocessing` to use the power of multiple CPUs to get work done faster  
 &bull; `numpy` for numerical processing  
 &bull; `pandas` for their data structure DataFrame and all the accompanying functions  
 &bull; `pyarrow` to work with Parquet files for WISE and ZTF  
 &bull; `pyvo` for accessing Virtual Observatory(VO) standard data  
 &bull; `requests` to get information from URLs  
 &bull; `scipy` to do statistics  
 &bull; `tqdm` to track progress on long running jobs  
 &bull; `urllib` to handle archive searches with website interface

## Authors:
Jessica Krick, Shoubaneh Hemmati, Andreas Faisst, Troy Raen, Brigitta Sipőcz, Dave Shupe

## Acknowledgements:
Suvi Gezari, Antara Basu-zych, Stephanie LaMassa  
MAST, HEASARC, & IRSA Fornax teams

```{code-cell} ipython3
# Ensure all dependencies are installed
!pip install -r requirements.txt
```

```{code-cell} ipython3
import multiprocessing as mp
import sys
import time

import astropy.units as u
import pandas as pd
from astropy.table import Table

# local code imports
sys.path.append('code_src/')
from data_structures import MultiIndexDFObject
from gaia_functions import gaia_get_lightcurves
from hcv_functions import hcv_get_lightcurves
from heasarc_functions import heasarc_get_lightcurves
from icecube_functions import icecube_get_lightcurves
from panstarrs_functions import panstarrs_get_lightcurves
from plot_functions import create_figures
from sample_selection import (clean_sample, get_green_sample, get_hon_sample, get_lamassa_sample, get_lopeznavas_sample,
    get_lyu_sample, get_macleod16_sample, get_macleod19_sample, get_ruan_sample, get_sdss_sample, get_sheng_sample, get_yang_sample)
from tess_kepler_functions import tess_kepler_get_lightcurves
# Note: WISE and ZTF data are temporarily located in a non-public AWS S3 bucket. It is automatically
# available from the Fornax SMCE, but will require user credentials for access outside the SMCE.
from wise_functions import wise_get_lightcurves
from ztf_functions import ztf_get_lightcurves
```

## 1. Define the sample
We define here a "gold" sample of spectroscopically confirmed changing look AGN and quasars. This sample includes both objects which change from type 1 to type 2 and also the opposite.  Future studies may want to treat these as separate objects or separate QSOs from AGN.  Bibcodes for the samples used are listed next to their functions for reference.  
 
Significant work went into the functions which grab the samples from the papers.  They use Astroquery, NED, SIMBAD, Vizier, and in a few cases grab the tables from the html versions of the paper.  There are trickeries involved in accessing coordinates from tables in the literature. Not every literature table is stored in its entirety in all of these resources, so be sure to check that your chosen method is actually getting the information that you see in the paper table.  Warning: You will get false results if using NED or SIMBAD on a table that has more rows than are printed in the journal.

```{code-cell} ipython3
# Build up the sample
# Initially set up lists to hold the coordinates and their reference paper name as a label
coords =[]
labels = []

# Choose your own adventure:

#get_lamassa_sample(coords, labels)  #2015ApJ...800..144L
#get_macleod16_sample(coords, labels) #2016MNRAS.457..389M
#get_ruan_sample(coords, labels) #2016ApJ...826..188R
#get_macleod19_sample(coords, labels)  #2019ApJ...874....8M
#get_sheng_sample(coords, labels)  #2020ApJ...889...46S
#get_green_sample(coords, labels)  #2022ApJ...933..180G
#get_lyu_sample(coords, labels)  #z32022ApJ...927..227L
#get_lopeznavas_sample(coords, labels)  #2022MNRAS.513L..57L
#get_hon_sample(coords, labels)  #2022MNRAS.511...54H
get_yang_sample(coords, labels)   #2018ApJ...862..109Y

# Get some "normal" QSOs 
# there are ~500K of these, so choose the number based on
# a balance between speed of running the light curves and whatever 
# the ML algorithms would like to have

# num_normal_QSO = 5000
# zmin, zmax = 0, 10
# randomize_z = False
#get_sdss_sample(coords, labels, num=num_normal_QSO, zmin=zmin, zmax=zmax, randomize_z=randomize_z)

# Remove duplicates, attach an objectid to the coords,
# convert to astropy table to keep all relevant info together
sample_table = clean_sample(coords, labels)
```

### 1.1 Build your own sample

To build your own sample, you can follow the examples of functions above to grab coordinates from your favorite literature resource, 

or

You can use [astropy's read](https://docs.astropy.org/en/stable/io/ascii/read.html) function to read in an input table
to an [astropy table](https://docs.astropy.org/en/stable/table/)

+++

### 1.2 Write out your sample to disk

At this point you may wish to write out your sample to disk and reuse that in future work sessions, instead of creating it from scratch again.

For the format of the save file, we would suggest to choose from various formats that fully support astropy objects(eg., SkyCoord).  One example that works is Enhanced Character-Separated Values or ['ecsv'](https://docs.astropy.org/en/stable/io/ascii/ecsv.html)

```{code-cell} ipython3
sample_table.write('data/input_sample.ecsv', format='ascii.ecsv', overwrite = True)
```

### 1.3 Load the sample table from disk

Do only this step from this section when you have a previously generated sample table

```{code-cell} ipython3
sample_table = Table.read('data/input_sample.ecsv', format='ascii.ecsv')
```

### 1.4 Initialize data structure to hold the light curves

```{code-cell} ipython3
# We wrote our own class for a Pandas MultiIndex [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) for storing the light curves
# This class helps simplify coding of common uses for the DataFrame.
df_lc = MultiIndexDFObject()
```

## 2. Find light curves for these targets in NASA catalogs
We search a curated list of time-domain catalogs from NASA astrophysics archives.  Because each archive is different, and in many cases each catalog is different, each function to access a catalog is necessarily specialized to the location and format of that particular catalog.

+++

### 2.1 HEASARC: FERMI & Beppo SAX
The function to retrieve HEASARC data accesses the HEASARC archive using a pyvo search with a table upload.  This is the fastest way to access data from HEASARC catalogs at scale.  

While these aren't strictly light curves, we would like to track if there are gamma rays detected in advance of any change in the CLAGN light curves. We store these gamma ray detections as single data points.  Because gamma ray detections typically have very large error radii, our current technique is to keep matches in the catalogs within some manually selected error radius, currently defaulting to 1 degree for Fermi and 3 degrees for Beppo SAX.  These values are chosen based on a histogram of all values for those catalogs.

```{code-cell} ipython3
start_serial = time.time()  #keep track of all serial archive calls to compare later with parallel archive call time
heasarcstarttime = time.time()

# What is the size of error_radius for the catalogs that we will accept for our cross-matching?
# in degrees; chosen based on histogram of all values for these catalogs
max_fermi_error_radius = str(1.0)  
max_sax_error_radius = str(3.0)

# catalogs to query and their corresponding max error radii
heasarc_catalogs = {"FERMIGTRIG": max_fermi_error_radius, "SAXGRBMGRB": max_sax_error_radius}

# get heasarc light curves in the above curated list of catalogs
df_lc_HEASARC = heasarc_get_lightcurves(sample_table, catalog_error_radii=heasarc_catalogs)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_HEASARC)

print('heasarc search took:', time.time() - heasarcstarttime, 's')
```

### 2.2 IRSA: ZTF
The function to retrieve ZTF light curves accesses a parquet version of the ZTF catalog stored in the cloud using pyarrow.  This is the fastest way to access the ZTF catalog at scale.  The ZTF [API](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html) is available for small sample searches.  One unique thing about this function is that it has parallelization built in to the function itself.

```{code-cell} ipython3
ZTFstarttime = time.time()

# get ZTF lightcurves
# use the nworkers arg to control the amount of parallelization in the data loading step
df_lc_ZTF = ztf_get_lightcurves(sample_table, nworkers=6)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_ZTF)

print('ZTF search took:', time.time() - ZTFstarttime, 's')
```

### 2.3 IRSA: WISE

We use the unWISE light curves catalog ([Meisner et al., 2023](https://ui.adsabs.harvard.edu/abs/2023AJ....165...36M/abstract)) which ties together all WISE & NEOWISE 2010 - 2020 epochs.  Specifically it combines all observations at a single epoch to achieve deeper mag limits than individual observations alone.

The function to retrieve WISE light curves accesses an IRSA generated version of the catalog in parquet format being stored in the AWS cloud [Open Data Repository](https://registry.opendata.aws/collab/nasa/)

```{code-cell} ipython3
WISEstarttime = time.time()

bandlist = ['W1', 'W2']  #list of the WISE band names
WISE_radius = 1.0  # arcsec
# get WISE light curves
df_lc_WISE = wise_get_lightcurves(sample_table, radius=WISE_radius, bandlist=bandlist)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_WISE)

print('WISE search took:', time.time() - WISEstarttime, 's')
```

### 2.4 MAST: Pan-STARRS
The function to retrieve lightcurves from Pan-STARRS currently uses their API; based on this [example](https://ps1images.stsci.edu/ps1_dr2_api.html).  This search is not efficient at scale and we expect it to be replaced in the future.

```{code-cell} ipython3
panstarrsstarttime = time.time()

panstarrs_search_radius = 1.0/3600.0    # search radius = 1 arcsec
# get panstarrs light curves
df_lc_panstarrs = panstarrs_get_lightcurves(sample_table, radius=panstarrs_search_radius)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_panstarrs)

print('Panstarrs search took:', time.time() - panstarrsstarttime, 's')
```

### 2.5 MAST: TESS, Kepler and K2
The function to retrieve lightcurves from these three missions currently uses the open source package [`lightKurve`](https://docs.lightkurve.org/index.html).  This search is not efficient at scale and we expect it to be replaced in the future.

```{code-cell} ipython3
lightkurvestarttime = time.time()

TESS_search_radius = 1.0  #arcseconds
# get TESS/Kepler/K2 light curves
df_lc_TESS = tess_kepler_get_lightcurves(sample_table, radius=TESS_search_radius)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_TESS)

print('TESS/Kepler/K2 search took:', time.time() - lightkurvestarttime, 's')

# LightKurve will return an "Error" when it doesn't find a match for a target
# These are not real errors and can be safely ignored.
```

### 2.6 MAST: Hubble Catalog of Variables ([HCV](https://archive.stsci.edu/hlsp/hcv))
The function to retrieve lightcurves from HCV currently uses their API; based on this [example](https://archive.stsci.edu/hst/hsc/help/HCV/HCV_API_demo.html). This search is not efficient at scale and we expect it to be replaced in the future.

```{code-cell} ipython3
HCVstarttime = time.time()

HCV_radius = 1.0/3600.0 # radius = 1 arcsec
# get HCV light curves
df_lc_HCV = hcv_get_lightcurves(sample_table, radius=HCV_radius)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_HCV)

print('HCV search took:', time.time() - HCVstarttime, 's')
```

## 3. Find light curves for these targets in relevant, non-NASA catalogs

+++

### 3.1 Gaia
The function to retrieve Gaia light curves accesses the Gaia DR3 "source lite" catalog using an astroquery search with a table upload to do the join with the Gaia photometry. This is currently the fastest way to access light curves from Gaia at scale.

```{code-cell} ipython3
gaiastarttime = time.time()

# get Gaia light curves
df_lc_gaia = gaia_get_lightcurves(sample_table, search_radius=1/3600, verbose=0)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_gaia)

print('gaia search took:', time.time() - gaiastarttime, 's')
```

### 3.3 IceCube neutrinos

There are several [catalogs](https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018) (basically one for each year of IceCube data from 2008 - 2018). The following code creates a large catalog by combining
all the yearly catalogs.
The IceCube catalog contains Neutrino detections with associated energy and time and approximate direction (which is uncertain by half-degree scales....). Usually, for active events only one or two Neutrinos are detected, which makes matching quite different compared to "photons". For our purpose, we will list the top 3 events in energy that are within a given distance to the target.

This time series (time vs. neutrino energy) information is similar to photometry. We choose to storing time and energy in our data structure, leaving error = 0. What is __not__ stored in this format is the distance or angular uncertainty of the event direction.

```{code-cell} ipython3
icecubestarttime = time.time()

# get icecube data points
df_lc_icecube = icecube_get_lightcurves(sample_table, icecube_select_topN=3)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_icecube)

print('icecube search took:', time.time() - icecubestarttime, 's')
end_serial = time.time()
```

```{code-cell} ipython3
# benchmarking
print('total time for serial archive calls is ', end_serial - start_serial, 's')
```

## 4. Make plots of luminosity as a function of time
These plots are modelled after [van Velzen et al., 2021](https://arxiv.org/pdf/2111.09391.pdf). We show flux in mJy as a function of time for all available bands for each object. `show_nbr_figures` controls how many plots are actually generated and returned to the screen.  If you choose to save the plots with `save_output`, they will be put in the output directory and labelled by sample number.

__Note__ that in the following, we can either plot the results from `df_lc` (from the serial call) or `parallel_df_lc` (from the parallel call). By default (see next cell) the output of the parallel call is used.

```{code-cell} ipython3
_ = create_figures(df_lc = parallel_df_lc, # either df_lc (serial call) or parallel_df_lc (parallel call)
                   show_nbr_figures = 5,  # how many plots do you actually want to see?
                   save_output = True ,  # should the resulting plots be saved?
                  )
```

## 5. Parallel processing the archive calls

The archive calls can run in parallel.
This may be convenient for samples of any size, but particularly helpful for those larger than about 1,000.
Larger runs come with different challenges, and this is complicated by the fact that different combinations of samples and archive calls can trigger different issues.

The code_src contains a "helper" module to facilitate these runs.
It can be used in combination with any parallel processing method.

This section contains three examples of using the helper, and covers the following parallelization methods:

- Python's `multiprocessing` library. Useful as a demonstration. May be convenient for runs with small to medium sample sizes.
- Command-line script. Recommended for most runs with medium to large samples. Allows ZTF to use additional parallelization internally, and so is often faster (ZTF often takes the longest and returns the most data for AGN-like samples). Writes stdout and stderr to log files, useful for monitoring jobs and resource usage. Can monitor and record `top` to help identify CPU and RAM usage/needs.

**Code for the command-line script is shown in non-executable cells.**
To run it, open a new terminal and copy/paste the cell text.
Also be aware that the script path shown in these cells assumes you are in the same directory as this notebook. Adjust it if needed.

```{code-cell} ipython3
# this section can be run independently using these imports
import json
import multiprocessing
import pandas as pd
import sys

sys.path.append("code_src/")
import bulk_run.helper
from data_structures import MultiIndexDFObject
from plot_functions import create_figures
```

### Parallel: Example 1

This is a basic example of the end-to-end process.
It uses python `multiprocessing` for parallelization and the defaults for most options.

Define basic keyword arguments ("kwargs") for the run:

```{code-cell} ipython3
kwargs_dict = {
    # run_id. Will be used to name the base directory that will be created/used for this run.
    "run_id": "basic-example",
    # Paper names to gather the sample from. Will use default keyword arguments for get_*_sample function calls.
    "get_samples": ["Hon"],
    # Keyword arguments for *_get_lightcurves archive calls. Will use defaults for all except ZTF where we must
    # turn off the internal parallelization because it cannot be combined with a `multiprocessing.Pool`.
    "archives": {"ZTF": {"nworkers": None}},
}
kwargs_dict
```

Decide which archives to query.
This is a separate list because the helper can only run one archive call at a time.
We will iterate over this list and launch each job separately.

```{code-cell} ipython3
archive_names = bulk_run.helper.ARCHIVE_NAMES["core"]  # predefined list ("core" or "all")
# archive_names = ["Gaia", "WISE"]  # choose your own list
archive_names
```

Collect the sample and write it as a .ecsv file:

```{code-cell} ipython3
sample_table = bulk_run.helper.run(build="sample", **kwargs_dict)
# sample_table is returned if you want to look at it but it is not used below
```

Query the archives in parallel using a `multiprocessing.Pool`:

```{code-cell} ipython3
with multiprocessing.Pool(processes=len(archive_names)) as pool:
    # submit one job per archive
    for archive in archive_names:
        pool.apply_async(bulk_run.helper.run, kwds={"build": "lightcurves", "archive": archive, **kwargs_dict})
    pool.close()  # signal that no more jobs will be submitted to the pool
    pool.join()  # wait for all jobs to complete

# Note: The console output from different archive calls gets jumbled together below.
# Worse, error messages tend to get lost in the background and never displayed.
# If you have trouble, consider running an archive call individually without the Pool
# or using the command-line script instead.
```

The light curve data is saved as a parquet dataset in the "parquet_dir" directory.
Load it:

```{code-cell} ipython3
# copy/paste the directory path from the output above, or ask the helper for it like this:
parquet_dir = bulk_run.helper.run(build="parquet_dir", **kwargs_dict)
df_lc = pd.read_parquet(parquet_dir)

df_lc.head()
```

Now we can make figures:

```{code-cell} ipython3
_ = create_figures(df_lc=MultiIndexDFObject(data=df_lc), show_nbr_figures=1, save_output=False)
```

### Parallel: Example 2

This example shows the `kwargs_dict` options in more detail.
A basic introduction to the command-line script is also given at the end.

`kwargs_dict` can contain:
  &bull; keyword arguments for any of the `get_*_sample` functions.
  &bull; keyword arguments for any of the `*_get_lightcurves` functions.
  &bull; keyword arguments used directly by the helper. These options and their defaults are shown below, further documented in the helper's `run` function. 

```{code-cell} ipython3
bulk_run.helper.DEFAULTS
```

It can be convenient to save the parameters in a yaml file, especially when using the command-line script or in cases where you want to store them for later reference or re-use.

Define parameters for a run with a diverse sample and custom kwargs for the archive calls, and save it as a yaml file:

```{code-cell} ipython3
run_id = "extended-example"  # we'll need to use the same run_id in several steps

get_samples = {
    "green": {},
    "ruan": {},
    "papers_list": {
        "paper_kwargs": [
            {"paper_link": "2022ApJ...933...37W", "label": "Galex variable 22"},
            {"paper_link": "2020ApJ...896...10B", "label": "Palomar variable 20"},
        ]
    },
    "SDSS": {"num": 10, "zmin": 0.5, "zmax": 2, "randomize_z": True},
    "ZTF_objectid": {"objectids": ["ZTF18aabtxvd", "ZTF18aahqkbt", "ZTF18abxftqm", "ZTF18acaqdaa"]},
}

archives = {
    "Gaia": {"search_radius": 2 / 3600},
    "HEASARC": {"catalog_error_radii": {"FERMIGTRIG": 1.0, "SAXGRBMGRB": 3.0}},
    "IceCube": {"icecube_select_topN": 4, "max_search_radius": 2.0},
    "WISE": {"radius": 1.5, "bandlist": ["W1", "W2"]},
    "ZTF": {"nworkers": 6, "match_radius": 2 / 3600},
}

kwargs_dict = {
    "get_samples": get_samples,
    "consolidate_nearby_objects": False,
    "archives": archives,
}

bulk_run.helper.write_kwargs_to_yaml(run_id=run_id, **kwargs_dict)
```

The path to the yaml file is printed in the output above.
You can alter the contents of the file as you like.

To use the file with the python commands shown in the previous example, set the kwarg `use_yaml=True`.
Be sure to use the same `run_id` as when writing the yaml.
Here's an example for the "sample" step:

```{code-cell} ipython3
sample_table = bulk_run.helper.run(build="sample", run_id=run_id, use_yaml=True)
```

To use the file with the command-line script, the basic call is:

```{raw-cell}
run_id=extended-example  # use the same run_id as when writing the yaml
./code_src/bulk_run/light_curve_generator.sh -r "$run_id" -d "use_yaml=true" -a "core"
```

The flags used in the call above are:

- `-r` (run_id)
- `-d` (dict). Any top-level item in kwargs_dict (syntax: "key=value") whose value is a basic type (e.g., bool or string but not list or dict.) Repeat the flag to send multiple kwargs.
- `-a` (archive_names). "all", "core", or space-separated list of names like "Gaia IceCube WISE"

### Parallel: Example 3

This example shows how to launch and monitor a large-scale job using the command-line script.

light_curve_generator.sh is a command-line script that can be used to execute a complete run with one call, including sample collection and running the archive calls in parallel.
This method is more powerful than python `multiprocessing` and is recommended for larger runs and/or when you need to monitor a run more closely for errors, RAM and CPU usage, etc.

The previous example showed how to submit kwargs with a yaml file, which is convenient for an extended set of kwargs.
However, large-scale jobs don't always require such extensive parameter definitions.
In such cases, it can be more convenient to submit kwargs as a json string.
Python can be used to construct the string:

```{code-cell} ipython3
kwargs_dict = {
    "get_samples": {"SDSS": {"num": 500_000}},  # 500,000 sample objects = very large-scale run
    "archives": {"ZTF": {"nworkers": 8}}  # around 8 ZTF workers is often good, but monitor resource usage
}
json.dumps(kwargs_dict)
```

Copy the json string output above, including the surrounding single quotes ('), and use it to launch a run with the `-j` flag like this:

```{raw-cell}
# This run will require a minimum of 4 CPU and 100G RAM, and should complete in about 5-10 hours.
./code_src/bulk_run/light_curve_generator.sh \
    -r "SDSS-500k" \
    -j '{"get_samples": {"SDSS": {"num": 500000}}, "archives": {"ZTF": {"nworkers": 8}}}' \
    -a "core"
```

The script will gather the sample, launch the archive calls in parallel, then exit.
Archive jobs will continue running in the background until they either complete or encounter an error.

You can cancel the run at any time.
If the script is still running, press `Control-C`.
If the script has exited, call it again with the `-k` flag to kill all processes (jobs) that were launched:

```{raw-cell}
# This will kill all processes that were started by the run with the given run_id.
./code_src/bulk_run/light_curve_generator.sh -r "SDSS-500k" -k
```

There are multiple ways to monitor the run, including:
- Check the logs for job status or errors. The script will redirect stdout and stderr to files and print out the paths for you.
- Check for parquet (light curve) data. The script will print out the "parquet_dir". `ls` this directory. You will see a subdirectory for each archive call that has completed successfully, assuming it retrieved data for the sample.
- Monitor resource usage with the `top` command. The script will print the job PIDs. The script can also monitor `top` for you and save the output to a log file. Use the `-t` flag as shown below:

```{raw-cell}
# This will print `top` to stdout and save it to a file once per interval.
interval=10m  # will be passed to the `sleep` command
./code_src/bulk_run/light_curve_generator.sh -r "SDSS-500k" -t "$interval"
```

The script will continue running until sometime after all jobs launched with the given run_id have completed.
You can cancel at anytime with `Control-C` and start it again with a new interval.

Once the `top` output has been saved, the helper can load it to pandas DataFrames:

```{code-cell} ipython3
top_summary_df, top_pid_df = bulk_run.helper.load_top_log(run_id)

top_summary_df.plot("time", "used_GiB", kind="scatter")
```

## References

This work made use of:

&bull; Astroquery; Ginsburg et al., 2019, 2019AJ....157...98G  
&bull; Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013,    2022ApJ...935..167A, 2018AJ....156..123A, 2013A&A...558A..33A  
&bull; Lightkurve; Lightkurve Collaboration 2018, 2018ascl.soft12013L  
&bull; acstools; https://zenodo.org/record/7406933#.ZBH1HS-B0eY  
&bull; unWISE light curves; Meisner et al., 2023, 2023AJ....165...36M  

```{code-cell} ipython3

```
