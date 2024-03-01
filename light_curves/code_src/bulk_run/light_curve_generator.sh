#
RUN_HELPER_PY(){
    build=$1
    archive=$2
    HELPER_PY="$(dirname "$0")/helper.py"

    if [ $archive ]; then
        nohup python $HELPER_PY --build $build --archive $archive --kwargs_json "$kwargs_json" --kwargs_dict ${kwargs_dict[@]} &
    else
        python $HELPER_PY --build $build --kwargs_json "$kwargs_json" --kwargs_dict ${kwargs_dict[@]}
    fi
}

get_running_pids(){
    logs_dir=$1

    # scrape log files and collect all PIDs into an array
    for file in "$logs_dir"/*.log; do
        # use regex to match the number in a string with the syntax [pid=1234]
        # https://unix.stackexchange.com/questions/13466/can-grep-output-only-specified-groupings-that-match
        # ----
        # [TODO] THIS WILL FAIL ON MacOS (invalid option '-P') but I haven't found a more general solution
        # https://stackoverflow.com/questions/77662026/grep-invalid-option-p-error-when-doing-regex-in-bash-script
        # (the '-P' is required for the look-behind, '\K', which excludes "[pid=" from the returned result)
        # ----
        all_pids+=($(grep -oP '\[pid=\K\d+' $file))
    done

    # deduplicate the array
    pids=($(for pid in "${all_pids[@]}"; do echo $pid; done | sort --sort=numeric --unique))
    # add currently running python PIDs
    # https://stackoverflow.com/questions/21470362/find-the-pids-of-running-processes-and-store-as-an-array
    pids+=($(ps -ef | grep python | awk '{print $2}'))
    # get only values that were in both lists (which are now duplicates in pids)
    running_pids=($(for pid in "${pids[@]}"; do echo $pid; done | sort | uniq --repeated))

    echo ${running_pids[@]}
}

kill_all_pids(){
    run_id=$1
    logs_dir=$2

    kill_pids=($(get_running_pids $logs_dir))

    # killing processes can be dangerous. make the user confirm.
    echo "WARNING, you are about to kill all processes started by the run run_id='${run_id}'."
    echo "This includes at least the following PIDs: ${kill_pids[@]}"
    echo "Enter 'y' to continue or any other key to abort: "
    read continue_kill
    continue_kill="${continue_kill:-n}"

    if [ $continue_kill == "y" ]; then
        echo "Killing."
        # processes may have started or ended while we waited for the user to confirm, so fetch them again
        kill_pids=($(get_running_pids $logs_dir))
        # kill
        for pid in "${kill_pids[@]}"; do kill $pid; done
    else
        echo "Aborting."
        # echo "Aborting." | tee -a ${logfile}
    fi
}

monitor_top(){
    nsleep=$1
    logs_dir=$2
    logfile="${logs_dir}/top.txt"

    running_pids=(1)  # just need a non-zero length array to start
    while [ ${#running_pids[@]} -gt 0 ]; do
        running_pids=($(get_running_pids $logs_dir))
        pid_flags=()
        for pid in ${running_pids[@]}; do pid_flags+=("-p${pid}"); done

        if [ ${#running_pids[@]} -gt 0 ]; then
            {
                echo ----
                date "+%Y/%m/%d %H:%M:%S %Z"
                top -b -n1 -o-PID ${pid_flags[@]}
            } | tee -a $logfile

            sleep $nsleep
        fi
    done
}

print_help_me(){
    echo "For help, call this script with the '-h' flag:"
    echo "    \$ ./$(basename $0) -h"
}

print_help_run_instructions(){
    echo "For instructions on monitoring a run and loading the output, use:"
    echo "    \$ ./$(basename $0) -i"
}

print_logs(){
    logfile=$1
    echo "-- ${logfile}:"
    cat $logfile
    echo "--"
}

print_run_instructions(){
    echo "When this script is executed it launches a 'run' consisting of multiple 'jobs':"
    echo "    - one job for this script"
    echo "    - one job for the function that builds the object sample"
    echo "    - one job for each call to an archive"
    echo
    echo "---- Check Progress ----"
    echo "There are several ways to check the progress, listed below."
    echo
    echo "1: Logs"
    echo "To check a job's progress, view its log file by setting the 'logfile' variable (you can"
    echo "copy/paste this from the script's output), then using:"
    echo "    \$ cat \$logfile"
    echo "View this script's log for high-level job info and variables to copy/paste."
    echo "View an archive's log to check its progress."
    echo
    echo "2: 'top'"
    echo "Use 'top' to monitor job activity (job PID's are in script output) and overall resource usage."
    echo
    echo "3: Output"
    echo "Once light curves are loaded, the data will be written to a parquet dataset with"
    echo "one partition for each archive."
    echo "To check which archives are done, set the 'parquet_dir' variable (copy/paste from script"
    echo "output), then use:"
    echo "    \$ ls -l \$parquet_dir"
    echo "You will see a directory for each archive that is complete."
    echo
    echo "---- Kill Jobs ----"
    echo "Kill one job:"
    echo "If a particular archive encounters problems and you need to kill its job, kill the process."
    echo "Set a 'pid' variable (copy/paste from script output), then use:"
    echo "    \$ kill \$pid"
    echo
    echo "Kill all jobs:"
    echo "If, at any point, you want to cancel all jobs launch by the run, killing every process,"
    echo "call the script again using the flags '-r' (with same value as before) and '-k'."
    echo "For example, if your run ID is 'my_run_id':"
    echo "    \$ ./$(basename $0) -r my_run_id -k"
    echo
    echo "---- Load the Light Curves (after jobs complete) ----"
    echo "Light curves will be written to a parquet dataset."
    echo "To load the data (in python, with pandas) set the 'parquet_dir' variable (copy/paste from"
    echo "script output), then use:"
    echo "    >>> df_lc = pd.read_parquet(parquet_dir)"
    echo

}

print_usage(){
    echo "---- Usage Examples ----"
    echo "  - Basic run with default options:"
    echo "    \$ ./$(basename $0) -r my_run_id "
    echo "  - Specify two papers to get sample objects from, and two archives to fetch light curves from:"
    echo "    \$ ./$(basename $0) -r run_two -l 'yang hon' -m 'gaia wise'"
    echo "  - Get ZTF light curves for 200 objects from the SDSS sample:"
    echo "    \$ ./$(basename $0) -r ztf -l SDSS -n 200 -m ztf"
    echo "  - If something went wrong with the 'ztf' run and you want to kill it:"
    echo "    \$ ./$(basename $0) -r ztf -k"
    echo
    echo "---- Available Flags ----"
    echo "Flags that require a value (defaults in parentheses):"
    echo "    -r : ID for the run. Used to label the output directory. There is no default."
    echo "         A value is required. No spaces or special characters."
    echo "    -l ('yang SDSS') : Space-separated list of literature/paper names from which to build the object sample."
    echo "    -c (true) : whether to consolidate_nearby_objects"
    echo "    -g ('{"SDSS": {"num": 10}}') : json string representing dicts with keyword arguments."
    echo "    -m ('gaia heasarc icecube wise ztf') : Space-separated list of archives from which to load light curves."
    echo "    -o (object_sample.ecsv) : File name storing the object sample."
    echo "    -p (lightcurves.parquet) : Directory name storing the light-curve data."
    echo "Flags to be used without a value:"
    echo "    -h : Print this help message."
    echo "    -i : Print instructions on monitoring a run and loading the output."
    echo "    -k : Kill the entire run by killing all jobs/processes started with the specified run ID ('-r')."
    echo
    echo "---- Instructions for Monitoring a Run ----"
    print_help_run_instructions
}

# ---- Set variable defaults.
archive_names=()  # "core", "all", or "gaia wise <...>"
kwargs_dict=()
kwargs_json='{}'
kill_all_processes=false

# ---- Set variables that were passed in as script arguments.
# info about getopts: https://www.computerhope.com/unix/bash/getopts.htm#examples
while getopts r:a:d:j:t:hik flag; do
    case $flag in
        r) run_id=$OPTARG
            kwargs_dict+=("run_id=${OPTARG}")
            ;;
        a) archive_names=("$OPTARG");;
        d) kwargs_dict+=("$OPTARG");;
        j) kwargs_json=$OPTARG;;
        t) nsleep=$OPTARG;;
        h) print_usage
            exit 0
            ;;
        i) print_run_instructions
            exit 0
            ;;
        k) kill_all_processes=true;;
        ?) print_help_me
            exit 1
            ;;
      esac
done

# If a run_id was not supplied, exit.
if [ -z $run_id ]; then
    echo "./$(basename $0): missing required option -- 'r'"
    print_help_me
    exit 1
fi

# ---- Request some kwarg values from the helper.
base_dir=$(RUN_HELPER_PY base_dir+)
# if HELPER_PY didn't create base_dir then something is wrong and we need to exit
if [ ! -d "$base_dir" ]; then
    echo "${base_dir} does not exist. Exiting."
    exit 1
fi
parquet_dir=$(RUN_HELPER_PY parquet_dir+)
logs_dir=$(RUN_HELPER_PY logs_dir+)
# expand an archive_names shortcut value.
if [ "${archive_names[0]}" == "all" ]; then archive_names=($(RUN_HELPER_PY archive_names_all+l)); fi
if [ "${archive_names[0]}" == "core" ]; then archive_names=($(RUN_HELPER_PY archive_names_core+l)); fi

# ---- Construct logs paths.
# logs_dir="${base_dir}/logs"
mkdir -p $logs_dir
mylogfile="${logs_dir}/$(basename $0).log"

# ---- If the user has requested to monitor with top, do it and then exit.
if [ $nsleep ]; then
    monitor_top $nsleep $logs_dir
    exit 0
fi

{  # we will tee the output of everything below here to $mylogfile

# ---- If the user has requested to kill processes, do it and then exit.
if [ $kill_all_processes == true ]; then
    kill_all_pids $run_id $logs_dir $mylogfile
    exit 0
fi

# ---- Report basic info about the run.
echo "*********************************************************************"
echo "**                          Run starting.                          **"
echo "run_id=${run_id}"
echo "base_dir=${base_dir}"
echo "logs_dir=${logs_dir}"
echo "parquet_dir=${parquet_dir}"
echo "**                                                                 **"
echo

# ---- Do the run. ---- #

# ---- 1: Run job to get the object sample, if needed. Wait for it to finish.
logfile_name="get_sample.log"
logfile="${logs_dir}/${logfile_name}"
echo
echo "Build sample is starting. logfile=${logfile_name}"
RUN_HELPER_PY sample >> ${logfile} 2>&1
echo "Build sample is done. Printing the log for convenience:"
echo
print_logs $logfile

# ---- 2: Start the jobs to fetch the light curves in the background. Do not wait for them to finish.
echo
echo "Archive calls are starting."
echo
for archive in ${archive_names[@]}; do
    logfile_name="$(awk '{ print tolower($0) }' <<< $archive).log"
    logfile="${logs_dir}/${logfile_name}"
    RUN_HELPER_PY lightcurves $archive >> ${logfile} 2>&1
    echo "[pid=${!}] ${archive} started. logfile=${logfile_name}"
done

# ---- 3: Print some instructions for the user, then exit.
# echo
# echo "Light curves are being loaded in background processes. PIDs are listed above."
# echo "Once loaded, data will be written to a parquet dataset in"
# echo "parquet_dir=${parquet_dir}"
echo
print_help_run_instructions
echo
echo "**                       Main script exiting.                       **"
echo "**           Jobs may continue running in the background.           **"
echo "**********************************************************************"
} | tee -a $mylogfile
