#!/bin/bash

# Exit if SLURM job ID isn't provided as a command line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    echo "Error: No job_id argument provided."
    exit 1
fi

# Enable nullglob so unmatched globs expand to nothing
shopt -s nullglob  

job_id=$1
LOGS_DIR="./output/logs"

# Check for ./output/logs
if [[ ! -d "$LOGS_DIR" ]]; then
    echo "No logs directory found at ./$LOGS_DIR"
    exit 1
fi

# Check for subdirectories under ./output/logs
subdirs=("$LOGS_DIR"/*)
if (( ${#subdirs[@]} == 0 )); then
    echo "No job name subdirectory found under ./$LOGS_DIR"
    exit 1
fi

# Check for ./output/logs/*/err 
err_dirs=("$LOGS_DIR"/*/err)
if (( ${#err_dirs[@]} == 0 )); then
    echo "No error directory found under ./$LOGS_DIR/*"
    exit 1
fi

# Check for ./output/logs/*/err/job_id 
job_dirs=("$LOGS_DIR"/*/err/"$job_id")
if (( ${#job_dirs[@]} == 0 )); then
    echo "No directory for job ${job_id} found under $LOGS_DIR/*/err/"
    exit 1
fi

# Check for .err files under ./output/logs/*/err/job_id 
err_files=("$LOGS_DIR"/*/err/"$job_id"/*.err)
if (( ${#err_files[@]} == 0 )); then
    echo "No .err files found under ./$LOGS_DIR/*/err/$job_id"
    exit 1
fi

num_err_files=${#err_files[@]}

printf '%*s\n' 80 '' | tr ' ' '-'
echo "Found $num_err_files .err files for job $job_id"
echo "Reading them now..."

# Create dump directory
dump="./output/dump"
mkdir -p "$dump"
output_file="$dump/${job_id}.csv"

# Write CSV header
csv_header="job_name,array_task_id,has_error,error_message"
echo "$csv_header" > "$output_file"

num_files_read=0
num_failed_jobs=0
job_name=""
error_files=()
failed_task_ids=()

for err_file in "${err_files[@]}"; do
    # Skip file if it doesn't exists
    [ -e "$err_file" ] || continue

    num_files_read=$((num_files_read + 1))

    # Parse job_name from ./logs/<job_name>/err/<job_id>/<task_id>.err
    job_name=$(echo "$err_file" | cut -d'/' -f2)
    
    # task_id is the file name without the .err extension
    task_id=$(basename "$err_file" .err)

    has_error="no"

    if [ -s "$err_file" ]; then
        if grep -q "Traceback" "$err_file"; then
            has_error="yes"
            num_failed_jobs=$((num_failed_jobs + 1))
            error_files+=("$err_file")
             failed_task_ids+=("$task_id")

            # Extract first line of the traceback block for preview
            error_message=$(awk '/Traceback/ {f=1} f' "$err_file" | tr -d '\000-\011\013\014\016-\037' | tr '\n' '␤' | sed 's/"/""/g')
        fi
    fi

    # Save row to CSV file
    printf '%s,%s,%s,%s,"%s"\n' "$job_name" "$task_id" "$has_error" "$error_message" >> "$output_file"
done

printf '%*s\n' 80 '' | tr ' ' '-'
echo "Finished reading ${num_err_files} .err files"

done_dir="${LOGS_DIR}/${job_name}/done/${job_id}"

if [ -d "$done_dir" ]; then
    num_done_files=$(find "$done_dir" -maxdepth 1 -type f -name "*.done" | wc -l)
    echo "Found $num_done_files .done files for job ${job_id}"
else
    echo "Directory $done_dir does not exist"
fi

printf '%.0s-' {1..80}
echo  # To add a newline
echo "${num_failed_jobs} job(s) failed"
echo ""

if [ "$num_failed_jobs" -gt 0 ]; then
    echo "Files with errors:"
    for err_file_name in "${error_files[@]}"; do
        echo "- $err_file_name"
    done

    if [ "${#failed_task_ids[@]}" -gt 0 ]; then
        IFS=','  
        echo "Failed task IDs: --array=${failed_task_ids[*]}"
        unset IFS  
    fi
    echo "" 
    echo "See job ${job_id}'s error summary at ${output_file}"
else
    rm "$output_file"
    echo "No errors found"
fi


