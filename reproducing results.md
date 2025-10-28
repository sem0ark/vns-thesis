Useful utility script to run jobs in parallel:

```bash
run_parallel_jobs() {
    local max_jobs=$1
    local commands=$2
    
    # Check for mandatory arguments
    if [[ -z "$max_jobs" || -z "$commands" ]]; then
        echo "Usage: run_parallel_jobs <max_jobs> <commands_string>" >&2
        return 1
    fi

    # 1. Create a unique, temporary FIFO file
    local fifo_file
    fifo_file=$(mktemp -u)
    mkfifo "$fifo_file"

    # 2. Open the FIFO on file descriptor 3 for read/write access
    # This allows the loop to run in parallel with the subshells.
    exec 3<>"$fifo_file"

    # 3. Initialize the semaphore with tokens equal to max_jobs
    echo "Initializing semaphore..."
    for i in $(seq "$max_jobs"); do
        # Writing to the FIFO gives a token/slot
        echo >&3 
    done

    echo "Starting parallel execution with a concurrency limit of $max_jobs..."
    
    # 4. Read commands line by line and execute
    # The read -u 3 waits for a token before proceeding.
    echo "$commands" | while IFS= read -r cmd; do
        if [[ -z "$cmd" ]]; then
            continue # Skip empty lines
        fi

        # WAIT: Read a token from the semaphore (blocking operation)
        read -r -u 3

        # Execute the command in a background subshell
        (
            echo "[RUNNING] $cmd"
            eval "$cmd"
            # SIGNAL: Write a token back to the semaphore when finished
            echo >&3
        ) &
    done 

    # 5. Wait for all background jobs (subshells) to complete
    wait

    echo "All jobs completed."

    # 6. Clean up: Close the file descriptor and remove the FIFO file
    exec 3>&- 
    rm -f "$fifo_file"
}
```


```bash
ALL_COMMANDS=$(cat <<-END_CMDS

export PYTHONPATH='.' && uv run ./src/problems/moscp/cli.py run -i "./data/moscp/2scp11*.json" -i "./data/moscp/2scp41*.json" -i "./data/moscp/2scp61*.json" -i "./data/moscp/2scp81*.json" -i "./data/moscp/2scp101*.json" -i "./data/moscp/2scp201*.json" -t 120s -f "batch and k3 and noop and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/moscp/cli.py run -i "./data/moscp/2scp11*.json" -i "./data/moscp/2scp41*.json" -i "./data/moscp/2scp61*.json" -i "./data/moscp/2scp81*.json" -i "./data/moscp/2scp101*.json" -i "./data/moscp/2scp201*.json" -t 120s -f "batch and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/moscp/cli.py run -i "./data/moscp/2scp11*.json" -i "./data/moscp/2scp41*.json" -i "./data/moscp/2scp61*.json" -i "./data/moscp/2scp81*.json" -i "./data/moscp/2scp101*.json" -i "./data/moscp/2scp201*.json" -t 120s -f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/moscp/cli.py run -i "./data/moscp/2scp11*.json" -i "./data/moscp/2scp41*.json" -i "./data/moscp/2scp61*.json" -i "./data/moscp/2scp81*.json" -i "./data/moscp/2scp101*.json" -i "./data/moscp/2scp201*.json" -t 120s -f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0

export PYTHONPATH='.' && uv run ./src/problems/moscp/cli.py run -i "./data/moscp/2scp11*.json" -i "./data/moscp/2scp41*.json" -i "./data/moscp/2scp61*.json" -i "./data/moscp/2scp81*.json" -i "./data/moscp/2scp101*.json" -i "./data/moscp/2scp201*.json" -t 120s -f "pymoo and pop_500" --repeat-times 30 --seed 0

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "batch and k3 and noop and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "batch and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "batch and k3 and noop and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "batch and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "pymoo and pop_500" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "pymoo and pop_500" --repeat-times 30 --seed 0

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "skewed_v3 and a1 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP50-1*.json" -i "./data/mokp/2KP100-1*.json" -i "./data/mokp/2KP150-1*.json" -i "./data/mokp/2KP200-1*.json" -i "./data/mokp/2KP250-1*.json" -t 120s -f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "skewed_v3 and a1 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP300-1*.json" -i "./data/mokp/2KP350-1*.json" -i "./data/mokp/2KP400-1*.json" -i "./data/mokp/2KP450-1*.json" -i "./data/mokp/2KP500-1*.json" -t 120s -f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip" --repeat-times 30 --seed 0


END_CMDS
)

run_parallel_jobs 8 "$ALL_COMMANDS"
```


Export metrics results for two given problems via:
```bash
export PYTHONPATH='.' && uv run python ./src/problems/mokp/cli.py metrics -i "./data/mokp/2KP*.json" -f "(vns and v18) or (pymoo and v5) and 120s" --export
export PYTHONPATH='.' && uv run python ./src/problems/moscp/cli.py metrics -i "./data/moscp/2scp*.json" -f "(vns and v18) or (pymoo and v5) and 120s" --export
```

Get plots, based on the grouping and configuration you want:
```bash
export PYTHONPATH='.' && uv run ./src/cli/multi_instance_metrics.py -i "./metrics/MOKP_2KP100*.json" \
-f "batch and k3 and noop and shake_flip and v18 and 120s" -n "RVNS" \
-f "batch and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "BVNS" \
-f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S3-BVNS a0.25" \
-f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.25" \
-f "skewed_v4 and a0.5 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.5" \
-f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a1" \
--plot-file ~/Downloads/svns_final_comparison_2KP100_120s_no_pymoo.png

export PYTHONPATH='.' && uv run ./src/cli/multi_instance_metrics.py -i "./metrics/MOKP_2KP100*.json" \
-f "pymoo and nsga2 and pop_500 and v5 and 120s" -n "NSGA2 pop=500" \
-f "pymoo and spea2 and pop_500 and v5 and 120s" -n "SPEA2 pop=500" \
-f "batch and k3 and noop and shake_flip and v18 and 120s" -n "RVNS" \
-f "batch and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "BVNS" \
-f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S3-BVNS a0.25" \
-f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.25" \
-f "skewed_v4 and a0.5 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.5" \
-f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a1" \
--plot-file ~/Downloads/svns_final_comparison_2KP100_120s.png


export PYTHONPATH='.' && uv run ./src/cli/multi_instance_metrics.py -i "./metrics/MOKP_2KP200*.json" \
-f "batch and k3 and noop and shake_flip and v18 and 120s" -n "RVNS" \
-f "batch and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "BVNS" \
-f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S3-BVNS a0.25" \
-f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.25" \
-f "skewed_v4 and a0.5 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.5" \
-f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a1" \
--plot-file ~/Downloads/svns_final_comparison_2KP200_120s_no_pymoo.png

export PYTHONPATH='.' && uv run ./src/cli/multi_instance_metrics.py -i "./metrics/MOKP_2KP200*.json" \
-f "pymoo and nsga2 and pop_500 and v5 and 120s" -n "NSGA2 pop=500" \
-f "pymoo and spea2 and pop_500 and v5 and 120s" -n "SPEA2 pop=500" \
-f "batch and k3 and noop and shake_flip and v18 and 120s" -n "RVNS" \
-f "batch and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "BVNS" \
-f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S3-BVNS a0.25" \
-f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.25" \
-f "skewed_v4 and a0.5 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.5" \
-f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a1" \
--plot-file ~/Downloads/svns_final_comparison_2KP200_120s.png


export PYTHONPATH='.' && uv run ./src/cli/multi_instance_metrics.py -i "./metrics/MOKP_2KP*.json" \
-f "batch and k3 and noop and shake_flip and v18 and 120s" -n "RVNS" \
-f "batch and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "BVNS" \
-f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S3-BVNS a0.25" \
-f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.25" \
-f "skewed_v4 and a0.5 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.5" \
-f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a1" \
--plot-file ~/Downloads/svns_final_comparison_2KP_all_120s_no_pymoo.png

export PYTHONPATH='.' && uv run ./src/cli/multi_instance_metrics.py -i "./metrics/MOKP_2KP*.json" \
-f "pymoo and nsga2 and pop_500 and v5 and 120s" -n "NSGA2 pop=500" \
-f "pymoo and spea2 and pop_500 and v5 and 120s" -n "SPEA2 pop=500" \
-f "batch and k3 and noop and shake_flip and v18 and 120s" -n "RVNS" \
-f "batch and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "BVNS" \
-f "skewed_v3 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S3-BVNS a0.25" \
-f "skewed_v4 and a0.25 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.25" \
-f "skewed_v4 and a0.5 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a0.5" \
-f "skewed_v4 and a1 and k3 and BI and op_flip and shake_flip and v18 and 120s" -n "S4-BVNS a1" \
--plot-file ~/Downloads/svns_final_comparison_2KP_all_120s.png
```
