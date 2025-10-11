This function takes a list of commands (separated by newlines) and executes them up to **N at a time** using a semaphore implemented with a temporary FIFO pipe. It is quite useful to run multiple instances in the terminal in parallel.
```bash
#!/bin/bash

run_parallel_jobs() {
    local max_jobs=$1
    local commands=$2

    if [[ -z "$max_jobs" || -z "$commands" ]]; then
        echo "Usage: run_parallel_jobs <max_jobs> <commands_string>" >&2
        return 1
    fi

    local fifo_file
    fifo_file=$(mktemp -u)
    mkfifo "$fifo_file"
    exec 3<>"$fifo_file"

    echo "Initializing semaphore..."
    for i in $(seq "$max_jobs"); do
        # Writing to the FIFO gives a token/slot
        echo >&3
    done

    echo "Starting parallel execution with a concurrency limit of $max_jobs..."

    echo "$commands" | while IFS= read -r cmd; do
        if [[ -z "$cmd" ]]; then
            continue # Skip empty lines
        fi

        # WAIT: Read a token from the semaphore (blocking operation)
        read -r -u 3
        (
            echo "[RUNNING] $cmd"
            eval "$cmd"
            # SIGNAL: Write a token back to the semaphore when finished
            echo >&3
        ) &
    done

    wait

    echo "All jobs completed."

    exec 3>&-
    rm -f "$fifo_file"
}
```

Run optimization in parallel:
```bash
#!/bin/bash

ALL_COMMANDS=$(cat <<-END_CMDS
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1B.json" -t 120s -f "batch and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1B.json" -t 120s -f "skewed-1 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1B.json" -t 120s -f "skewed-2 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1B.json" -t 120s -f "skewed-4 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1B.json" -t 120s -f "skewed-8 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1B.json" -t 120s -f "skewed-16 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1B.json" -t 120s -f "batch and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1B.json" -t 120s -f "skewed-1 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1B.json" -t 120s -f "skewed-2 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1B.json" -t 120s -f "skewed-4 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1B.json" -t 120s -f "skewed-8 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1B.json" -t 120s -f "skewed-16 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1A.json" -t 120s -f "batch and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1A.json" -t 120s -f "skewed-1 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1A.json" -t 120s -f "skewed-2 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1A.json" -t 120s -f "skewed-4 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1A.json" -t 120s -f "skewed-8 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP100-1A.json" -t 120s -f "skewed-16 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3

export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1A.json" -t 120s -f "batch and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1A.json" -t 120s -f "skewed-1 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1A.json" -t 120s -f "skewed-2 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1A.json" -t 120s -f "skewed-4 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1A.json" -t 120s -f "skewed-8 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3
export PYTHONPATH='.' && uv run ./src/problems/mokp/cli.py run -i "./data/mokp/2KP200-1A.json" -t 120s -f "skewed-16 and (k3 or k2) and BI and op_flip_v2 and (shake_flip or shake_swap)" --repeat-times 3

END_CMDS
)

run_parallel_jobs 12 "$ALL_COMMANDS"
```

Calculate metrics per run:
```bash
export PYTHONPATH='.' && uv run python ./src/problems/mokp/cli.py metrics -i "./data/mokp/2KP*" --export
```
