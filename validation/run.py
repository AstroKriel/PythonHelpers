## { SCRIPT

##
## === DEPENDENCIES
##

import sys
import time
import subprocess

from pathlib import Path

from jormi.ww_io import log_manager

##
## === MAIN PROGRAM
##


def main():
    ## discover all validation scripts
    validation_root = Path(__file__).parent
    scripts = sorted(validation_root.rglob("test_*.py"))
    if not scripts:
        log_manager.log_alert("No validation scripts found.")
        sys.exit(0)
    log_manager.log_section("Validation Suite", show_time=True)
    log_manager.log_empty_lines()
    ## run each script as a subprocess and collect results
    results: list[tuple[str, bool, float]] = []
    for script_path in scripts:
        label = str(script_path.relative_to(validation_root))
        log_manager.log_task(label, show_time=False)
        start_time = time.perf_counter()
        process = subprocess.run(
            args=[sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        elapsed_time = time.perf_counter() - start_time
        passed = process.returncode == 0
        results.append((label, passed, elapsed_time))
        if passed:
            log_manager.log_action(
                title=label,
                outcome=log_manager.ActionOutcome.SUCCESS,
                notes={"elapsed": f"{elapsed_time:.2f}s"},
            )
        else:
            output = (process.stdout + process.stderr).strip()
            log_manager.log_action(
                title=label,
                outcome=log_manager.ActionOutcome.FAILURE,
                notes={
                    "elapsed": f"{elapsed_time:.2f}s",
                    "output": output[:500] if output else "(no output)",
                },
            )
    ## print summary and exit with non-zero code if any script failed
    num_passed = sum(1 for _, passed, _ in results if passed)
    num_total = len(results)
    total_elapsed_time = sum(elapsed_time for _, _, elapsed_time in results)
    log_manager.log_summary(
        title="Validation Results",
        notes={
            label: f"{'pass' if passed else 'FAIL'} ({elapsed_time:.2f}s)"
            for label, passed, elapsed_time in results
        },
        message=f"{num_passed}/{num_total} scripts passed in {total_elapsed_time:.2f}s.",
    )
    if num_passed < num_total:
        sys.exit(1)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
