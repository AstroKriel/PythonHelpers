## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import subprocess
import sys
import time

from pathlib import Path

## local
from jormi.ww_io import manage_log

##
## === MAIN PROGRAM
##


def main():
    ## discover all demo scripts, which are every script here but this one
    demos_root = Path(__file__).parent
    demo_scripts = sorted(
        script_path for script_path in demos_root.rglob("*.py")
        if script_path.name != Path(__file__).name
    )
    if not demo_scripts:
        manage_log.log_alert(text="No demo scripts found.")
        sys.exit(0)
    manage_log.log_section(
        title="Demo Suite",
        show_time=True,
    )
    manage_log.log_empty_lines()
    ## run each script as a subprocess and collect results
    results: list[tuple[str, bool, float]] = []
    for script_path in demo_scripts:
        demo_label = str(
            script_path.relative_to(
                demos_root,
            ),
        )
        manage_log.log_task(
            text=demo_label,
            show_time=False,
        )
        start_time = time.perf_counter()
        demo = subprocess.run(
            args=[sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        elapsed_time = time.perf_counter() - start_time
        demo_passed = demo.returncode == 0
        results.append((demo_label, demo_passed, elapsed_time))
        if demo_passed:
            manage_log.log_action(
                title=demo_label,
                outcome=manage_log.ActionOutcome.SUCCESS,
                notes={"elapsed": f"{elapsed_time:.2f}s"},
            )
        else:
            demo_output = (demo.stdout + demo.stderr).strip()
            manage_log.log_action(
                title=demo_label,
                outcome=manage_log.ActionOutcome.FAILURE,
                notes={
                    "elapsed": f"{elapsed_time:.2f}s",
                    "demo_output": demo_output[:500] if demo_output else "(no demo_output)",
                },
            )
    ## print summary and exit with non-zero code if any script failed
    num_demos_passed = sum(1 for (_, _demo_passed, _) in results if _demo_passed)
    total_demos = len(results)
    total_elapsed_time = sum(_elapsed_time for (_, _, _elapsed_time) in results)
    manage_log.log_summary(
        title="Demo Results",
        notes={
            _demo_label: f"{'pass' if _demo_passed else 'FAIL'} ({_elapsed_time:.2f}s)"
            for (_demo_label, _demo_passed, _elapsed_time) in results
        },
        message=f"{num_demos_passed}/{total_demos} scripts passed in {total_elapsed_time:.2f}s.",
    )
    if num_demos_passed < total_demos:
        sys.exit(1)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
