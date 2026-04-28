## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_io import manage_log

##
## === DEMOS
##


def demo_lines() -> None:
    manage_log.log_section(
        title="Demo: line helpers",
        add_spacing=True,
    )
    manage_log.log_task(
        text="Prepare environment",
        show_time=True,
    )
    manage_log.log_note(
        text="Using cache at /gdata/user/cache",
        show_time=True,
    )
    manage_log.log_hint(text="This may take a while")
    manage_log.log_alert(text="Running with default settings")
    manage_log.log_debug(text="rank=0 seed=42")
    manage_log.log_outcome(
        text="Initialized MPI",
        outcome=manage_log.ActionOutcome.SUCCESS,
    )
    manage_log.log_outcome(
        text="Optional step skipped",
        outcome=manage_log.ActionOutcome.SKIPPED,
    )
    manage_log.log_outcome(
        text="Post-check failed",
        outcome=manage_log.ActionOutcome.FAILURE,
    )
    manage_log.log_task(text="printing 2 empty lines...")
    manage_log.log_empty_lines(lines=2)
    manage_log.log_task(
        text="^there are two empty lines above^",
        show_time=False,
    )
    manage_log.log_empty_lines()


def demo_blocks() -> None:
    manage_log.log_section(
        title="Demo: block helpers",
        add_spacing=True,
    )
    manage_log.log_action(
        title="Copy File",
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="File copied successfully.",
        notes={
            "File": "orszag_tang.in",
            "From": "/Users/necoturb/Documents/Codes/quokka/inputs",
            "To": "/made/up/address/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8",
        },
    )
    manage_log.log_action(
        title="Create PBS Job",
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="Submit with: qsub /path/to/job.sh",
        notes={
            "Script": "this/is/a/really/long/path/to/the/simulation/job.sh",
            "Tagname": "OrszagTang_cfl0.3_rk2_ro2_ld04_N64_Nbo32_Nbl32_bopr1_mpir8",
            "CPUs": 8,
            "Memory": "32 GB",
            "Walltime": "02:00:00",
        },
    )
    manage_log.log_action(
        title="Create Directory",
        outcome=manage_log.ActionOutcome.SKIPPED,
        message="Directory already exists; nothing to do.",
        notes={"Path": "/tmp/sim/OT_N32"},
    )
    manage_log.log_warning(
        text="Existing file was overwritten.",
        notes={
            "Path": "/tmp/sim/OT_N64/sim_params.json",
            "Format": "json",
        },
    )
    manage_log.log_error(
        text="Command not found on PATH.",
        notes={"Command": "qstat -f"},
    )
    manage_log.log_action(
        title="Run Simulation",
        outcome=manage_log.ActionOutcome.FAILURE,
        message="One or more checks failed (dry-run validation).",
        notes={
            "Executable": "/path/to/test_orszag_tang",
            "Args": "--dry-run",
        },
        message_position="top",
    )
    manage_log.log_context(
        title="System Info",
        message="Environment detected.",
        notes={
            "OS": "macOS 14.5",
            "Python": "3.12.2",
            "Rich": "13.7.1",
        },
        message_position="bottom",
    )
    manage_log.log_context(
        title="Usage Hint",
        message="Run with --help to list all CLI options.",
    )
    manage_log.log_context(
        title="Configuration Notice",
        message="Using default settings; performance may be suboptimal.",
    )
    manage_log.log_items(
        title="Available Datasets",
        items=[
            "orszag_tang_N64",
            "mhd_turbulence_Re200",
            "galaxy_merger_highPm",
        ],
        message="A few datasets are available.",
        message_position="top",
    )
    manage_log.log_summary(
        title="Run Summary",
        notes={
            "jobs_submitted": 3,
            "jobs_skipped": 1,
            "failures": 1,
            "total_walltime_h": 5.3,
        },
        message="Key metrics for this session.",
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    demo_lines()
    demo_blocks()
    manage_log.log_note(
        text="finished!",
        show_time=True,
    )


if __name__ == "__main__":
    main()

## } MODULE
