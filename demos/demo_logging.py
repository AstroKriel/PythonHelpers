from jormi.ww_io import log_manager as lm


def demo_lines() -> None:
    lm.log_section("Demo: line helpers", add_spacing=True)
    lm.log_task("Prepare environment", show_time=True)
    lm.log_note("Using cache at /gdata/user/cache", show_time=True)
    lm.log_hint("This may take a while")
    lm.log_alert("Running with default settings")
    lm.log_debug("rank=0 seed=42")
    lm.log_outcome("Initialized MPI", outcome=lm.ActionOutcome.SUCCESS)
    lm.log_outcome("Optional step skipped", outcome=lm.ActionOutcome.SKIPPED)
    lm.log_outcome("Post-check failed", outcome=lm.ActionOutcome.FAILURE)
    lm.log_task("printing 2 empty lines...")
    lm.log_empty_lines(lines=2)
    lm.log_task("^there are two empty lines above^", show_time=False)
    lm.log_empty_lines()


def demo_blocks() -> None:
    lm.log_section("Demo: block helpers", add_spacing=True)
    lm.log_action(
        title="Copy File",
        succeeded=True,
        message="File copied successfully.",
        notes={
            "File": "orszag_tang.in",
            "From": "/Users/necoturb/Documents/Codes/quokka/inputs",
            "To": "/made/up/address/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8",
        },
    )
    lm.log_action(
        title="Create PBS Job",
        succeeded=True,
        message="Submit with: qsub /path/to/job.sh",
        notes={
            "Script": "this/is/a/really/long/path/to/the/simulation/job.sh",
            "Tagname": "OrszagTang_cfl0.3_rk2_ro2_ld04_N64_Nbo32_Nbl32_bopr1_mpir8",
            "CPUs": 8,
            "Memory": "32 GB",
            "Walltime": "02:00:00",
        },
    )
    lm.log_action(
        title="Create Directory",
        succeeded=None,
        message="Directory already exists; nothing to do.",
        notes={"Path": "/tmp/sim/OT_N32"},
    )
    lm.log_warning(
        "Existing file was overwritten.",
        notes={
            "Path": "/tmp/sim/OT_N64/sim_params.json",
            "Format": "json",
        },
    )
    lm.log_error(
        "Command not found on PATH.",
        notes={"Command": "qstat -f"},
    )
    lm.log_action(
        title="Run Simulation",
        succeeded=False,
        message="One or more checks failed (dry-run validation).",
        notes={
            "Executable": "/path/to/test_orszag_tang",
            "Args": "--dry-run",
        },
        message_position="top",
    )
    lm.log_context(
        title="System Info",
        message="Environment detected.",
        notes={
            "OS": "macOS 14.5",
            "Python": "3.12.2",
            "Rich": "13.7.1",
        },
        message_position="bottom",
    )
    lm.log_context(
        title="Usage Hint",
        message="Run with --help to list all CLI options.",
    )
    lm.log_context(
        title="Configuration Notice",
        message="Using default settings; performance may be suboptimal.",
    )
    lm.log_items(
        title="Available Datasets",
        items=[
            "orszag_tang_N64",
            "mhd_turbulence_Re200",
            "galaxy_merger_highPm",
        ],
        message="A few datasets are available.",
        message_position="top",
    )
    lm.log_summary(
        title="Run Summary",
        notes={
            "jobs_submitted": 3,
            "jobs_skipped": 1,
            "failures": 1,
            "total_walltime_h": 5.3,
        },
        message="Key metrics for this session.",
    )


def demo_renderers() -> None:
    lm.log_section("Demo: raw renderers (internal)")
    lm.render_line(
        lm.Message("raw line (NOTE)", message_type=lm.MessageType.NOTE),
        show_time=True,
        add_spacing=True,
    )
    lm.render_block(
        lm.Message(
            message_title="Raw block (NOTE)",
            message_type=lm.MessageType.NOTE,
            message="Called via render_block directly.",
            message_notes={
                "why": "demonstration",
                "prefer": "log helpers",
            },
        ),
        show_time=True,
    )
    lm.log_empty_lines()


def main() -> None:
    demo_lines()
    demo_blocks()
    demo_renderers()
    lm.log_note("finished!", show_time=True)


if __name__ == "__main__":
    main()
