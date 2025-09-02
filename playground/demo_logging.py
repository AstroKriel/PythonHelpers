from jormi.ww_io import log_manager


def demo_header() -> None:
    log_manager.render_line(
        log_manager.Message(
            "Something is happening",
            message_type=log_manager.MessageType.DETAILS,
        ),
        show_time=True,
    )
    log_manager.render_line(
        log_manager.Message(
            "Operation succeeded",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.SUCCESS,
        ),
    )
    log_manager.render_line(
        log_manager.Message(
            "This may take a while",
            message_type=log_manager.MessageType.HINT,
        ),
    )
    log_manager.render_line(
        log_manager.Message(
            "List item example",
            message_type=log_manager.MessageType.LIST,
        ),
    )
    log_manager.render_line(
        log_manager.Message(
            "Step was skipped",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.SKIPPED,
        ),
    )
    log_manager.render_line(
        log_manager.Message(
            "Debug info: x=42",
            message_type=log_manager.MessageType.DEBUG,
        ),
    )
    log_manager.render_line(
        log_manager.Message(
            "An error occurred",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.ERROR,
        ),
        add_spacing=True,
    )


def demo_blocks() -> None:
    log_manager.render_block(
        log_manager.Message(
            message_title="Copy File",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.SUCCESS,
            message="File copied successfully.",
            message_notes={
                "File": "orszag_tang.in",
                "From": "/Users/necoturb/Documents/Codes/quokka/inputs",
                "To": "/made/up/address/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8",
            },
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Create PBS Job",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.SUCCESS,
            message="Submit with: qsub /path/to/job.sh",
            message_notes={
                "Script": "this/is/a/really/long/path/to/the/simulation/job.sh",
                "Tagname": "OrszagTang_cfl0.3_rk2_ro2_ld04_N64_Nbo32_Nbl32_bopr1_mpir8",
                "CPUs": 8,
                "Memory": "32 GB",
                "Walltime": "02:00:00",
            },
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Create Directory",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.SKIPPED,
            message="Directory already exists; nothing to do.",
            message_notes={"Path": "/tmp/sim/OT_N32"},
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Save File",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.WARNING,
            message="Existing file was overwritten.",
            message_notes={
                "Path": "/tmp/sim/OT_N64/sim_params.json",
                "Format": "json",
            },
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Probe PBS Queue",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.ERROR,
            message="Command not found on PATH.",
            message_notes={"Command": "qstat -f"},
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Run Simulation",
            message_type=log_manager.MessageType.ACTION,
            action_outcome=log_manager.ActionOutcome.FAILURE,
            message="One or more checks failed (dry-run validation).",
            message_notes={
                "Executable": "/path/to/test_orszag_tang",
                "Args": "--dry-run",
            },
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="System Info",
            message_type=log_manager.MessageType.DETAILS,
            message="Environment detected.",
            message_notes={
                "OS": "macOS 14.5",
                "Python": "3.12.2",
                "Rich": "13.7.1",
            },
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Usage Hint",
            message_type=log_manager.MessageType.HINT,
            message="Run with --help to list all CLI options.",
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Configuration Notice",
            message_type=log_manager.MessageType.ALERT,
            message="Using default settings; performance may be suboptimal.",
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Available Datasets",
            message_type=log_manager.MessageType.LIST,
            message="A few datasets are available.",
            message_notes={
                "Dataset A": "orszag_tang_N64",
                "Dataset B": "mhd_turbulence_Re200",
                "Dataset C": "galaxy_merger_highPm",
            },
        ),
    )

    log_manager.render_block(
        log_manager.Message(
            message_title="Debug State",
            message_type=log_manager.MessageType.DEBUG,
            message="Internal diagnostics for development.",
            message_notes={
                "Seed": 12345,
                "Params": {
                    "cfl": 0.3,
                    "rk": 2,
                },
                "Cache": "enabled",
            },
        ),
    )


def main() -> None:
    demo_header()
    demo_blocks()
    log_manager.render_line(log_manager.Message("finished!", message_type=log_manager.MessageType.DETAILS))


if __name__ == "__main__":
    main()
