from jormi.logging import log_manager

def demo_header() -> None:
  log_manager.print_line("Something is happening", level=log_manager.Level.INFO, show_symbol=True)
  log_manager.print_line("Something is happening", level=log_manager.Level.INFO)
  log_manager.print_line("Operation succeeded",    level=log_manager.Level.SUCCESS)
  log_manager.print_line("This may take a while",  level=log_manager.Level.WARNING)
  log_manager.print_line("List item example",      level=log_manager.Level.LIST)
  log_manager.print_line("Step was skipped",       level=log_manager.Level.SKIP)
  log_manager.print_line("Debug info: x=42",       level=log_manager.Level.DEBUG)
  log_manager.print_line("An error occurred",      level=log_manager.Level.ERROR)

def demo_blocks() -> None:
  log_manager.print_block(
    log_manager.RecordAction(
      action = "Copied File",
      status = log_manager.Status.SUCCESS,
      notes  = {
        "File" : "orszag_tang.in",
        "From" : "/Users/necoturb/Documents/Codes/quokka/inputs",
        "To"   : "/made/up/address/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8",
      },
    )
  )
  log_manager.print_block(
    record = log_manager.RecordAction(
      action = "Created PBS Job",
      status = log_manager.Status.SUCCESS,
      status_message = "Submit with: qsub /path/to/job.sh",
      notes = {
        "Script"   : "this/is/a/really/long/path/to/the/simulation/job.sh",
        "Tagname"  : "OrszagTang_cfl0.3_rk2_ro2_ld04_N64_Nbo32_Nbl32_bopr1_mpir8",
        "CPUs"     : 8,
        "Memory"   : "32 GB",
        "Walltime" : "02:00:00",
      },
    )
  )
  log_manager.print_block(
    log_manager.RecordAction(
      action = "Init Directory",
      status = log_manager.Status.SKIP,
      notes = {
        "Path": "/tmp/sim/OT_N32",
      },
    )
  )
  log_manager.print_block(
    log_manager.RecordAction(
      action = "Saved File",
      status = log_manager.Status.WARNING,
      status_message = "Existing file was overwritten.",
      notes = {
        "Path"  : "/tmp/sim/OT_N64/sim_params.json",
        "Format": "json",
      },
    )
  )
  log_manager.print_block(
    log_manager.RecordAction(
      action = "Queue Probe",
      status = log_manager.Status.ERROR,
      status_message = "Command not found on PATH",
      notes = {
        "Command": "qstat -f",
      },
    )
  )
  log_manager.print_block(
    log_manager.RecordAction(
      action = "Validation Run",
      status = log_manager.Status.FAILURE,
      status_message = "One or more checks failed (dry-run validation).",
      notes = {
        "Executable": "/path/to/test_orszag_tang",
        "Args"      : "--dry-run",
      },
    )
  )

def main() -> None:
  demo_header()
  demo_blocks()

if __name__ == "__main__":
  main()
