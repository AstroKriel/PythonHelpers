from jormi.utils import logging_utils

Message       = logging_utils.Message
MessageType   = logging_utils.MessageType
ActionOutcome = logging_utils.ActionOutcome
render_line   = logging_utils.render_line
render_block  = logging_utils.render_block


def demo_header() -> None:
  render_line(Message("Something is happening", message_type=MessageType.GENERAL), show_time=True)
  render_line(Message("Operation succeeded",    message_type=MessageType.ACTION, action_outcome=ActionOutcome.SUCCESS))
  render_line(Message("This may take a while",  message_type=MessageType.HINT))
  render_line(Message("List item example",      message_type=MessageType.LIST))
  render_line(Message("Step was skipped",       message_type=MessageType.ACTION, action_outcome=ActionOutcome.SKIPPED))
  render_line(Message("Debug info: x=42",       message_type=MessageType.DEBUG))
  render_line(Message("An error occurred",      message_type=MessageType.ACTION, action_outcome=ActionOutcome.ERROR), add_spacing=True)


def demo_blocks() -> None:
  render_block(Message(
    message_title  = "Copy File",
    message_type   = MessageType.ACTION,
    action_outcome = ActionOutcome.SUCCESS,
    message        = "File copied successfully.",
    message_notes  = {
      "File": "orszag_tang.in",
      "From": "/Users/necoturb/Documents/Codes/quokka/inputs",
      "To":   "/made/up/address/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8",
    },
  ))

  render_block(Message(
    message_title  = "Create PBS Job",
    message_type   = MessageType.ACTION,
    action_outcome = ActionOutcome.SUCCESS,
    message        = "Submit with: qsub /path/to/job.sh",
    message_notes  = {
      "Script":   "this/is/a/really/long/path/to/the/simulation/job.sh",
      "Tagname":  "OrszagTang_cfl0.3_rk2_ro2_ld04_N64_Nbo32_Nbl32_bopr1_mpir8",
      "CPUs":     8,
      "Memory":   "32 GB",
      "Walltime": "02:00:00",
    },
  ))

  render_block(Message(
    message_title  = "Create Directory",
    message_type   = MessageType.ACTION,
    action_outcome = ActionOutcome.SKIPPED,
    message        = "Directory already exists; nothing to do.",
    message_notes  = {"Path": "/tmp/sim/OT_N32"},
  ))

  render_block(Message(
    message_title  = "Save File",
    message_type   = MessageType.ACTION,
    action_outcome = ActionOutcome.WARNING,
    message        = "Existing file was overwritten.",
    message_notes  = {
      "Path":   "/tmp/sim/OT_N64/sim_params.json",
      "Format": "json",
    },
  ))

  render_block(Message(
    message_title  = "Probe PBS Queue",
    message_type   = MessageType.ACTION,
    action_outcome = ActionOutcome.ERROR,
    message        = "Command not found on PATH.",
    message_notes  = {"Command": "qstat -f"},
  ))

  render_block(Message(
    message_title  = "Run Simulation",
    message_type   = MessageType.ACTION,
    action_outcome = ActionOutcome.FAILURE,
    message        = "One or more checks failed (dry-run validation).",
    message_notes  = {
      "Executable": "/path/to/test_orszag_tang",
      "Args":       "--dry-run",
    },
  ))

  render_block(Message(
    message_title = "System Info",
    message_type  = MessageType.GENERAL,
    message       = "Environment detected.",
    message_notes = {
      "OS":     "macOS 14.5",
      "Python": "3.12.2",
      "Rich":   "13.7.1",
    },
  ))

  render_block(Message(
    message_title = "Usage Hint",
    message_type  = MessageType.HINT,
    message       = "Run with --help to list all CLI options.",
  ))

  render_block(Message(
    message_title = "Configuration Notice",
    message_type  = MessageType.ALERT,
    message       = "Using default settings; performance may be suboptimal.",
  ))

  render_block(Message(
    message_title = "Available Datasets",
    message_type  = MessageType.LIST,
    message       = "A few datasets are available.",
    message_notes = {
      "Dataset A": "orszag_tang_N64",
      "Dataset B": "mhd_turbulence_Re200",
      "Dataset C": "galaxy_merger_highPm",
    },
  ))

  render_block(Message(
    message_title = "Debug State",
    message_type  = MessageType.DEBUG,
    message       = "Internal diagnostics for development.",
    message_notes = {
      "Seed":   12345,
      "Params": {"cfl": 0.3, "rk": 2},
      "Cache":  "enabled",
    },
  ))


def main() -> None:
  demo_header()
  demo_blocks()
  render_line(Message("finished!", message_type=MessageType.GENERAL))


if __name__ == "__main__":
  main()
