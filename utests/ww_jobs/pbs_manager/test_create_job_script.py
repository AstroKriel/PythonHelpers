## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import tempfile
import unittest
from pathlib import Path

## local
from jormi.ww_jobs.pbs_manager import _create_job_script

##
## === TEST SUITE
##


def _make_minimal_kwargs(
    *,
    directory: str | Path,
) -> dict:
    return {
        "system_name": "gadi",
        "directory": directory,
        "file_name": "test_job.sh",
        "main_command": "python run.py",
        "tag_name": "my_job",
        "queue_name": "normal",
        "compute_group_name": "jh2",
        "num_procs": 48,
        "memory_gb": 192,
        "wall_time_hours": 2,
        "verbose": False,
    }


class TestCreateJobScript_FileCreation(unittest.TestCase):

    def test_creates_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            self.assertTrue(file_path.is_file())

    def test_returns_path(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            self.assertIsInstance(file_path, Path)

    def test_rejects_non_sh_extension(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    system_name="gadi",
                    directory=tmp_dir,
                    file_name="test_job.txt",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="normal",
                    compute_group_name="jh2",
                    num_procs=48,
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )


class TestCreateJobScript_Header(unittest.TestCase):

    def test_pbs_directives_present(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertIn("#!/bin/bash", content)
            self.assertIn("#PBS -P jh2", content)
            self.assertIn("#PBS -q normal", content)
            self.assertIn("#PBS -l ncpus=48", content)
            self.assertIn("#PBS -l mem=192GB", content)
            self.assertIn("#PBS -l walltime=02:00:00", content)
            self.assertIn("#PBS -N my_job", content)

    def test_storage_defaults_to_compute_group(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertIn("scratch/jh2", content)
            self.assertIn("gdata/jh2", content)

    def test_custom_storage_group(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                system_name="gadi",
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="normal",
                compute_group_name="jh2",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                storage_group_name="ek9",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn("scratch/ek9", content)
            self.assertIn("gdata/ek9", content)

    def test_no_email_directives_when_none(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertNotIn("#PBS -m", content)
            self.assertNotIn("#PBS -M", content)

    def test_email_directives_present_when_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                system_name="gadi",
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="normal",
                compute_group_name="jh2",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                email_address="user@example.com",
                email_on_start=True,
                email_on_finish=True,
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn("#PBS -M user@example.com", content)
            self.assertIn("a", content)  # always-fail flag
            self.assertIn("b", content)  # begin flag
            self.assertIn("e", content)  # end flag

    def test_email_fail_only_when_no_start_or_finish(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                system_name="gadi",
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="normal",
                compute_group_name="jh2",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                email_address="user@example.com",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn("#PBS -m a\n", content)


class TestCreateJobScript_Commands(unittest.TestCase):

    def test_main_command_present(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertIn("python run.py", content)

    def test_prep_command_present_when_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                system_name="gadi",
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="normal",
                compute_group_name="jh2",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                prep_command="module load python",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn("module load python", content)

    def test_post_command_present_when_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                system_name="gadi",
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="normal",
                compute_group_name="jh2",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                post_command="echo done",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn("echo done", content)

    def test_post_command_conditional_on_main_success(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                system_name="gadi",
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="normal",
                compute_group_name="jh2",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                post_command="echo done",
                always_run_post=False,
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn('if [ "$main_command_exit_code" -eq 0 ]', content)

    def test_no_prep_command_when_not_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertNotIn("preparation step", content)

    def test_exit_code_propagation(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertIn('exit "$main_command_exit_code"', content)

    def test_log_file_uses_tag_name(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertIn('LOG_FILE="my_job.out"', content)


class TestCreateJobScript_InvalidParams(unittest.TestCase):

    def test_invalid_system_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    system_name="unknown",
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="normal",
                    compute_group_name="jh2",
                    num_procs=48,
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )

    def test_invalid_queue_group_raises(
        self,
    ):
        ## mk27 cannot use normal queue
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    system_name="gadi",
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="normal",
                    compute_group_name="mk27",
                    num_procs=48,
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )

    def test_wall_time_exceeds_limit_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    system_name="gadi",
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="normal",
                    compute_group_name="jh2",
                    num_procs=48,
                    memory_gb=192,
                    wall_time_hours=999,
                    verbose=False,
                )

    def test_zero_procs_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    system_name="gadi",
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="normal",
                    compute_group_name="jh2",
                    num_procs=0,  # type: ignore
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
