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
## === HELPERS
##


def _make_minimal_kwargs(
    *,
    directory: str | Path,
) -> dict:
    return {
        "directory": directory,
        "file_name": "test_job.sh",
        "main_command": "python run.py",
        "tag_name": "my_job",
        "queue_name": "queue_a",
        "num_procs": 48,
        "memory_gb": 192,
        "wall_time_hours": 2,
        "verbose": False,
    }


def _make_directives() -> list[str]:
    return [
        "#!/bin/bash -l",
        "#PBS -l nodes=1:ppn=16",
        "#PBS -l walltime=04:00:00",
        "#PBS -r n",
        "#PBS -j oe",
        "#PBS -q queue_b",
        "#PBS -N my_job",
    ]


##
## === TEST SUITE
##


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
                    directory=tmp_dir,
                    file_name="test_job.txt",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="queue_a",
                    num_procs=48,
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )


class TestCreateJobScript_Header(unittest.TestCase):

    def test_generic_pbs_directives_present(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                **_make_minimal_kwargs(directory=tmp_dir),
            )
            content = file_path.read_text()
            self.assertIn("#!/bin/bash", content)
            self.assertIn("#PBS -q queue_a", content)
            self.assertIn("#PBS -l ncpus=48", content)
            self.assertIn("#PBS -l mem=192GB", content)
            self.assertIn("#PBS -l walltime=02:00:00", content)
            self.assertIn("#PBS -N my_job", content)

    def test_custom_directives_are_written_verbatim(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_pbs_job_script(
                directory=tmp_dir,
                file_name="test_job.sh",
                directives=_make_directives(),
                main_command="python run.py",
                tag_name="my_job",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn("#!/bin/bash -l", content)
            self.assertIn("#PBS -l nodes=1:ppn=16", content)
            self.assertIn("#PBS -q queue_b", content)

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
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="queue_a",
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
            self.assertIn("#PBS -m abe", content)


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
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="queue_a",
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
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="queue_a",
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
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                queue_name="queue_a",
                num_procs=48,
                memory_gb=192,
                wall_time_hours=2,
                post_command="echo done",
                always_run_post=False,
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn('if [ "$main_command_exit_code" -eq 0 ]', content)


class TestCreateJobScript_InvalidParams(unittest.TestCase):

    def test_empty_tag_name_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="",  # type: ignore
                    queue_name="queue_a",
                    num_procs=48,
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )

    def test_zero_procs_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    queue_name="queue_a",
                    num_procs=0,  # type: ignore
                    memory_gb=192,
                    wall_time_hours=2,
                    verbose=False,
                )

    def test_missing_queue_raises_without_directives(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_pbs_job_script(
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    num_procs=48,
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
