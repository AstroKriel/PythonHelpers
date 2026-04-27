## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

## local
from jormi.ww_jobs.slurm_manager import _create_job_script

##
## === HELPERS
##


class _SlurmMinimalKwargs(TypedDict):
    directory: str | Path
    file_name: str
    main_command: str
    tag_name: str
    partition_name: str
    num_cpus: int
    memory_gb: int
    wall_time_hours: int
    verbose: bool


def _make_minimal_kwargs(
    *,
    directory: str | Path,
) -> _SlurmMinimalKwargs:
    return {
        "directory": directory,
        "file_name": "test_job.sh",
        "main_command": "python run.py",
        "tag_name": "my_job",
        "partition_name": "partition_a",
        "num_cpus": 4,
        "memory_gb": 16,
        "wall_time_hours": 2,
        "verbose": False,
    }


def _make_directives() -> list[str]:
    return [
        "#!/bin/bash -l",
        "#SBATCH --job-name=my_job",
        "#SBATCH --partition=partition_b",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=01:00:00",
        "#SBATCH --output=%x_%j.out",
        "#SBATCH --error=%x_%j.err",
    ]


##
## === TEST SUITE
##


class TestCreateJobScript_FileCreation(unittest.TestCase):

    def test_creates_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                **_make_minimal_kwargs(
                    directory=tmp_dir,
                ),
            )
            self.assertTrue(
                file_path.is_file(),
            )

    def test_returns_path(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                **_make_minimal_kwargs(
                    directory=tmp_dir,
                ),
            )
            self.assertIsInstance(
                file_path,
                Path,
            )

    def test_rejects_non_sh_extension(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_slurm_job_script(
                    directory=tmp_dir,
                    file_name="test_job.txt",
                    main_command="python run.py",
                    tag_name="my_job",
                    partition_name="partition_a",
                    num_cpus=4,
                    memory_gb=16,
                    wall_time_hours=2,
                    verbose=False,
                )


class TestCreateJobScript_Header(unittest.TestCase):

    def test_generic_slurm_directives_present(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                **_make_minimal_kwargs(
                    directory=tmp_dir,
                ),
            )
            content = file_path.read_text()
            self.assertIn(
                "#!/bin/bash",
                content,
            )
            self.assertIn(
                "#SBATCH --job-name=my_job",
                content,
            )
            self.assertIn(
                "#SBATCH --partition=partition_a",
                content,
            )
            self.assertIn(
                "#SBATCH --cpus-per-task=4",
                content,
            )
            self.assertIn(
                "#SBATCH --mem=16G",
                content,
            )
            self.assertIn(
                "#SBATCH --time=02:00:00",
                content,
            )

    def test_custom_directives_are_written_verbatim(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                directory=tmp_dir,
                file_name="test_job.sh",
                directives=_make_directives(),
                main_command="python run.py",
                tag_name="my_job",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn(
                "#!/bin/bash -l",
                content,
            )
            self.assertIn(
                "#SBATCH --partition=partition_b",
                content,
            )
            self.assertIn(
                "#SBATCH --cpus-per-task=8",
                content,
            )

    def test_no_email_directives_when_none(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                **_make_minimal_kwargs(
                    directory=tmp_dir,
                ),
            )
            content = file_path.read_text()
            self.assertNotIn(
                "--mail-user",
                content,
            )
            self.assertNotIn(
                "--mail-type",
                content,
            )

    def test_email_directives_present_when_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                partition_name="partition_a",
                num_cpus=4,
                memory_gb=16,
                wall_time_hours=2,
                email_address="user@example.com",
                email_on_start=True,
                email_on_finish=True,
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn(
                "--mail-user=user@example.com",
                content,
            )
            self.assertIn(
                "FAIL",
                content,
            )
            self.assertIn(
                "BEGIN",
                content,
            )
            self.assertIn(
                "END",
                content,
            )


class TestCreateJobScript_Commands(unittest.TestCase):

    def test_main_command_present(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                **_make_minimal_kwargs(
                    directory=tmp_dir,
                ),
            )
            content = file_path.read_text()
            self.assertIn(
                "python run.py",
                content,
            )

    def test_prep_command_present_when_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                partition_name="partition_a",
                num_cpus=4,
                memory_gb=16,
                wall_time_hours=2,
                prep_command="module load python",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn(
                "module load python",
                content,
            )

    def test_post_command_present_when_given(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                partition_name="partition_a",
                num_cpus=4,
                memory_gb=16,
                wall_time_hours=2,
                post_command="echo done",
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn(
                "echo done",
                content,
            )

    def test_post_command_conditional_on_main_success(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = _create_job_script.create_slurm_job_script(
                directory=tmp_dir,
                file_name="test_job.sh",
                main_command="python run.py",
                tag_name="my_job",
                partition_name="partition_a",
                num_cpus=4,
                memory_gb=16,
                wall_time_hours=2,
                post_command="echo done",
                always_run_post=False,
                verbose=False,
            )
            content = file_path.read_text()
            self.assertIn(
                'if [ "$main_command_exit_code" -eq 0 ]',
                content,
            )


class TestCreateJobScript_InvalidParams(unittest.TestCase):

    def test_empty_tag_name_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_slurm_job_script(
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="",  # type: ignore
                    partition_name="partition_a",
                    num_cpus=4,
                    memory_gb=16,
                    wall_time_hours=2,
                    verbose=False,
                )

    def test_zero_cpus_raises(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_slurm_job_script(
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    partition_name="partition_a",
                    num_cpus=0,  # type: ignore
                    memory_gb=16,
                    wall_time_hours=2,
                    verbose=False,
                )

    def test_missing_partition_raises_without_directives(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _create_job_script.create_slurm_job_script(
                    directory=tmp_dir,
                    file_name="test_job.sh",
                    main_command="python run.py",
                    tag_name="my_job",
                    num_cpus=4,
                    memory_gb=16,
                    wall_time_hours=2,
                    verbose=False,
                )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
