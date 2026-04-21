## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from jormi.ww_jobs.slurm_manager import _job_validation

##
## === TEST SUITE
##


class TestJobValidation_ValidParams(unittest.TestCase):

    def test_valid_cita_all(
        self,
    ):
        _job_validation.validate_job_params(
            system_name="cita",
            partition_name="all",
            num_cpus=4,
            wall_time_hours=24,
        )

    def test_wall_time_at_limit(
        self,
    ):
        _job_validation.validate_job_params(
            system_name="cita",
            partition_name="all",
            num_cpus=1,
            wall_time_hours=72,
        )


class TestJobValidation_InvalidSystem(unittest.TestCase):

    def test_unknown_system_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                system_name="unknown_system",
                partition_name="all",
                num_cpus=4,
                wall_time_hours=24,
            )


class TestJobValidation_InvalidPartition(unittest.TestCase):

    def test_unknown_partition_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                system_name="cita",
                partition_name="unknown_partition",
                num_cpus=4,
                wall_time_hours=24,
            )


class TestJobValidation_WallTimeLimits(unittest.TestCase):

    def test_exceeding_wall_time_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                system_name="cita",
                partition_name="all",
                num_cpus=4,
                wall_time_hours=73,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
