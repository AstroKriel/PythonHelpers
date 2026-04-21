## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from jormi.ww_jobs.pbs_manager import _job_validation

##
## === TEST SUITE
##


class TestJobValidation_ValidParams(unittest.TestCase):

    def test_valid_gadi_normal_jh2(
        self,
    ):
        _job_validation.validate_job_params(
            "gadi",
            "normal",
            "jh2",
            48,
            24,
        )

    def test_valid_gadi_rsaa_mk27(
        self,
    ):
        _job_validation.validate_job_params(
            "gadi",
            "rsaa",
            "mk27",
            104,
            48,
        )

    def test_single_node_any_cpu_count(
        self,
    ):
        _job_validation.validate_job_params(
            "gadi",
            "normal",
            "jh2",
            1,
            1,
        )

    def test_multi_node_requires_full_nodes(
        self,
    ):
        ## 96 cpus = 2 * 48 (exact multiple, must pass)
        _job_validation.validate_job_params(
            "gadi",
            "normal",
            "jh2",
            96,
            24,
        )


class TestJobValidation_InvalidSystem(unittest.TestCase):

    def test_unknown_system_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "unknown_system",
                "normal",
                "jh2",
                48,
                24,
            )


class TestJobValidation_InvalidQueue(unittest.TestCase):

    def test_unknown_queue_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "unknown_queue",
                "jh2",
                48,
                24,
            )


class TestJobValidation_InvalidComputeGroup(unittest.TestCase):

    def test_unknown_compute_group_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "normal",
                "unknown_group",
                48,
                24,
            )

    def test_queue_not_valid_for_group_raises(
        self,
    ):
        ## mk27 only has access to rsaa, not normal
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "normal",
                "mk27",
                48,
                24,
            )


class TestJobValidation_CpuRules(unittest.TestCase):

    def test_non_multiple_cpus_above_node_raises(
        self,
    ):
        ## 50 > 48 (one node) but 50 % 48 != 0
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "normal",
                "jh2",
                50,
                24,
            )

    def test_exceeding_max_cpus_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "normal",
                "jh2",
                20737,
                5,
            )


class TestJobValidation_WallTimeLimits(unittest.TestCase):

    def test_exceeding_wall_time_for_cpu_tier_raises(
        self,
    ):
        ## <= 672 cpus: max 48h; requesting 49h should fail
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "normal",
                "jh2",
                48,
                49,
            )

    def test_wall_time_at_limit_passes(
        self,
    ):
        _job_validation.validate_job_params(
            "gadi",
            "normal",
            "jh2",
            48,
            48,
        )

    def test_rsaa_wall_time_at_limit_passes(
        self,
    ):
        _job_validation.validate_job_params(
            "gadi",
            "rsaa",
            "jh2",
            104,
            48,
        )

    def test_rsaa_exceeding_wall_time_raises(
        self,
    ):
        with self.assertRaises(_job_validation.QueueValidationError):
            _job_validation.validate_job_params(
                "gadi",
                "rsaa",
                "jh2",
                104,
                49,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
