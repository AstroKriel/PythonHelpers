## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

## local
from jormi.ww_io import manage_io

##
## === TEST SUITE
##


class TestFilterDirectory_Validation(unittest.TestCase):

    def test_nonexistent_directory_raises(
        self,
    ) -> None:
        with self.assertRaises(NotADirectoryError):
            manage_io.filter_directory(
                directory=Path("/nonexistent/path/that/does/not/exist"),
            )

    def test_no_include_types_raises(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                manage_io.filter_directory(
                    directory=Path(tmp),
                    include_files=False,
                    include_folders=False,
                )


class TestFilterDirectory_IncludeExclude(unittest.TestCase):

    def setUp(
        self,
    ) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        ## create files and folders
        (self.tmp_dir / "Mach2_Re1500_Pm1_Nres576").mkdir()
        (self.tmp_dir / "Mach4_Re1500_Pm1_Nres576").mkdir()
        (self.tmp_dir / "Mach2_Re500_Pm4_Nres576").mkdir()
        (self.tmp_dir / "notes.txt").touch()

    def tearDown(
        self,
    ) -> None:
        self._tmp.cleanup()

    def test_no_filter_returns_all(
        self,
    ) -> None:
        results = manage_io.filter_directory(directory=self.tmp_dir)
        self.assertEqual(
            len(results),
            4,
        )

    def test_include_files_false_excludes_files(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            include_files=False,
        )
        self.assertTrue(
            all(path.is_dir() for path in results),
        )
        self.assertEqual(
            len(results),
            3,
        )

    def test_include_folders_false_excludes_folders(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            include_folders=False,
        )
        self.assertTrue(
            all(path.is_file() for path in results),
        )
        self.assertEqual(
            len(results),
            1,
        )

    def test_req_include_words_single_string(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            req_include_words="Mach2",
        )
        self.assertEqual(
            len(results),
            2,
        )
        self.assertTrue(
            all("Mach2" in path.name for path in results),
        )

    def test_req_include_words_list(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            req_include_words=["Mach2", "Re1500"],
        )
        self.assertEqual(
            len(results),
            1,
        )
        self.assertIn(
            "Mach2_Re1500_Pm1_Nres576",
            results[0].name,
        )

    def test_req_exclude_words(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            req_exclude_words="Re500",
            include_files=False,
        )
        self.assertEqual(
            len(results),
            2,
        )
        self.assertTrue(
            all("Re500" not in path.name for path in results),
        )

    def test_results_are_sorted(
        self,
    ) -> None:
        results = manage_io.filter_directory(directory=self.tmp_dir)
        self.assertEqual(
            results,
            sorted(results),
        )


class TestFilterDirectory_PrefixSuffix(unittest.TestCase):

    def setUp(
        self,
    ) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        (self.tmp_dir / "Turb_hdf5_plt_cnt_0000").touch()
        (self.tmp_dir / "Turb_hdf5_plt_cnt_0001").touch()
        (self.tmp_dir / "Turb_hdf5_plt_cnt_0002").touch()
        (self.tmp_dir / "output.log").touch()
        (self.tmp_dir / "snapshot_0001.npz").touch()

    def tearDown(
        self,
    ) -> None:
        self._tmp.cleanup()

    def test_prefix_filter(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            prefix="Turb_hdf5_plt_cnt_",
        )
        self.assertEqual(
            len(results),
            3,
        )
        self.assertTrue(
            all(path.name.startswith("Turb_hdf5_plt_cnt_") for path in results),
        )

    def test_suffix_filter(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            suffix=".npz",
        )
        self.assertEqual(
            len(results),
            1,
        )
        self.assertEqual(
            results[0].name,
            "snapshot_0001.npz",
        )

    def test_num_parts_filter(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            prefix="Turb_hdf5_plt_cnt_",
            num_parts=5,
        )
        self.assertEqual(
            len(results),
            3,
        )

    def test_num_parts_excludes_mismatches(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            num_parts=1,
        )
        self.assertEqual(
            len(results),
            1,
        )
        self.assertEqual(
            results[0].name,
            "output.log",
        )

    def test_custom_delimiter(
        self,
    ) -> None:
        results = manage_io.filter_directory(
            directory=self.tmp_dir,
            suffix=".npz",
            delimiter=".",
            num_parts=2,
        )
        self.assertEqual(
            len(results),
            1,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
