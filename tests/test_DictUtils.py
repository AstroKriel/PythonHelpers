## ###############################################################
## DEPENDENCIES
## ###############################################################
import unittest
from Loki.WWCollections import DictUtils


## ###############################################################
## TESTS
## ###############################################################
class TestDictUtils(unittest.TestCase):
  def test_mergeDicts(self):
    ## usual case 1: merging two dictionaries with a mix of nested and simple keys
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    dict_merged = DictUtils.mergeDicts(dict1, dict2)
    dict_expected = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    self.assertEqual(dict_merged, dict_expected)
    self.assertEqual(dict1, {"a": 1, "b": {"c": 2}})
    self.assertEqual(dict2, {"b": {"d": 3}, "e": 4})
    ## usual case 2: merging two dictionaries with lists in the same key
    dict_with_list = {"a": [1, 2], "b": 3}
    dict_to_merge = {"a": [3, 4], "c": 5}
    merged_lists = DictUtils.mergeDicts(dict_with_list, dict_to_merge)
    expected_lists = {"a": [1, 2, 3, 4], "b": 3, "c": 5}
    self.assertEqual(merged_lists, expected_lists)
    ## edge case 1: merging two empty dictionaries
    dict1_empty = {}
    dict2_empty = {}
    dict_merged = DictUtils.mergeDicts(dict1_empty, dict2_empty)
    self.assertEqual(dict_merged, {})
    ## edge case 2: merging a dictionary with values and an empty dictionary
    dict1_empty_value = {"a": 1}
    dict2_empty_value = {}
    dict_merged = DictUtils.mergeDicts(dict1_empty_value, dict2_empty_value)
    self.assertEqual(dict_merged, {"a": 1})

  def test_mergeSimpleDicts(self):
    ## typical case: merging two dictionaries with simple key-value pairs
    dict_1 = {"a": 1, "b": 2}
    dict_2 = {"b": 3, "c": 4}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": 1, "b": 3, "c": 4}
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeEmptyDicts(self):
    ## typical case 1: merging an empty dictionary with a non-empty one
    dict_1 = {}
    dict_2 = {"a": 1, "b": 2}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": 1, "b": 2}
    self.assertEqual(dict_merged, dict_expected)
    ## typical case 2: merging a non-empty dictionary with an empty one
    dict_1 = {"a": 1}
    dict_2 = {}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": 1}
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeNestedDicts(self):
    ## typical case: merging two dictionaries with nested dictionaries
    dict_1 = {"a": {"x": 1}, "b": 2}
    dict_2 = {"a": {"y": 2}, "c": 3}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": {"x": 1, "y": 2}, "b": 2, "c": 3}
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeLists(self):
    ## typical case: merging two dictionaries with lists in the same key
    dict_1 = {"a": [1, 2], "b": 3}
    dict_2 = {"a": [3, 4], "c": 5}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": [1, 2, 3, 4], "b": 3, "c": 5}
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeSets(self):
    ## typical case: merging two dictionaries with sets in the same key
    dict_1 = {"a": {1, 2}, "b": 3}
    dict_2 = {"a": {3, 4}, "c": 5}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": {1, 2, 3, 4}, "b": 3, "c": 5}
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeWithSimpleTypes(self):
    ## typical case: merging two dictionaries with simple data types (integer and string)
    dict_1 = {"a": 1, "b": "hello"}
    dict_2 = {"a": 2, "b": "world", "c": 3}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": 2, "b": "world", "c": 3}
    self.assertEqual(dict_merged, dict_expected)

  def test_noSideEffectsOnOriginals(self):
    ## edge case: check that modifying the dict_merged dictionary does not affect the originals
    dict_1 = {"a": [1, 2]}
    dict_2 = {"a": [3, 4], "b": 5}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    ## modify dict_merged dict and check originals are unchanged
    dict_merged["a"].append(5)
    self.assertEqual(dict_1, {"a": [1, 2]})
    self.assertEqual(dict_2, {"a": [3, 4], "b": 5})

  def test_mergeComplexStructures(self):
    ## typical case: merging two complex dictionaries with mixed structures (lists, sets, and nested dictionaries)
    dict_1 = {
        "a": [1, 2],
        "b": {"x": 1},
        "c": {1, 2}
    }
    dict_2 = {
        "a": [3, 4],
        "b": {"y": 2},
        "c": {3, 4},
        "d": "new"
    }
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {
        "a": [1, 2, 3, 4],
        "b": {"x": 1, "y": 2},
        "c": {1, 2, 3, 4},
        "d": "new"
    }
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeKeyConflict(self):
    ## edge case: merging a dictionary with a list with a value in the other dictionary that is not a list
    dict_1 = {"a": [1, 2]}
    dict_2 = {"a": 3}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": 3}
    self.assertEqual(dict_merged, dict_expected)

  def test_mergeNoneValues(self):
    ## typical case: merging a dictionary with `None` values
    dict_1 = {"a": None, "b": 2}
    dict_2 = {"a": 1, "c": 3}
    dict_merged = DictUtils.mergeDicts(dict_1, dict_2)
    dict_expected = {"a": 1, "b": 2, "c": 3}
    self.assertEqual(dict_merged, dict_expected)

  def test_filterDict2ExcludeKeys(self):
    ## typical case: exclude key without affecting input dict
    dict_in = {"a": 1, "b": 2, "c": 3}
    filtered = DictUtils.filterDict2ExcludeKeys(dict_in, ["b"])
    dict_expected = {"a": 1, "c": 3}
    self.assertEqual(filtered, dict_expected)
    self.assertEqual(dict_in, {"a": 1, "b": 2, "c": 3})
    ## edge case 1: key does not exist
    filtered_empty = DictUtils.filterDict2ExcludeKeys({}, ["b"])
    self.assertEqual(filtered_empty, {})
    ## edge case 2: no keys to exclude
    filtered_none_exclude = DictUtils.filterDict2ExcludeKeys(dict_in, [])
    self.assertEqual(filtered_none_exclude, dict_in)
    ## edge case 3: all keys are excluded
    filtered_all_exclude = DictUtils.filterDict2ExcludeKeys(dict_in, ["a", "b", "c"])
    self.assertEqual(filtered_all_exclude, {})
    ## edge case 4: nested key is excluded
    dict_with_nested = {"a": 1, "b": {"c": 2}}
    filtered_nested = DictUtils.filterDict2ExcludeKeys(dict_with_nested, ["b"])
    self.assertEqual(filtered_nested, {"a": 1})

  def test_checkIfDictsAreDifferent(self):
    ## typical case 1: identical dictionaries
    dict1 = {"a": 1, "b": 2}
    dict2 = {"a": 1, "b": 2}
    self.assertFalse(DictUtils.checkIfDictsAreDifferent(dict1, dict2))
    ## typical case 2: different dictionaries
    dict3 = {"a": 1, "b": 3}
    dict4 = {"a": 1}
    self.assertTrue(DictUtils.checkIfDictsAreDifferent(dict1, dict3))
    self.assertTrue(DictUtils.checkIfDictsAreDifferent(dict1, dict4))
    ## edge case 1: comparing two empty dictionaries
    dict_empty1 = {}
    dict_empty2 = {}
    self.assertFalse(DictUtils.checkIfDictsAreDifferent(dict_empty1, dict_empty2))
    ## edge case 2: comparing an empty dictionary with a non-empty one
    dict_empty_non_empty = {}
    dict_non_empty = {"a": 1}
    self.assertTrue(DictUtils.checkIfDictsAreDifferent(dict_empty_non_empty, dict_non_empty))
    ## edge case 3: comparing dictionaries with lists in the same key
    dict_with_list1 = {"a": [1, 2], "b": 3}
    dict_with_list2 = {"a": [1, 2], "b": 4}
    self.assertTrue(DictUtils.checkIfDictsAreDifferent(dict_with_list1, dict_with_list2))
    ## edge case 4: comparing nested dictionaries
    dict_with_nested1 = {"a": 1, "b": {"c": 2}}
    dict_with_nested2 = {"a": 1, "b": {"c": 3}}
    self.assertTrue(DictUtils.checkIfDictsAreDifferent(dict_with_nested1, dict_with_nested2))
    ## edge case 5: comparing identical nested dictionaries
    dict_with_nested_same = {"a": 1, "b": {"c": 2}}
    self.assertFalse(DictUtils.checkIfDictsAreDifferent(dict_with_nested1, dict_with_nested_same))


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  unittest.main()


## END OF TEST