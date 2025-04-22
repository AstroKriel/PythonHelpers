## START OF MODULE

## ###############################################################
## DEPENDENCIES
## ###############################################################
import csv
from pathlib import Path
from jormi.ww_io import file_manager


## ###############################################################
## FUNCTIONS
## ###############################################################
def ensure_csv_extension(path: Path):
  if path.suffix != ".csv": raise ValueError(f"Expected .csv file, got {path}")

def read_csv_file_into_dict(
    file_path: str | Path,
    verbose: bool = True,
  ) -> dict:
  file_path = Path(file_path).resolve()
  ensure_csv_extension(file_path)
  if not file_manager.does_file_exist(file_path):
    raise FileNotFoundError(f"No csv-file found: {file_path}")
  if verbose: print(f"Reading in csv-file: {file_path}")
  data = {}
  with open(file_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for key in reader.fieldnames:
      data[key] = []
    for row in reader:
      for key, value in row.items():
        data[key].append(float(value))
  return data


def save_dict_to_csv_file(
    file_path: str | Path,
    input_dict: dict,
    overwrite: bool = False,
    verbose: bool = True,
  ):
  file_path = Path(file_path).resolve()
  ensure_csv_extension(file_path)
  lengths = [len(v) for v in input_dict.values()]
  if len(set(lengths)) != 1:
    raise ValueError("All columns in input_dict must be of equal length.")
  if file_manager.does_file_exist(file_path):
    if overwrite:
      _write_csv(file_path, input_dict)
      if verbose: print(f"Overwritten csv-file: {file_path}")
    else:
      _merge_columns_into_csv(file_path, input_dict)
      if verbose: print(f"Merged columns into existing csv-file: {file_path}")
  else:
    _write_csv(file_path, input_dict)
    if verbose: print(f"Saved new csv-file: {file_path}")

def _write_csv(
    file_path: Path,
    input_dict: dict
  ):
  with open(file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(input_dict.keys())
    writer.writerows(zip(*input_dict.values()))

def _merge_columns_into_csv(
    file_path: Path,
    input_dict: dict,
  ):
  # Read existing data
  existing_data = read_csv_file_into_dict(file_path, verbose=False)
  # Validate shared keys match
  for key in existing_data:
    if key in input_dict:
      if existing_data[key] != input_dict[key]:
        raise ValueError(f"Mismatch in values for shared key: '{key}'")
  ## Check new keys are same length
  existing_len = len(next(iter(existing_data.values())))
  for key in input_dict:
    if key not in existing_data:
      if len(input_dict[key]) != existing_len:
        raise ValueError(
          f"Length mismatch in new column '{key}': expected {existing_len}, got {len(input_dict[key])}"
        )
  ## merge new keys
  for key in input_dict:
    if key not in existing_data:
      existing_data[key] = input_dict[key]
  _write_csv(file_path, existing_data)


## END OF MODULE