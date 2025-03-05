import os
import importlib
import inspect

for module_name in os.listdir("."):
  module_path = os.path.join(".", module_name)
  if os.path.isdir(module_path) and not module_name.startswith("__"):
    print("===============================")
    print(f"Module: {module_name}")
    print("===============================")
    for script in os.listdir(module_path):
      if script.endswith(".py") and not script.startswith("__"):
        script_name = script.replace(".py", "")
        full_module_name = f"{module_name}.{script_name}"
        try:
          imported_script = importlib.import_module(full_module_name)
          functions = [
            name
            for name, _ in inspect.getmembers(
              imported_script,
              inspect.isfunction
            )
          ]
          print(f"Script: {script_name}.py")
          for func in functions:
            print(f"> {func}()")
        except Exception as e:
          print(f"Error: Could not load {script_name}: {e}")
    print(" ")
## end