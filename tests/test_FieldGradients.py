import numpy as np
import matplotlib.pyplot as plt

# Load user-defined modules
from ThePlottingModule import PlotFuncs
from TheAnalysisModule import WWFields

plt.switch_backend("agg")

def main():
  fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 3*5))
  dict_methods = [
    {"func": WWFields.gradient_2ocd, "color": "red", "order": 2},
    {"func": WWFields.gradient_4ocd, "color": "forestgreen", "order": 4},
    {"func": WWFields.gradient_6ocd, "color": "royalblue", "order": 6},
  ]
  x_exact = np.linspace(0, 2 * np.pi, 1000)
  y_exact = np.sin(x_exact)
  dydx_exact = np.cos(x_exact)
  axs[0].plot(x_exact, y_exact, "k-", lw=2, label=r"$y = \sin(x)$")
  axs[1].plot(x_exact, dydx_exact, "k-", lw=2, label=r"$dy/dx = \cos(x)$")
  list_num_points = [5, 10, 20, 40, 100, 200]
  dict_errors = {
    dict_method["order"]: []
    for dict_method in dict_methods
  }
  for dict_method in dict_methods:
    for num_points in list_num_points:
      x = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
      y = np.sin(x)
      dydx_analytic = np.cos(x)
      dydx_numeric = dict_method["func"](y, 2 * np.pi / num_points, 0)
      error = np.linalg.norm(dydx_numeric - dydx_analytic, ord=2) / np.sqrt(num_points)
      dict_errors[dict_method["order"]].append(error)
      # Plot one representative numerical derivative (e.g., for num_points=40)
      if num_points == 40: axs[1].plot(x, dydx_numeric, "o-", lw=2, color=dict_method["color"], label=f"{dict_method['order']}th order")
    axs[2].loglog(
      list_num_points,
      dict_errors[dict_method["order"]],
      "o-", lw=2, color=dict_method["color"], label=f"{dict_method['order']}th order"
    )
  p = np.linspace(5, 200, 10)
  axs[2].loglog(p, 5 * p ** (-2), "r--", label=r"$O(h^2)$")
  axs[2].loglog(p, 50 * p ** (-4), "g--", label=r"$O(h^4)$")
  axs[2].loglog(p, 500 * p ** (-6), "b--", label=r"$O(h^6)$")
  axs[2].set_title("Error Convergence")
  axs[2].set_xlabel("Number of Points (log scale)")
  axs[2].set_ylabel("Error (log scale)")
  axs[2].legend()
  axs[2].grid(True, which="both", linestyle="--", linewidth=0.5)
  print("Saving figure...")
  PlotFuncs.saveFigure(fig, "demo.png", bool_draft=False)

if __name__ == "__main__":
  main()
