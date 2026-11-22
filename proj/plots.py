import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics import ANGLES, MASKS, WINDOWS


LABELS = {"full": "Full scan", "low_information": "Low-information window", "contrast": "Contrast window"}
METHODS = {
    "grid_dense": "Dense grid",
    "grid_budget": "Budget-matched grid",
    "mlp": "MLP",
    "mdn": "Conditional mixture",
    "mdn_physics": "Mixture + physics",
}
COLORS = ["#233e59", "#6793aa", "#ad7142", "#7c649d", "#269282"]


def case_figure(path, observations, truth, case, dense_grid, grid_loss, proposal, reranked):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    side = int(np.sqrt(len(dense_grid)))
    a = dense_grid[:, 0].reshape(side, side)[:, 0]
    c = dense_grid[:, 1].reshape(side, side)[0]
    for col, (condition, mask) in enumerate(MASKS.items()):
        spectrum, landscape = axes[:, col]
        spectrum.plot(ANGLES, observations["full"][case], color="#9aabba", alpha=0.55, linewidth=1)
        spectrum.plot(ANGLES[mask], observations[condition][case, mask], color="#233e59", linewidth=1.1)
        spectrum.set(xlabel="2θ (°)", xlim=(20, 70), title=LABELS[condition])
        if col == 0:
            spectrum.set_ylabel("Relative intensity")
            for reflection, limits, offset in [
                ("(110)", (26, 29), (7, -10)),
                ("(101)", (34, 38), (7, 8)),
            ]:
                indices = np.flatnonzero((ANGLES >= limits[0]) & (ANGLES <= limits[1]))
                index = indices[np.argmax(observations[condition][case, indices])]
                spectrum.annotate(
                    reflection, (ANGLES[index], observations[condition][case, index]),
                    xytext=offset, textcoords="offset points", fontsize=8,
                )
        if condition != "full":
            spectrum.axvspan(*WINDOWS[condition], color="#269282", alpha=0.13)

        loss = grid_loss[condition][case].reshape(side, side)
        excess_loss = np.log10(np.maximum(loss - loss.min(), 1e-8))
        heatmap = landscape.pcolormesh(a, c, excess_loss.T, cmap="cividis", vmin=-8, vmax=-2)
        candidates = proposal[condition][case]
        final = reranked[condition][case]
        landscape.scatter(*candidates.T, s=16, color="#aee0f5", alpha=0.7, label="Generated pool")
        landscape.scatter(*final.T, s=32, facecolors="none", edgecolors="white", label="Physics top 5")
        landscape.scatter(*truth, marker="*", s=125, color="#fa9676", edgecolors="#33251d", label="True cell")
        landscape.set(xlabel="a (Å)", xlim=(a[0], a[-1]), ylim=(c[0], c[-1]))
        if col == 0:
            landscape.set_ylabel("c (Å)")

    axes[1, 0].legend(loc="lower left", fontsize=7)
    fig.colorbar(heatmap, ax=axes[1, :], shrink=0.86, label="log₁₀ excess observed MSE")
    fig.suptitle(f"Held-out sample {case} · same crystal under three observations", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metrics_figure(path, metrics):
    lookup = {(row["condition"], row["method"]): row for row in metrics}
    columns = [
        ("mae_a_angstrom", "Top-1 MAE for a (Å)"),
        ("mae_c_angstrom", "Top-1 MAE for c (Å)"),
        ("top5_hit_rate", "Top-5 hit rate (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    x = np.arange(len(MASKS))
    width = 0.16
    for axis, (column, title) in zip(axes, columns):
        for i, (method, label) in enumerate(METHODS.items()):
            values = [lookup[(condition, method)][column] for condition in MASKS]
            if all(value is None for value in values):
                continue
            if column == "top5_hit_rate":
                values = [100 * value for value in values]
            axis.bar(x + (i - 2) * width, values, width=width, color=COLORS[i], label=label)
        axis.set(xticks=x, xticklabels=["Full", "Low-info", "Contrast"], title=title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[-1].set_ylim(0, 108)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=5)
    fig.savefig(path, dpi=180)
    plt.close(fig)
