"""Render paper figures from every completed condition under a result root."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from experiments import datasets, executors
from experiments.layer2 import quantile

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE = tomllib.loads(Path(__file__).with_name("figstyle.toml").read_text())
T, MARKS = STYLE["tokens"], STYLE["marks"]
_FONTS = {font.name for font in fm.fontManager.ttflist}


def _font(*names: str) -> str:
    return next((name for name in names if name in _FONTS), names[-1])


SERIF = _font("Source Serif 4", "Source Serif Pro", "DejaVu Serif")
SANS = _font("Inter", "DejaVu Sans")
MONO = _font("JetBrains Mono", "DejaVu Sans Mono")
CI_MULTIPLIER = 1.959963984540054


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": T["paper"], "axes.facecolor": T["paper"],
        "savefig.facecolor": T["paper"], "font.family": SANS,
        "text.color": T["ink"], "axes.labelcolor": T["ink_2"],
        "xtick.color": T["ink_2"], "ytick.color": T["ink_2"],
        "axes.edgecolor": T["rule"], "grid.color": T["rule"],
        "axes.linewidth": MARKS["hairline"], "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 200,
    })


ARM_LABEL = {"naive": "Naive", "continuation": "Branchpoint", "spread": "BpDynamic"}
MODEL_LABEL = {"sol": "gpt-5.6-sol", "opus": "claude-opus-5", "glm": "glm-5.3"}
# Fixed reading windows. Eucalyptus is set by hand: one human analysis sits at
# Zr -4.5, and letting it set the axis compresses every LLM point (which span
# -0.13 to +0.22) into an unreadable sliver. Points outside are counted on the
# figure rather than silently dropped.
FOREST_XLIM = {"eu": (-0.25, 0.30)}


def _color(condition: dict) -> str:
    # Branchpoint is the pink; BpDynamic the green. The red the two used to
    # share sat at CVD dE 3.9 (tritan) against the pink -- unreadable for some
    # readers in the one figure where both appear.
    return T[{"naive": "naive", "spread": "spread"}.get(condition["arm"], "neon")]


def _focal_test(row: dict) -> float | None:
    stat = row.get("stats", {})
    t, z = stat.get("t_stat"), stat.get("z_stat")
    return t if z is None else z if t is None else None


def discover(root: Path) -> list[dict]:
    """Discover conditions from recorded metadata, never directory naming rules."""
    found = []
    for raw_path in sorted(root.rglob("results.json")) if root.exists() else []:
        condition = raw_path.parent
        final_path = condition / "results_final.json"
        if not final_path.exists():
            continue
        raw, result = json.loads(raw_path.read_text()), json.loads(final_path.read_text())
        rows = [
            row for row in result["rows"]
            if row.get("estimate_standardized") is not None
            and row.get("scope") == "E"
        ]
        found.append({
            "name": condition.name, "dataset": raw["dataset"], "model": raw["model"],
            "arm": (
                "naive"
                if raw["mode"] == "naive"
                # spread and continuation share branching_method="continuation";
                # only the allocation separates them, and without this they
                # collide on (dataset, model, arm, seed) and one is dropped.
                else "spread"
                if raw.get("allocation") == "spread"
                else raw.get("branching_method", raw.get("allocation", "uniform"))
            ),
            "seed": raw.get("seed"),
            "counts": {
                "raw": len(result["rows"]),
                "standardized": sum(
                    row.get("estimate_standardized") is not None for row in result["rows"]
                ),
                "in_scope": len(rows),
            },
            "values": [row["estimate_standardized"] for row in rows],
            "runs": [
                {"id": row.get("id") or row.get("branch") or row.get("seed"),
                 "z": row["estimate_standardized"],
                 "se": row.get("se_standardized")}
                for row in rows
            ],
            "tests": [
                {
                    "id": row.get("id") or row.get("branch") or row.get("seed"),
                    "value": _focal_test(row),
                }
                for row in result["rows"]
            ],
        })
    return found


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def _label(condition: dict) -> str:
    seed = f" s{condition['seed']}" if condition["seed"] is not None else ""
    return f"{executors.short_name(condition['model'])} · {condition['arm']}{seed}"


def _dataset_label(ds: str) -> str:
    return datasets.get(ds).get("crowd", ds).title()


def _point_interval(point: dict) -> tuple[float, float]:
    """Return plot errors as distances from the estimate to a 95% interval."""
    estimate = point["estimate"]
    if point.get("lower") is not None and point.get("upper") is not None:
        return estimate - point["lower"], point["upper"] - estimate
    if point.get("se") is not None:
        width = CI_MULTIPLIER * point["se"]
        return width, width
    return 0.0, 0.0


def _draw_ranked(ax, points: list[dict], *, color: str, label: str, human: bool) -> None:
    points = sorted(points, key=lambda point: point["estimate"])
    if not points:
        return
    ys = [(index + .5) / len(points) for index in range(len(points))]
    errors = [_point_interval(point) for point in points]
    ax.errorbar(
        [point["estimate"] for point in points], ys,
        xerr=[
            [error[0] for error in errors],
            [error[1] for error in errors],
        ],
        ls="", marker="o", ms=2 if human else 3, color=color, ecolor=color,
        elinewidth=MARKS["hairline"], alpha=.4 if human else .9,
        zorder=1 if human else 3, label=label,
        markeredgecolor=T["paper"], markeredgewidth=0 if human else .4,
    )


def _interval_bounds(points: list[dict]) -> list[float]:
    bounds = []
    for point in points:
        left, right = _point_interval(point)
        bounds.extend((point["estimate"] - left, point["estimate"] + right))
    return bounds


def _pick(conditions: list[dict], ds: str, model: str, arm: str) -> dict | None:
    matches = [
        condition for condition in conditions
        if condition["dataset"] == ds
        and executors.short_name(condition["model"]) == model
        and condition["arm"] == arm
    ]
    return next((condition for condition in matches if condition["seed"] == 7),
                matches[0] if matches else None)


def forest_panels(conditions: list[dict], model: str, out: Path) -> None:
    """Human, naive and Branchpoint distributions, one panel per dataset."""
    panels = []
    for ds in ("bt", "eu", "marp"):
        arms = [_pick(conditions, ds, model, arm) for arm in ("naive", "continuation")]
        if any(arms):
            panels.append((ds, [condition for condition in arms if condition]))
    if not panels:
        return
    _style()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(2.35 * len(panels) + .5, 3.5), squeeze=False,
    )
    for ax, (ds, selected) in zip(axes[0], panels, strict=True):
        crowd = datasets.crowd(ds)
        series = [("Human", T["ink_2"], crowd)] + [
            (ARM_LABEL[condition["arm"]],
             _color(condition),
             [{"estimate": run["z"], "se": run["se"]} for run in condition["runs"]])
            for condition in selected if condition["runs"]
        ]
        agent_bounds = [
            bound for name, _, points in series if name != "Human"
            for bound in _interval_bounds(points)
        ]
        human_values = [point["estimate"] for point in crowd]
        low = min([quantile(human_values, .025), *agent_bounds])
        high = max([quantile(human_values, .975), *agent_bounds])
        margin = (high - low) * .06 or .05
        window = FOREST_XLIM.get(ds)
        for name, color, points in series:
            _draw_ranked(ax, points, color=color, label=name, human=name == "Human")
        ax.set(xlim=window or (low - margin, high + margin),
               ylim=(-.04, 1.04), yticks=[])
        metric = "Standardized beta" if ds == "marp" else "Fisher z"
        ax.set_xlabel(metric, size=9)
        ax.set_title({"bt": "Blue tit", "eu": "Eucalyptus",
                      "marp": "Religiosity and well-being"}[ds],
                     family=SERIF, size=10.5, loc="center", pad=8)
    axes[0][0].set_ylabel("Rank", size=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(handles, labels, prop={"size": 8},
                       loc="lower right", ncol=1, handletextpad=.4,
                       borderaxespad=.8, labelspacing=.5, frameon=True,
                       facecolor=T["paper"], edgecolor="none", framealpha=.88)
    fig.subplots_adjust(wspace=.16, top=.88)
    stem = "forest_arms" if model == "sol" else f"forest_arms_{model}"
    _save(fig, out, stem)


def forest_allocation(conditions: list[dict], model: str, out: Path) -> None:
    """Branchpoint against BpDynamic -- the allocation contrast, agents only.

    No human series: this asks whether spending the same branch budget on
    least-tried decisions covers more than drawing them at random, which is an
    internal comparison. Adding the crowd would set the axis by a population
    neither arm is being measured against here.
    """
    panels = []
    for ds in ("bt", "eu"):
        arms = [_pick(conditions, ds, model, arm) for arm in ("continuation", "spread")]
        selected = [condition for condition in arms if condition and condition["runs"]]
        if len(selected) == 2:
            panels.append((ds, selected))
    if not panels:
        return
    _style()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.3 * len(panels) + .6, 3.8), squeeze=False,
    )
    for ax, (ds, selected) in zip(axes[0], panels, strict=True):
        series = [
            (ARM_LABEL[condition["arm"]], _color(condition),
             [{"estimate": run["z"], "se": run["se"]} for run in condition["runs"]])
            for condition in selected
        ]
        bounds = sorted(b for _, _, points in series for b in _interval_bounds(points))
        low, high = quantile(bounds, .01), quantile(bounds, .99)
        margin = (high - low) * .06 or .05
        low, high = low - margin, high + margin
        for name, color, points in series:
            _draw_ranked(ax, points, color=color, label=name, human=False)
        ax.set(xlim=(low, high), ylim=(-.04, 1.04), yticks=[])
        ax.set_xlabel("Fisher z", size=9)
        ax.set_title({"bt": "Blue tit", "eu": "Eucalyptus"}[ds],
                     family=SERIF, size=10.5, loc="center", pad=8)
    axes[0][0].set_ylabel("Rank", size=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(handles, labels, prop={"size": 8},
                       loc="lower right", ncol=1, handletextpad=.4,
                       borderaxespad=.8, labelspacing=.5, frameon=True,
                       facecolor=T["paper"], edgecolor="none", framealpha=.88)
    fig.subplots_adjust(wspace=.18, top=.86)
    stem = "forest_allocation" if model == "sol" else f"forest_allocation_{model}"
    _save(fig, out, stem)


def _complete_manual_artifacts(condition: Path) -> None:
    """Refuse a hand-audit plot until every source row has a disposition."""
    source = json.loads(condition.joinpath("results.json").read_text())
    overrides = json.loads(condition.joinpath("standardization_overrides.json").read_text())
    final = json.loads(condition.joinpath("results_final.json").read_text())
    source_ids = [
        f"seed{row['seed']:03d}" if "seed" in row else row["branch"]
        for row in source["rows"]
    ]
    if set(overrides) != set(source_ids):
        raise ValueError(
            f"{condition.name}: manual audit is incomplete "
            f"({len(overrides)}/{len(source_ids)} dispositions)"
        )
    if len(final["rows"]) != len(source_ids):
        raise ValueError(f"{condition.name}: final artifact is stale or incomplete")
    for row_id, override in overrides.items():
        if not override.get("evidence") or not override.get("method"):
            raise ValueError(f"{condition.name}/{row_id}: unaudited override")


def bt_sol_overlay(conditions: list[dict], root: Path, out: Path) -> None:
    """Validated blue-tit Sol overlay for the manual-normalization audit."""
    # This overlay is a two-arm audit artifact and asserts exactly two below;
    # it deliberately does not enumerate spread.
    selected = [
        condition for arm in ("naive", "continuation")
        if (condition := _pick(conditions, "bt", "sol", arm))
    ]
    if len(selected) != 2:
        raise ValueError("blue-tit Sol overlay requires naive and continuation conditions")
    for condition in selected:
        _complete_manual_artifacts(root / condition["name"])

    crowd = datasets.crowd("bt")
    _style()
    fig, ax = plt.subplots(figsize=(5.3, 4.1))
    series = [(f"Human ({len(crowd)}/{len(crowd)})", T["ink_2"], crowd, True)]
    for condition in selected:
        included, total = condition["counts"]["in_scope"], condition["counts"]["raw"]
        name = ARM_LABEL[condition["arm"]]
        points = [
            {"estimate": run["z"], "se": run["se"]}
            for run in condition["runs"]
        ]
        series.append((f"{name} ({included}/{total})", _color(condition), points, False))
    bounds = [bound for _, _, points, _ in series for bound in _interval_bounds(points)]
    low, high = min(bounds), max(bounds)
    margin = (high - low) * .05 or .05
    for label, color, points, human in series:
        _draw_ranked(ax, points, color=color, label=label, human=human)
    ax.axvline(0, color=T["rule"], lw=MARKS["hairline"], zorder=0)
    ax.set(
        xlim=(low - margin, high + margin), ylim=(-.04, 1.04), yticks=[],
        xlabel="Fisher z (points with 95% intervals)", ylabel="Rank",
    )
    ax.set_title("Blue tit", family=SERIF, size=12, loc="left", pad=12)
    ax.legend(frameon=False, prop={"size": 8}, loc="best")
    _save(fig, out, "bt_sol_human_overlay")


def _audited_standardized_points(
    condition: Path, expected_overrides: dict[str, dict]
) -> tuple[list[dict], int]:
    """Load only rows reproduced by the current executable audit ledger."""
    stored = json.loads(condition.joinpath("standardization_overrides.json").read_text())
    if stored != expected_overrides:
        raise ValueError(f"{condition.name}: overrides are stale relative to audit recipes")
    source = json.loads(condition.joinpath("results.json").read_text())
    standardized = json.loads(
        condition.joinpath("results_standardized.json").read_text()
    )
    if len(source["rows"]) != len(standardized["rows"]):
        raise ValueError(f"{condition.name}: standardized artifact is stale")

    points = []
    for raw, row in zip(source["rows"], standardized["rows"], strict=True):
        row_id = f"seed{raw['seed']:03d}" if "seed" in raw else raw["branch"]
        override = expected_overrides[row_id]
        value, standard_error = override["value"], override["se"]
        if value is not None and raw.get("orientation") == "reversed":
            value = -value
        observed = row.get("estimate_standardized")
        observed_se = row.get("se_standardized")
        if value is None:
            if observed is not None:
                raise ValueError(f"{condition.name}/{row_id}: excluded row was plotted")
            continue
        if observed is None or not math.isclose(observed, value, rel_tol=1e-12):
            raise ValueError(f"{condition.name}/{row_id}: stale normalized estimate")
        if standard_error is not None and (
            observed_se is None
            or not math.isclose(observed_se, standard_error, rel_tol=1e-12)
        ):
            raise ValueError(f"{condition.name}/{row_id}: stale normalized SE")
        points.append({"estimate": observed, "se": observed_se})
    return points, len(source["rows"])


def eu_sol_normalization_overlay(root: Path, out: Path) -> None:
    """EU normalization audit; deliberately does not apply the scope cache."""
    from experiments.manual_normalization_eu import validate_all

    audited = validate_all()
    arms = []
    for name, label in (
        ("eu_sol_naive", "Naive"),
        ("eu_sol_continuation", "Continuation"),
    ):
        relative = f"results/sol_v6/{name}"
        points, total = _audited_standardized_points(root / name, audited[relative])
        arms.append((label, points, total))

    crowd = datasets.crowd("eu")
    _style()
    fig, ax = plt.subplots(figsize=(5.3, 4.1))
    series = [(f"Human ({len(crowd)}/{len(crowd)})", T["ink_2"], crowd, True)]
    for label, points, total in arms:
        condition = {"arm": "naive" if label == "Naive" else "continuation"}
        series.append((f"{label} ({len(points)}/{total})", _color(condition), points, False))
    bounds = [bound for _, _, points, _ in series for bound in _interval_bounds(points)]
    low, high = min(bounds), max(bounds)
    margin = (high - low) * .05 or .05
    for label, color, points, human in series:
        _draw_ranked(ax, points, color=color, label=label, human=human)
    ax.axvline(0, color=T["rule"], lw=MARKS["hairline"], zorder=0)
    ax.set(
        xlim=(low - margin, high + margin), ylim=(-.04, 1.04), yticks=[],
        xlabel="Fisher z (points with 95% intervals)", ylabel="Rank",
    )
    ax.set_title("Eucalyptus — normalization audit", family=SERIF,
                 size=12, loc="left", pad=12)
    ax.legend(frameon=False, prop={"size": 8}, loc="best")
    _save(fig, out, "eu_sol_human_overlay_normalization_audit")


def _marp_human_points() -> list[dict]:
    source = REPO_ROOT / "research/scratch/marp/MARP_data_main.csv"
    points = []
    for row in csv.DictReader(source.open(errors="replace")):
        if row.get("MARP_ES_type_1") != "beta":
            continue
        try:
            estimate = float(row["Standardized_ES_1"])
            lower = float(row["lower_CI_1"])
            upper = float(row["upper_CI_1"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(estimate) < 1 and lower <= estimate <= upper:
            points.append({"estimate": estimate, "lower": lower, "upper": upper})
    return points


def marp_official_sol_normalization_overlay(root: Path, out: Path) -> None:
    """Plot audited Sol continuation against the publication's RQ1 betas."""
    from experiments.manual_normalization_marp_official_continuation import validate

    relative = "results/marp_official_v7/marp_sol_continuation"
    audited = validate()[relative]
    points, total = _audited_standardized_points(
        root / "marp_sol_continuation", audited
    )
    crowd = _marp_human_points()
    _style()
    fig, ax = plt.subplots(figsize=(5.3, 4.1))
    series = [
        (f"Human ({len(crowd)}/{len(crowd)})", T["ink_2"], crowd, True),
        (f"Continuation ({len(points)}/{total})", T["branchpoint"], points, False),
    ]
    bounds = [bound for _, _, values, _ in series for bound in _interval_bounds(values)]
    low, high = min(bounds), max(bounds)
    margin = (high - low) * .05 or .05
    for label, color, values, human in series:
        _draw_ranked(ax, values, color=color, label=label, human=human)
    ax.axvline(0, color=T["rule"], lw=MARKS["hairline"], zorder=0)
    ax.set(
        xlim=(low - margin, high + margin), ylim=(-.04, 1.04), yticks=[],
        xlabel="Standardized beta (points with 95% intervals)", ylabel="Rank",
    )
    ax.set_title("Religiosity and well-being — official data", family=SERIF,
                 size=12, loc="left", pad=12)
    ax.legend(frameon=False, prop={"size": 8}, loc="best")
    _save(fig, out, "marp_official_sol_continuation_normalization_audit")


def _costs(root: Path) -> dict[str, float]:
    costs = {}
    for path in root.rglob("costs.jsonl"):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("cond")
            if name:
                name = name.removeprefix("std_")
                costs[name] = costs.get(name, 0) + row.get("usd", 0)
    return costs


def scatter(conditions: list[dict], root: Path, out: Path) -> None:
    conditions = [condition for condition in conditions if condition["values"]]
    if not conditions:
        return
    _style()
    costs = _costs(root)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for condition in conditions:
        x = max(costs.get(condition["name"], 0), 0.001)
        y = len({round(value, 3) for value in condition["values"]})
        ax.scatter(x, y, s=44, color=_color(condition), edgecolor=T["paper"],
                   linewidth=MARKS["gap"], zorder=3)
        ax.annotate(condition["name"], (x, y), xytext=(7, -3),
                    textcoords="offset points", family=MONO, size=7, color=T["ink_2"])
    ax.set(xscale="log", xlabel="condition cost (USD)", ylabel="distinct answers")
    ax.grid(axis="y", lw=MARKS["hairline"])
    ax.set_title("distinct answers per dollar", family=SERIF, size=12, loc="left", pad=14)
    _save(fig, out, "scatter_cost")


def convergence(ds: str, conditions: list[dict], out: Path) -> None:
    """Width against analyses drawn, one panel per model.

    Faceted rather than overlaid: colour carries the ARM throughout this figure
    set, so three models on one axis would put three different conditions on the
    same red. One panel per model keeps that mapping intact and makes each line
    identifiable without a second colour scale.
    """
    conditions = [condition for condition in conditions if len(condition["values"]) >= 12]
    if not conditions:
        return
    models = sorted({executors.short_name(c["model"]) for c in conditions})
    _style()
    fig, axes = plt.subplots(
        1, len(models), figsize=(2.5 * len(models) + .8, 3.4),
        sharey=True, squeeze=False,
    )
    crowd = [row["estimate"] for row in datasets.crowd(ds)]
    target = quantile(crowd, .95) - quantile(crowd, .05) if crowd else None
    for ax, model in zip(axes[0], models, strict=True):
        for condition in sorted(conditions, key=lambda c: c["arm"]):
            if executors.short_name(condition["model"]) != model:
                continue
            ns = list(range(10, len(condition["values"]) + 1))
            widths = [
                quantile(condition["values"][:n], .95) - quantile(condition["values"][:n], .05)
                for n in ns
            ]
            ax.plot(ns, widths, color=_color(condition), lw=MARKS["series"],
                    label=ARM_LABEL[condition["arm"]], solid_capstyle="round")
        if target:
            ax.axhline(target, color=T["ink_2"], lw=MARKS["hairline"], ls=(0, (4, 3)))
            ax.annotate("human", (ax.get_xlim()[1], target), xytext=(-2, -4),
                        textcoords="offset points", ha="right", va="top",
                        color=T["ink_2"], family=SANS, size=7)
        ax.set_title(MODEL_LABEL.get(model, model), family=SANS, size=9,
                     color=T["ink_2"], loc="left", pad=6)
        ax.set_xlabel("analyses drawn")
        ax.grid(axis="y", lw=MARKS["hairline"])
    axes[0][0].set_ylabel("q05-q95")
    if target:
        axes[0][0].set_ylim(0, target * 1.12)
    # every arm, not just those present in the first panel -- spread exists for
    # one model only and would otherwise be missing from the key
    seen: dict[str, object] = {}
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            seen.setdefault(label, handle)
    fig.suptitle(f"{_dataset_label(ds)} convergence", family=SERIF, size=12,
                 x=.01, ha="left", y=.99, va="top")
    fig.legend(seen.values(), seen.keys(), frameon=False, ncol=len(seen),
               loc="upper left", bbox_to_anchor=(.01, .93),
               prop={"family": SANS, "size": 8})
    fig.subplots_adjust(top=.74)
    _save(fig, out, f"convergence_{ds}")


def attrition(conditions: list[dict], model: str, out: Path) -> None:
    """Show how many terminal analyses survive each reporting boundary."""
    selected = [
        condition for condition in conditions
        if executors.short_name(condition["model"]) == model
        and condition["arm"] in {"naive", "continuation", "spread"}
    ]
    if not selected:
        return
    selected.sort(key=lambda c: (c["dataset"], c["arm"] == "continuation"))
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [f"{_dataset_label(c['dataset'])} · {c['arm']}" for c in selected]
    for y, condition in enumerate(selected):
        counts, color = condition["counts"], _color(condition)
        ax.barh(y, counts["raw"], color=T["paper"], edgecolor=color,
                linewidth=MARKS["series"])
        ax.barh(y, counts["standardized"], color=color, alpha=.28)
        ax.barh(y, counts["in_scope"], color=color, alpha=.9)
        ax.text(counts["raw"] + .7, y,
                f"{counts['in_scope']}/{counts['standardized']}/{counts['raw']}",
                va="center", family=MONO, size=7, color=T["ink_2"])
    ax.set(yticks=range(len(labels)), yticklabels=labels,
           xlabel="terminal analyses", xlim=(0, max(c["counts"]["raw"] for c in selected) + 9))
    ax.invert_yaxis()
    ax.grid(axis="x", lw=MARKS["hairline"])
    ax.set_title("Layer-2 attrition", family=SERIF, size=12, loc="left", pad=14)
    ax.text(0, 1.01, "labels show in-scope / metric-capable / raw",
            transform=ax.transAxes, size=8, color=T["ink_2"], va="bottom")
    stem = "attrition" if model == "sol" else f"attrition_{model}"
    _save(fig, out, stem)


def test_statistic_panels(conditions: list[dict], model: str, out: Path) -> None:
    """All unambiguous focal t/z statistics; diagnostic, not an effect-size plot."""
    panels = []
    for ds in ("bt", "eu", "marp"):
        selected = [
            condition
            for arm in ("naive", "continuation", "spread")
            if (condition := _pick(conditions, ds, model, arm))
        ]
        if selected:
            panels.append((ds, selected))
    if not panels:
        return
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels) + .5, 3.5))
    axes = [axes] if len(panels) == 1 else axes
    for ax, (ds, selected) in zip(axes, panels, strict=True):
        for condition in selected:
            values = sorted(row["value"] for row in condition["tests"] if row["value"] is not None)
            ys = [(index + .5) / len(values) for index in range(len(values))]
            ax.scatter(values, ys, s=11, color=_color(condition), edgecolor=T["paper"],
                       linewidth=.25, label=condition["arm"].title(), zorder=3)
        ax.axvline(0, color=T["rule"], lw=MARKS["hairline"], zorder=0)
        ax.set(ylim=(-.04, 1.04), yticks=[], xlabel="Signed focal t or Wald z")
        ax.set_title({"bt": "Blue tit", "eu": "Eucalyptus",
                      "marp": "Religiosity and well-being"}[ds],
                     family=SERIF, size=10.5, pad=8)
    axes[0].set_ylabel("Rank", size=9)
    axes[0].legend(frameon=False, prop={"size": 8}, loc="best")
    fig.suptitle("Test-statistic diagnostic — not a common effect-size scale",
                 family=SERIF, size=11)
    fig.subplots_adjust(wspace=.2)
    stem = "test_statistics" if model == "sol" else f"test_statistics_{model}"
    _save(fig, out, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/grid")
    args = parser.parse_args()
    root = Path(args.root)
    root = root if root.is_absolute() else REPO_ROOT / root
    out, conditions = root / "figures", discover(root)
    print(f"writing figures from {len(conditions)} discovered condition(s):")
    for model in ("sol", "opus", "glm"):
        forest_panels(conditions, model, out)
        forest_allocation(conditions, model, out)
        attrition(conditions, model, out)
        test_statistic_panels(conditions, model, out)
    if root.name == "sol_v6":
        bt_sol_overlay(conditions, root, out)
        eu_sol_normalization_overlay(root, out)
    for ds in sorted({condition["dataset"] for condition in conditions}):
        selected = [condition for condition in conditions if condition["dataset"] == ds]
        convergence(ds, selected, out)
    scatter(conditions, root, out)
    if SERIF == "DejaVu Serif":
        print("  NOTE: storyline webfonts not installed; fell back to DejaVu.")


if __name__ == "__main__":
    main()
