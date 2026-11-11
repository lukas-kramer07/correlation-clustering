"""Print summary tables of the experiment results from results/*.csv.

All ratios are mean cost / LP (an upper bound on the ratio to OPT); opt/lp is the
integrality gap, averaged over the instances where the ILP finished.
"""
import pandas as pd

from common import RESULTS


def table(df, key):
    g = df.groupby(key, sort=False)
    out = pd.DataFrame({
        "n": g["n"].first(),
        "instances": g.size(),
        "opt/lp": g.apply(lambda x: (x["opt"] / x["lp"]).mean() if x["opt"].notna().any() and (x["lp"] > 0).all() else float("nan")),
        "pivot": g["pivot_mean/lp"].mean(),
        "pivot worst run": g["pivot_max/lp"].max(),
        "lp rounding": g["lppivot_mean/lp"].mean(),
        "lp rounding worst run": g["lppivot_max/lp"].max(),
        "lp time [s]": g["lp_time"].mean(),
    })
    return out.round(3)


syn = pd.read_csv(RESULTS / "synthetic.csv")
syn["key"] = syn.apply(lambda r: r["family"] + ("" if pd.isna(r["eps"]) else f", eps={r['eps']}"), axis=1)
real = pd.read_csv(RESULTS / "real.csv")
pd.set_option("display.width", 200)
print(pd.read_csv(RESULTS / "tight.csv").round(3).to_string(index=False), end="\n\n")
print(table(syn, "key").to_string(), end="\n\n")
print(table(real, "dataset").to_string(), end="\n\n")
ex = pd.read_csv(RESULTS / "exhaustive.csv")
print("exhaustive check, maximum over all sign patterns on n vertices:")
print(ex.groupby("n").agg(classes=("opt", "size"), **{c: (c, "max") for c in ex.columns[-3:]}).round(4).to_string())
