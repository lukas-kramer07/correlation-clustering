"""Print summary tables of the experiment results from results/*.csv."""
import pandas as pd

from common import RESULTS

syn = pd.read_csv(RESULTS / "synthetic.csv")
real = pd.read_csv(RESULTS / "real.csv")


def rows(df, label_fn):
    out = []
    for key, g in df.groupby(label_fn, sort=False):
        ref = g["ref"].iloc[0]
        gap = (g["opt"] / g["lp"]).mean() if g["opt"].notna().all() and (g["lp"] > 0).all() else float("nan")
        out.append(dict(instance=key, n=int(g["n"].iloc[0]), ref=ref, k=len(g),
                        pivot=g["pivot_mean/ref"].mean(), pivot_worst=g["pivot_max/ref"].max(),
                        lppivot=g["lppivot_mean/ref"].mean(), lppivot_worst=g["lppivot_max/ref"].max(),
                        ls=g["pivot_ls/ref"].mean(), gap=gap,
                        ari_p=g.get("ari_pivot", pd.Series(dtype=float)).mean(),
                        ari_l=g.get("ari_lppivot", pd.Series(dtype=float)).mean(),
                        lp_time=g["lp_time"].mean()))
    return pd.DataFrame(out)


syn["key"] = syn.apply(lambda r: r["family"] + ("" if pd.isna(r["eps"]) else f", eps={r['eps']}"), axis=1)
S = rows(syn, "key")
real["key"] = real["dataset"] + " " + real["kind"]
Rl = rows(real, "key")
pd.set_option("display.width", 250)
print(S.round(3).to_string())
print(Rl.round(3).to_string())
print(real[["key", "lp", "pivot_mean", "lppivot_mean", "pivot_ls", "pick_a_cluster/lp",
            "prob_constraints", "triangle", "lp_time"]].round(3).to_string())
