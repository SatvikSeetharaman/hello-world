from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

files = sorted(DATA_DIR.glob("passes_part*.csv"))
if not files:
    raise FileNotFoundError("No processed pass CSVs found")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Clean / derived controls
df["abs_start_y_center"] = (df["start_y"] - 40).abs()
df["forward"] = (df["progression"] > 0).astype(int)
df["pressure_relief_tau5"] = -df["pt_tau5"]
df["pressure_relief_tau3"] = -df["pt_tau3"]
df["pressure_relief_tau8"] = -df["pt_tau8"]

# Primary outcome: successful short-horizon continuation already encoded during extraction.
# good5 = possession retained at 5 seconds OR a shot within 5 seconds.
OUTCOME = "good5"

continuous = [
    "pt_tau5", "progression", "pass_length", "p_start_tau5",
    "start_x", "abs_start_y_center"
]
means = {}
stds = {}
for col in continuous:
    means[col] = float(df[col].mean())
    stds[col] = float(df[col].std(ddof=0))
    df["z_" + col] = (df[col] - means[col]) / stds[col]

def fit_logit(outcome, include_pt=True, tau="tau5"):
    ptcol = f"pt_{tau}"
    zpt = f"z_{ptcol}"
    if zpt not in df.columns:
        mu = df[ptcol].mean()
        sd = df[ptcol].std(ddof=0)
        df[zpt] = (df[ptcol] - mu) / sd
    cols = ["z_progression", "z_pass_length", "z_p_start_tau5",
            "z_start_x", "z_abs_start_y_center"]
    if include_pt:
        cols = [zpt] + cols
    X = sm.add_constant(df[cols], has_constant="add")
    model = sm.GLM(df[outcome], X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["match_id"]})
    pred = res.predict(X)
    auc = roc_auc_score(df[outcome], pred)
    return res, auc, cols

base_res, base_auc, _ = fit_logit(OUTCOME, include_pt=False)
main_res, main_auc, main_cols = fit_logit(OUTCOME, include_pt=True, tau="tau5")

sens = {}
for tau in ["tau3", "tau5", "tau8"]:
    r, auc, _ = fit_logit(OUTCOME, include_pt=True, tau=tau)
    k = f"z_pt_{tau}"
    b = float(r.params[k])
    se = float(r.bse[k])
    sens[tau] = {
        "coef_log_odds_per_sd": b,
        "odds_ratio_per_sd": math.exp(b),
        "ci95_or": [math.exp(b - 1.96*se), math.exp(b + 1.96*se)],
        "p_value": float(r.pvalues[k]),
        "auc": float(auc)
    }

# Secondary outcomes
secondary = {}
for outcome in ["retained5", "shot10"]:
    r, auc, _ = fit_logit(outcome, include_pt=True, tau="tau5")
    k = "z_pt_tau5"
    b = float(r.params[k]); se = float(r.bse[k])
    secondary[outcome] = {
        "rate": float(df[outcome].mean()),
        "odds_ratio_per_sd_pt": math.exp(b),
        "ci95_or": [math.exp(b-1.96*se), math.exp(b+1.96*se)],
        "p_value": float(r.pvalues[k]),
        "auc": float(auc)
    }

# Interaction: does pressure transfer become more/less costly for progressive passes?
df["z_pt_x_z_prog"] = df["z_pt_tau5"] * df["z_progression"]
cols_int = ["z_pt_tau5","z_progression","z_pt_x_z_prog","z_pass_length",
            "z_p_start_tau5","z_start_x","z_abs_start_y_center"]
Xint = sm.add_constant(df[cols_int], has_constant="add")
int_res = sm.GLM(df[OUTCOME], Xint, family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups":df["match_id"]})

# Quintile summaries
df["pt_quintile"] = pd.qcut(df["pt_tau5"], 5,
    labels=["Q1 most relief","Q2","Q3","Q4","Q5 most transfer"])
q = df.groupby("pt_quintile", observed=False).agg(
    n=("pass_id","size"),
    mean_pt=("pt_tau5","mean"),
    mean_progression=("progression","mean"),
    good5_rate=("good5","mean"),
    retained5_rate=("retained5","mean"),
    shot10_rate=("shot10","mean")
).reset_index()
q.to_csv(OUT_DIR/"pressure_quintiles.csv", index=False)

# 2D descriptive table: progression x pressure transfer
df["prog_quartile"] = pd.qcut(df["progression"], 4,
    labels=["P1 least","P2","P3","P4 most progressive"])
grid = df.pivot_table(index="pt_quintile", columns="prog_quartile",
    values="good5", aggfunc=["mean","count"], observed=False)
grid.to_csv(OUT_DIR/"pressure_progression_grid.csv")

# Player descriptive leaderboard: pressure relief among meaningful forward passes.
# Minimum 40 qualifying forward passes to reduce tiny-sample noise.
fwd = df[df["progression"] > 5].copy()
players = fwd.groupby(["team","passer"]).agg(
    forward_passes=("pass_id","size"),
    mean_progression=("progression","mean"),
    mean_pressure_relief=("pressure_relief_tau5","mean"),
    good5_rate=("good5","mean")
).reset_index()
players = players[players.forward_passes >= 40].sort_values(
    ["mean_pressure_relief","good5_rate"], ascending=[False,False])
players.to_csv(OUT_DIR/"player_pressure_relief.csv", index=False)

# Figure 1: success rate by pressure-transfer quintile
plotq = q.copy()
p = plotq["good5_rate"].to_numpy()
n = plotq["n"].to_numpy()
se = np.sqrt(p*(1-p)/n)
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(np.arange(5), p, yerr=1.96*se, marker="o", capsize=4)
ax.set_xticks(np.arange(5), plotq["pt_quintile"], rotation=15)
ax.set_ylabel("Successful 5-second continuation rate")
ax.set_xlabel("Pressure transfer quintile (tau=5)")
ax.set_title("Pass outcomes decline as pressure is transferred to the target")
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG_DIR/"success_by_pressure_transfer.png", dpi=220)
plt.close(fig)

# Figure 2: binned relationship controlling visually for progression
df["pt_decile"] = pd.qcut(df["pt_tau5"], 10, labels=False, duplicates="drop")
b = df.groupby("pt_decile").agg(
    mean_pt=("pt_tau5","mean"),
    good5=("good5","mean"),
    n=("good5","size")
).reset_index()
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(b["mean_pt"], b["good5"], s=np.sqrt(b["n"])*4)
coef=np.polyfit(b["mean_pt"], b["good5"],1)
x=np.linspace(b["mean_pt"].min(),b["mean_pt"].max(),100)
ax.plot(x,coef[0]*x+coef[1])
ax.set_xlabel("Pressure transfer: target pressure - passer pressure")
ax.set_ylabel("Successful 5-second continuation rate")
ax.set_title("Higher pressure transfer is associated with worse continuation")
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG_DIR/"binned_pressure_transfer.png",dpi=220)
plt.close(fig)

k="z_pt_tau5"
bmain=float(main_res.params[k]); semain=float(main_res.bse[k])
or_main=math.exp(bmain)
ci_main=[math.exp(bmain-1.96*semain), math.exp(bmain+1.96*semain)]

summary = {
    "n_passes": int(len(df)),
    "n_matches": int(df.match_id.nunique()),
    "n_teams": int(df.team.nunique()),
    "outcome_rate_good5": float(df.good5.mean()),
    "retained5_rate": float(df.retained5.mean()),
    "shot10_rate": float(df.shot10.mean()),
    "pt_tau5_mean": float(df.pt_tau5.mean()),
    "pt_tau5_sd": float(df.pt_tau5.std(ddof=0)),
    "primary": {
        "odds_ratio_per_sd_pressure_transfer": or_main,
        "ci95_or": ci_main,
        "p_value": float(main_res.pvalues[k]),
        "coef_log_odds_per_sd": bmain,
        "auc_baseline": float(base_auc),
        "auc_with_pressure_transfer": float(main_auc),
        "auc_delta": float(main_auc-base_auc),
        "baseline_aic": float(base_res.aic),
        "full_aic": float(main_res.aic),
        "aic_delta": float(base_res.aic-main_res.aic)
    },
    "sensitivity": sens,
    "secondary": secondary,
    "interaction": {
        "coef": float(int_res.params["z_pt_x_z_prog"]),
        "p_value": float(int_res.pvalues["z_pt_x_z_prog"])
    },
    "standardization": {"means":means,"stds":stds}
}
with open(OUT_DIR/"results.json","w") as f:
    json.dump(summary,f,indent=2)

# Compact markdown report
lines=[]
lines.append("# Results")
lines.append("")
lines.append(f"Dataset: **{len(df):,} completed passes from {df.match_id.nunique()} matches and {df.team.nunique()} teams**.")
lines.append(f"Successful 5-second continuation rate: **{df.good5.mean():.1%}**.")
lines.append("")
lines.append("## Primary model")
lines.append("")
lines.append("Cluster-robust logistic regression (standard errors clustered by match) controlling for progression, pass length, passer pressure, field position, and lateral starting position.")
lines.append("")
lines.append(f"- One SD increase in pressure transfer (more pressure at the pass target relative to the passer) had OR **{or_main:.3f}** (95% CI {ci_main[0]:.3f}-{ci_main[1]:.3f}, p={main_res.pvalues[k]:.3g}) for successful 5-second continuation.")
lines.append(f"- Baseline AUC: **{base_auc:.3f}**; with pressure transfer: **{main_auc:.3f}** (Δ={main_auc-base_auc:+.3f}).")
lines.append(f"- AIC improvement after adding pressure transfer: **{base_res.aic-main_res.aic:.1f}**.")
lines.append("")
lines.append("## Sensitivity to pressure radius")
for tau,v in sens.items():
    lines.append(f"- {tau}: OR {v['odds_ratio_per_sd']:.3f} (95% CI {v['ci95_or'][0]:.3f}-{v['ci95_or'][1]:.3f}), p={v['p_value']:.3g}.")
lines.append("")
lines.append("## Secondary outcomes")
for out,v in secondary.items():
    lines.append(f"- {out}: event rate {v['rate']:.1%}; pressure-transfer OR {v['odds_ratio_per_sd_pt']:.3f} (95% CI {v['ci95_or'][0]:.3f}-{v['ci95_or'][1]:.3f}), p={v['p_value']:.3g}.")
lines.append("")
lines.append("## Descriptive quintiles")
lines.append("")
lines.append(q.to_markdown(index=False,floatfmt=".3f"))
lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append("The sign convention is target pressure minus passer pressure. Positive values therefore mean the pass moves the ball into a more pressured spatial state; negative values mean the pass relieves pressure. Results are associative, not causal.")
(OUT_DIR/"RESULTS.md").write_text("\n".join(lines))

print(json.dumps(summary, indent=2))
