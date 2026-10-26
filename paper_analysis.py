"""
paper_analysis.py  --  every number reported in the manuscript comes from this one script.

Input : outages_model_ready.csv   (862 rows; made by build_model_ready.py from outages_raw.csv)
Output: paper_results.json, table_*.csv, Figure_1.png ... Figure_4.png, paper_analysis_output.txt
Run   : python3 paper_analysis.py        (needs pandas numpy scikit-learn matplotlib pytz)

Protocol (same for every model, so models are compared on identical folds):
  * target y = natural log of outage duration in hours
  * "unseen-country" evaluation: countries are shuffled into 5 folds; each fold is predicted by a model trained on the
    other 4 folds. Repeated 20 times with different shuffles (100 fits per model). All models share the same folds.
  * "random-row" evaluation (the leaky comparison): ordinary shuffled 5-fold on rows, repeated 20 times.
"""
import json
import numpy as np, pandas as pd, pytz
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, K, REPS, NBOOT = 42, 5, 20, 1000
df = pd.read_csv("outages_model_ready.csv", keep_default_na=False, na_values=[""])
n = len(df)
y = df.log_duration_hours.values
dur = df.duration_hours.values
country = df.primary_country.values
epi = df.episode_id.values
R = {}                                    # everything that goes into the paper

# ------------------------------------------------------------------ descriptives
start = pd.to_datetime(df.startDate, utc=True)
R["n_rows"] = int(n); R["n_episodes"] = int(df.episode_id.nunique())
R["start_min"] = start.min().strftime("%Y-%m-%d"); R["start_max"] = start.max().strftime("%Y-%m-%d")
known = df[df.primary_country != "UNK"]
R["n_countries"] = int(known.primary_country.nunique()); R["n_unknown_country_rows"] = int((df.primary_country == "UNK").sum())
cc = df.primary_country.value_counts()
R["top_countries"] = {k: int(v) for k, v in cc.head(10).items()}
R["iraq_rows"] = int(cc.get("IQ", 0)); R["iraq_share"] = float(cc.get("IQ", 0) / n)
R["top3_share"] = float(cc.head(3).sum() / n)
R["rows_in_multi_episodes"] = int((df.episode_size > 1).sum()); R["share_in_multi_episodes"] = float((df.episode_size > 1).mean())
R["largest_episode"] = int(df.episode_size.max())
q = np.percentile(dur, [0, 25, 50, 75, 100])
R["dur_min"], R["dur_q1"], R["dur_median"], R["dur_q3"], R["dur_max"] = [float(v) for v in q]
R["dur_mean"] = float(dur.mean()); R["n_over_30d"] = int((dur > 720).sum()); R["n_flag_shared_end"] = int(df.flag_shared_endpoint.sum())
R["type_counts"] = {k: int(v) for k, v in df.type_group.value_counts().items()}
cause_tab = (df.groupby("cause_group").agg(rows=("id", "size"), episodes=("episode_id", "nunique"),
             median_h=("duration_hours", "median"), mean_h=("duration_hours", "mean")).sort_values("rows", ascending=False))
cause_tab.to_csv("table_causes.csv"); R["cause_table"] = cause_tab.round(2).reset_index().to_dict("records")
gov = (df.cause_group == "GOVERNMENT_DIRECTED").values
R["gov_rows"] = int(gov.sum()); R["gov_share"] = float(gov.mean())
R["gov_iraq_rows"] = int(((country == "IQ") & gov).sum())
R["gov_by_country"] = {k: int(v) for k, v in df[gov].primary_country.value_counts().head(6).items()}

# ------------------------------------------------------------------ design matrices
CAT, NUM = ["cause_group", "type_group"], ["n_asns", "n_locations", "n_countries", "has_asn_record"]
DUM = pd.get_dummies(df[CAT], drop_first=True).astype(float)
cols_cause = [c for c in DUM.columns if c.startswith("cause_group_")]
cols_type = [c for c in DUM.columns if c.startswith("type_group_")]
Xnum = df[NUM].values.astype(float)
desc = df.description.fillna("").str.lower()
flags = np.column_stack([(desc != "").astype(float),
                         desc.str.contains(r"\bexam|cheat").astype(float),
                         desc.str.contains("protest|election|unrest|demonstrat").astype(float)])
X = {"A": None,
     "B": DUM[cols_cause].values,
     "C": DUM[cols_cause + cols_type].values,
     "D": np.hstack([DUM[cols_cause + cols_type].values, Xnum])}
X["E"] = np.hstack([X["D"], flags])
NAMES = {"A": "Global mean (no predictors)", "B": "Cause only", "C": "Cause + type",
         "D": "Cause + type + scope (full linear model)", "E": "Full model + description flags (exploratory)",
         "GB": "Gradient boosting (same predictors as D)", "RF": "Random forest (same predictors as D)"}

# ------------------------------------------------------------------ folds shared by every model
rng = np.random.default_rng(SEED); uniq = np.unique(country)
FOLDS = []
for _ in range(REPS):
    fo = {g: i % K for i, g in enumerate(rng.permutation(uniq))}
    FOLDS.append(np.array([fo[g] for g in country]))

def ols_fit_predict(Xm, tr, te):
    if Xm is None: return np.full(te.sum(), y[tr].mean())
    A = np.hstack([Xm[tr], np.ones((tr.sum(), 1))])
    beta = np.linalg.lstsq(A, y[tr], rcond=1e-8)[0]          # SVD least squares: stable even if columns are collinear
    p = np.hstack([Xm[te], np.ones((te.sum(), 1))]) @ beta
    assert np.isfinite(p).all() and np.abs(p).max() < 30
    return p
def gb_fit_predict(Xm, tr, te):
    m = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0)
    return m.fit(Xm[tr], y[tr]).predict(Xm[te])
def rf_fit_predict(Xm, tr, te):
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
    return m.fit(Xm[tr], y[tr]).predict(Xm[te])
FIT = {"A": ols_fit_predict, "B": ols_fit_predict, "C": ols_fit_predict, "D": ols_fit_predict, "E": ols_fit_predict,
       "GB": gb_fit_predict, "RF": rf_fit_predict}
XM = dict(X, GB=X["D"], RF=X["D"])

def oof(key, fold_list):
    P = np.zeros((len(fold_list), n))
    for r, f in enumerate(fold_list):
        for i in range(K):
            tr, te = f != i, f == i
            P[r, te] = FIT[key](XM[key], tr, te)
    return P
def r2(p, m=None, w=None):
    m = np.ones(n, bool) if m is None else m
    w = np.ones(n) if w is None else w
    e, yy = p[m] - y[m], y[m]
    return 1 - np.sum(w[m] * e**2) / np.sum(w[m] * (yy - np.average(yy, weights=w[m]))**2)
def rep_stats(P, m=None):
    m = np.ones(n, bool) if m is None else m
    r = np.array([r2(p, m) for p in P]); mae = np.array([np.mean(np.abs(p[m] - y[m])) for p in P])
    w2 = np.array([np.mean(np.abs(p[m] - y[m]) < np.log(2)) for p in P])
    med_h = np.array([np.median(np.abs(np.exp(np.clip(p[m], -10, 15)) - np.exp(y[m]))) for p in P])
    return dict(R2_mean=r.mean(), R2_sd=r.std(ddof=1), MAE_mean=mae.mean(), MAE_sd=mae.std(ddof=1),
                within2x_mean=w2.mean(), within2x_sd=w2.std(ddof=1), median_abs_err_h=med_h.mean())

ep_rows = {e: np.where(epi == e)[0] for e in np.unique(epi)}; ep_ids = np.array(list(ep_rows))
brng = np.random.default_rng(SEED + 1)
BOOT = [np.concatenate([ep_rows[e] for e in brng.choice(ep_ids, len(ep_ids))]) for _ in range(NBOOT)]   # shared draws
def boot_ci_R2(P):
    vals = np.array([np.mean([1 - np.sum((p[ix] - y[ix])**2) / np.sum((y[ix] - y[ix].mean())**2) for p in P]) for ix in BOOT])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# ------------------------------------------------------------------ 1. main result: unseen countries
PRED, MAIN = {}, {}
for key in ["A", "B", "C", "D", "E", "GB", "RF"]:
    PRED[key] = oof(key, FOLDS)
    s = rep_stats(PRED[key]); s["R2_ci"] = boot_ci_R2(PRED[key]); s["name"] = NAMES[key]; MAIN[key] = s
R["main"] = MAIN

# ------------------------------------------------------------------ 2. leakage: random rows vs unseen countries
rowfolds = []
for s_ in range(REPS):
    kf = KFold(K, shuffle=True, random_state=s_); f = np.zeros(n, int)
    for i, (_, te) in enumerate(kf.split(np.arange(n))): f[te] = i
    rowfolds.append(f)
LEAK = {}
for key in ["A", "B", "D", "GB"]:
    P = oof(key, rowfolds); s = rep_stats(P); s["R2_ci"] = boot_ci_R2(P); LEAK[key] = s
R["random_row"] = LEAK

# ------------------------------------------------------------------ 3. Iraq checks
iq = country == "IQ"
iq_test = {}
for key in ["A", "D"]:
    tr, te = ~iq, iq
    p = FIT[key](XM[key], tr, te)
    iq_test[key] = dict(MAE=float(np.mean(np.abs(p - y[te]))), within2x=float(np.mean(np.abs(p - y[te]) < np.log(2))),
                        median_pred_h=float(np.exp(np.median(p))), median_true_h=float(np.exp(np.median(y[te]))))
R["iraq_heldout"] = iq_test
keep = ~iq; idx = np.where(keep)[0]; sub_uniq = np.unique(country[keep]); r_ = np.random.default_rng(SEED + 2)
noiraq = {k: [] for k in ["A", "B", "D"]}
for _ in range(REPS):
    fo = {g: i % K for i, g in enumerate(r_.permutation(sub_uniq))}; f = np.array([fo[g] for g in country[idx]])
    for key in noiraq:
        p = np.zeros(len(idx)); yy = y[idx]
        for i in range(K):
            tr, te = f != i, f == i
            Xs = None if X[key] is None else X[key][idx]
            if Xs is None: p[te] = yy[tr].mean()
            else:
                A = np.hstack([Xs[tr], np.ones((tr.sum(), 1))]); b = np.linalg.lstsq(A, yy[tr], rcond=1e-8)[0]
                p[te] = np.hstack([Xs[te], np.ones((te.sum(), 1))]) @ b
        noiraq[key].append(1 - np.sum((p - yy)**2) / np.sum((yy - yy.mean())**2))
R["no_iraq_R2"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in noiraq.items()}
R["no_iraq_rows"] = int(keep.sum())

# ------------------------------------------------------------------ 4. by cause (full model, mean prediction over repeats)
full, base = PRED["D"].mean(0), PRED["B"].mean(0)
glob = PRED["A"].mean(0)
bc = []
for c, g in df.groupby("cause_group"):
    mk = (df.cause_group == c).values; eps = np.unique(epi[mk]); er = np.abs(full - y)
    rows_c = {e: np.where((epi == e) & mk)[0] for e in eps}; b = np.random.default_rng(SEED + 3); bs = []
    for _ in range(NBOOT):
        ix = np.concatenate([rows_c[e] for e in b.choice(eps, len(eps))]); bs.append(er[ix].mean())
    bc.append(dict(cause=c, rows=int(mk.sum()), episodes=int(len(eps)), median_h=float(np.median(dur[mk])),
                   MAE_full=float(er[mk].mean()), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                   MAE_causeonly=float(np.abs(base - y)[mk].mean()), MAE_globalmean=float(np.abs(glob - y)[mk].mean()),
                   within2x=float(np.mean(er[mk] < np.log(2))), typical_factor=float(np.exp(er[mk].mean()))))
bc = sorted(bc, key=lambda d: d["MAE_full"]); R["by_cause"] = bc
pd.DataFrame(bc).to_csv("table_by_cause.csv", index=False)
R["MAE_global_all"] = float(np.abs(glob - y).mean())

# ------------------------------------------------------------------ 5. sensitivity (full model D)
P = PRED["D"]; w = df.sample_weight.values
def ms(vals): return [float(np.mean(vals)), float(np.std(vals, ddof=1))]
sens = {"All 862 records (main result)": ms([r2(p) for p in P]),
        "Each episode counted once (weights = 1/episode size)": ms([r2(p, None, w) for p in P]),
        "Excluding 7 long records with a shared end time": ms([r2(p, (df.flag_shared_endpoint == 0).values) for p in P]),
        "Excluding all 21 records longer than 30 days": ms([r2(p, (df.flag_long_over_30d == 0).values) for p in P]),
        "Government-directed records only": ms([r2(p, gov) for p in P]),
        "Non-government records only": ms([r2(p, ~gov) for p in P])}
R["sensitivity"] = sens
R["E_sens"] = {"gov_only": ms([r2(p, gov) for p in PRED["E"]]), "non_gov": ms([r2(p, ~gov) for p in PRED["E"]])}

# ------------------------------------------------------------------ 6. exam description exploration
says_exam = flags[:, 1] == 1
ge = df[gov].assign(says_exam=says_exam[gov])
ex = ge.groupby("says_exam").duration_hours.agg(["count", "median", "mean", lambda s: s.quantile(.25), lambda s: s.quantile(.75), "max"])
ex.columns = ["n", "median_h", "mean_h", "q1_h", "q3_h", "max_h"]; ex.to_csv("table_exam.csv")
R["exam_table"] = ex.round(2).reset_index().to_dict("records")
R["gov_exam_share"] = float(ge.says_exam.mean()); R["gov_desc_missing_share"] = float((ge.description.fillna("") == "").mean())
R["desc_missing_share_all"] = float((desc == "").mean())
R["exam_by_country"] = {k: int(v) for k, v in ge[ge.says_exam].primary_country.value_counts().head(5).items()}
R["exam_n_countries"] = int(ge[ge.says_exam].primary_country.nunique())
R["exam_nonexam_countries_nonexam"] = int(ge[~ge.says_exam].primary_country.nunique())

# ------------------------------------------------------------------ 7. coefficients of final linear model D (all data)
A = np.hstack([X["D"], np.ones((n, 1))]); beta = np.linalg.lstsq(A, y, rcond=1e-8)[0]
names_D = cols_cause + cols_type + NUM
R["coefs"] = [dict(feature=nm, coef=float(b), multiplier=float(np.exp(b))) for nm, b in zip(names_D, beta[:-1])]
R["intercept"] = float(beta[-1]); R["intercept_h"] = float(np.exp(beta[-1]))
R["in_sample_R2_D"] = float(r2(A @ beta))

json.dump(R, open("paper_results.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ figures (no titles/captions in the image files, per journal rule)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
SZ = (6.5, 3.8); C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
names = dict(pytz.country_names)
# Figure 1: records per country
top = cc.head(12)[::-1]
fig, ax = plt.subplots(figsize=SZ)
labs = [names.get(k, k) if k != "UNK" else "Unresolved" for k in top.index]
labs = [l.split(",")[0] for l in labs]
ax.barh(labs, top.values, color=[C2 if k == "IQ" else C1 for k in top.index])
for i, v in enumerate(top.values): ax.text(v + 3, i, str(int(v)), va="center", fontsize=8)
ax.set_xlabel("Number of Outage Records"); ax.set_ylabel("Country"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("Figure_1.png", dpi=300); plt.close(fig)
# Figure 2: duration histogram
fig, ax = plt.subplots(figsize=SZ)
bins = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 36)
ax.hist(dur[gov & says_exam], bins=bins, color=C2, alpha=.85, label="Government-Directed, Exam Mentioned")
ax.hist(dur[~(gov & says_exam)], bins=bins, color=C1, alpha=.65, label="All Other Records")
ax.set_xscale("log"); ax.set_xlabel("Outage Duration (hours, log scale)"); ax.set_ylabel("Number of Records")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("Figure_2.png", dpi=300); plt.close(fig)
# Figure 3: random-row vs unseen-country R2
fig, ax = plt.subplots(figsize=SZ); keys = ["B", "D", "GB"]; lab3 = ["Cause Only\n(Linear)", "Full Linear Model", "Gradient Boosting"]
xs = np.arange(len(keys)); wd = .36
ax.bar(xs - wd/2, [LEAK[k]["R2_mean"] for k in keys], wd, yerr=[LEAK[k]["R2_sd"] for k in keys], color=C3, capsize=3, label="Random Rows (Leaky)")
ax.bar(xs + wd/2, [MAIN[k]["R2_mean"] for k in keys], wd, yerr=[MAIN[k]["R2_sd"] for k in keys], color=C1, capsize=3, label="Unseen Countries")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(xs); ax.set_xticklabels(lab3); ax.set_ylabel("R² (mean ± SD of 20 repeats)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("Figure_3.png", dpi=300); plt.close(fig)
# Figure 4: error by cause
b = pd.DataFrame(bc).iloc[::-1]
fig, ax = plt.subplots(figsize=SZ)
ax.barh(b.cause.str.replace("_", " ").str.title().replace({"Weather Natural": "Weather or Natural Disaster", "Government Directed": "Government-Directed"}), b.MAE_full, xerr=[b.MAE_full - b.ci_lo, b.ci_hi - b.MAE_full], color=C1, capsize=2)
ax.axvline(R["MAE_global_all"], color=C2, ls="--", lw=1, label="Overall Error of the Global-Mean Guess")
ax.set_xlabel("Mean Absolute Error (log hours; lower is more predictable)"); ax.set_xlim(0, 3.4); ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("Figure_4.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ printed summary
print(json.dumps({k: R[k] for k in ["n_rows", "n_episodes", "start_min", "start_max", "n_countries", "top_countries", "iraq_share", "top3_share",
      "share_in_multi_episodes", "dur_median", "dur_q1", "dur_q3", "dur_mean", "dur_max"]}, indent=1))
for k, v in MAIN.items(): print(f"{k:3s} {v['name'][:45]:45s} R2={v['R2_mean']:.3f} sd={v['R2_sd']:.3f} CI=({v['R2_ci'][0]:.3f},{v['R2_ci'][1]:.3f}) MAE={v['MAE_mean']:.3f} within2x={v['within2x_mean']:.1%} medErr={v['median_abs_err_h']:.2f}h")
print("random-row:", {k: (round(v['R2_mean'], 3), round(v['R2_sd'], 3)) for k, v in LEAK.items()})
print("Iraq held out:", iq_test); print("No-Iraq R2:", R["no_iraq_R2"], R["no_iraq_rows"])
print("sensitivity:", {k: [round(x, 3) for x in v] for k, v in R["sensitivity"].items()}, R["E_sens"])
print(ex.round(2)); print("exam share of gov", R["gov_exam_share"], "missing", R["gov_desc_missing_share"], R["exam_by_country"], R["exam_n_countries"])
print(pd.DataFrame(bc).round(2).to_string(index=False))
