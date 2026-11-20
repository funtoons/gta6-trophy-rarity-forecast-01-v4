"""GTA 6 trophy rarity forecast v4 (Python 3.13 stdlib only).

Forecasts, for every trophy, five quantiles of its Exophase earn rate at
the case target, in percent, scored by pinball loss on the logit scale.

Design (written fresh for this entry):
  * Plateau-collapse: consecutive reads with identical players and rates
    are one observation (Exophase refreshes on its own schedule; a frozen
    page is not fresh evidence). The collapsed point sits at the run's
    first read time.
  * Per-trophy ensemble in logit space of four estimators: persistence
    (last rate), a log-time WLS trend, an earners/players decomposition
    (log earners and log players extrapolated separately, rate = ratio),
    and a real-time WLS trend. Ensemble weights come from a
    leave-future-out backtest (fit up to checkpoint-24h, scored on the
    last 24h), pooled per trophy group and shrunk to global weights.
  * Flood insurance (the v4 differentiator): the history already holds
    Exophase refresh jumps, so a large players jump or players growing
    faster than earners raises a flood score. When it is high the median
    is anchored toward persistence and the LOWER tail is widened
    asymmetrically -- a late flood can only push rates down, never up.
  * Soft structural priors, which are logical necessities on Exophase
    counts: platinum earners are a subset of every trophy's earners;
    chapter trophies nest (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final mission);
    100% completion nests inside the final mission. Enforced on every
    quantile by cumulative minima.
  * Adaptive spread: base random-walk term in horizon, plus estimator
    disagreement, slope standard error times extrapolation distance,
    and backtest residual; floored and capped. Lower tail gets the
    flood multiplier; upper tail does not.
  * A trophy at 100% stays near 100% (it survived a 6x flood in the
    practice history); its upper tail is pinned tight.

Deterministic, offline, no wall clock, no randomness, no absolute paths.
Reads every *.in in MAKEFUN_INPUT_DIR (else /input), writes *.out to
MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Tunable configuration; the sweep harness overrides entries of CFG.
CFG = {
    "bt_window_h": 24.0,    # backtest holdout window (hours before checkpoint)
    "bt_power": 4.0,        # weight power on backtest error
    "bt_eps": 0.05,         # error floor in weight formula
    "halflife_h": 24.0,     # recency half-life for WLS weights (hours)
    "group_shrink": 0.5,    # weight on group weights vs global weights
    "pers_floor": 0.15,     # minimum ensemble weight on persistence
    "flood_trig": 0.5,      # flood score above which insurance activates
    "flood_w": 1.0,         # how far the median is pulled to persistence
    "low_mult": 2.0,        # extra lower-tail sigma per unit flood score
    "sigma0": 0.5,          # random-walk sigma per sqrt(day)
    "sigma_floor": 0.8,     # minimum total sigma (logit)
    "sigma_cap": 3.0,       # maximum total sigma (logit)
    "clamp": 2.5,           # median clamp vs last logit, at 4.52-day horizon
    "def_pers": 0.45,       # default weights when no backtest possible
    "def_logtime": 0.25,
    "def_decomp": 0.20,
    "def_realtime": 0.10,
    "trend_shrink": 0.4,    # shrink ensemble move toward persistence (0..1)
    "est_clamp": 1.5,       # clamp each trend pred vs last logit
    "rt_slope": 0.03,       # real-time slope clamp, logit per hour
    "low_tail": 1.5,        # lower-tail sigma multiplier (flood cover)
    "hi_tail": 0.75,        # upper-tail sigma multiplier
}

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000.0
    return int(v) if float(v).is_integer() else v


def parse_time(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(dt, release):
    return (dt - release).total_seconds() / 3600.0


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, pred_se, resid)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / sw
    ym = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ym - b * xm
    n = len(xs)
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / n) if sw else 0.0
    s2 = max(s2, 1e-6)

    def pred_se(x0):
        var = s2 * (1.0 / max(n, 1) + (x0 - xm) ** 2 / sxx)
        return math.sqrt(max(var, 0.0))

    return (a, b, pred_se, resid)


def recency_w(t_h, t_ref_h, halflife):
    age = max(t_ref_h - t_h, 0.0)
    return 0.5 ** (age / halflife)


def collapse(history):
    """Merge consecutive identical reads into one point at run start."""
    pts = []
    for h in history:
        rates = h.get("rates") or []
        cur = {"t": parse_time(h["at"]), "players": h.get("players_tracked"),
               "earned": h.get("trophies_earned"), "rates": list(rates)}
        if pts:
            prev = pts[-1]
            if (prev["players"] == cur["players"]
                    and prev["rates"] == cur["rates"]):
                continue
        pts.append(cur)
    return pts


def trophy_series(pts, idx):
    """Collapsed (datetime, logit, players, earners) for one trophy."""
    out = []
    for p in pts:
        r = p["rates"][idx] if idx < len(p["rates"]) else None
        if r is None:
            continue
        pl = p["players"] or 0
        earn = (r / 100.0) * pl if pl else 0.0
        out.append((p["t"], logit(r), pl, earn))
    return out


def fit_estimators(series, t_now, t_target, release, cfg):
    """Fit the four estimators on a series. Returns dict name -> (pred, se)."""
    est = {}
    if not series:
        return est
    last_lg = series[-1][1]
    est["pers"] = (last_lg, 0.0)
    if len(series) < 3:
        return est
    hl = cfg["halflife_h"]
    t0 = series[0][0]
    span_h = (series[-1][0] - t0).total_seconds() / 3600.0
    if span_h < 2.0:
        return est
    tnow_h = (t_now - t0).total_seconds() / 3600.0
    # --- log-time trend ---
    xs, ys, ws = [], [], []
    for (t, lg, pl, er) in series:
        h = hours_since_release(t, release)
        xs.append(math.log1p(max(h, 0.0)))
        ys.append(lg)
        ws.append(recency_w((t - t0).total_seconds() / 3600.0, tnow_h, hl))
    fit = wls_line(xs, ys, ws)
    if fit is not None:
        a, b, se_fn, _ = fit
        b = max(-2.5, min(2.5, b))
        xt = math.log1p(max(hours_since_release(t_target, release), 0.0))
        est["logtime"] = (a + b * xt, se_fn(xt))
    # --- earners / players decomposition ---
    xs2, ye, yp, ws2 = [], [], [], []
    ok = True
    for (t, lg, pl, er) in series:
        if pl is None or pl <= 0:
            ok = False
            break
        h = hours_since_release(t, release)
        xs2.append(math.log1p(max(h, 0.0)))
        ye.append(math.log(max(er, 0.5)))
        yp.append(math.log(max(pl, 1.0)))
        ws2.append(recency_w((t - t0).total_seconds() / 3600.0, tnow_h, hl))
    if ok:
        fe = wls_line(xs2, ye, ws2)
        fp = wls_line(xs2, yp, ws2)
        if fe is not None and fp is not None:
            ae, be, se_e, _ = fe
            ap, bp, se_p, _ = fp
            be = max(0.0, min(3.0, be))
            bp = max(0.0, min(3.0, bp))
            xt = math.log1p(max(hours_since_release(t_target, release), 0.0))
            rate = math.exp((ae + be * xt) - (ap + bp * xt)) * 100.0
            se = math.sqrt(se_e(xt) ** 2 + se_p(xt) ** 2)
            est["decomp"] = (logit(rate), se)
    # --- real-time trend ---
    xs3, ys3, ws3 = [], [], []
    for (t, lg, pl, er) in series:
        xs3.append((t - t0).total_seconds() / 3600.0)
        ys3.append(lg)
        ws3.append(recency_w((t - t0).total_seconds() / 3600.0, tnow_h, hl))
    fit3 = wls_line(xs3, ys3, ws3)
    if fit3 is not None:
        a, b, se_fn, _ = fit3
        b = max(-cfg["rt_slope"], min(cfg["rt_slope"], b))
        xt = (t_target - t0).total_seconds() / 3600.0
        est["realtime"] = (a + b * xt, se_fn(xt))
    return est


def group_of(trophy, last_rate):
    if trophy.get("tier") == "platinum":
        return "plat"
    if last_rate is not None and last_rate >= 99.9:
        return "full"
    tid = trophy.get("id", 0)
    if 2 <= tid <= 14:
        return "story"
    return "grind"


def backtest_weights(all_series, t_cut, t_now, release, cfg, groups):
    """Leave-future-out weights per group + global.

    Fits use points with t <= t_cut; holdout points are t_cut < t <= t_now.
    Returns (group_w, global_w) or (None, default) when impossible.
    """
    names = ["pers", "logtime", "decomp", "realtime"]
    default = {"pers": cfg["def_pers"], "logtime": cfg["def_logtime"],
               "decomp": cfg["def_decomp"], "realtime": cfg["def_realtime"]}
    errs = {g: {n: [] for n in names} for g in ("plat", "full", "story", "grind")}
    gerrs = {n: [] for n in names}
    any_holdout = False
    for series, grp in zip(all_series, groups):
        fit_pts = [s for s in series if s[0] <= t_cut]
        hold = [s for s in series if t_cut < s[0] <= t_now]
        if len(fit_pts) < 2 or not hold:
            continue
        any_holdout = True
        for h in hold:
            est = fit_estimators(fit_pts, t_cut, h[0], release, cfg)
            for n in names:
                if n in est:
                    e = abs(est[n][0] - h[1])
                    errs[grp][n].append(e)
                    gerrs[n].append(e)

    def to_weights(pool):
        tot = {}
        for n in names:
            es = pool.get(n, [])
            if not es:
                continue
            m = sum(es) / len(es)
            tot[n] = 1.0 / (m + cfg["bt_eps"]) ** cfg["bt_power"]
        s = sum(tot.values())
        if s <= 0:
            return None
        return {n: v / s for n, v in tot.items()}

    gw = to_weights(gerrs)
    if gw is None or not any_holdout:
        return None, default
    group_w = {}
    for g in errs:
        w = to_weights(errs[g])
        if w is None:
            group_w[g] = dict(gw)
        else:
            sh = cfg["group_shrink"]
            merged = {}
            for n in names:
                merged[n] = sh * w.get(n, 0.0) + (1 - sh) * gw.get(n, 0.0)
            s = sum(merged.values())
            group_w[g] = {n: v / s for n, v in merged.items()} if s > 0 else dict(gw)
    return group_w, gw


def flood_signals(pts):
    """Flood/dilution risk from the collapsed history.

    A players jump alone is not a flood -- launch growth is steep too.
    What matters is DILUTION: players growing faster than earners, so
    rates fall. Measured per trophy at every refresh jump (median over
    trophies, robust) and over the whole history. Returns
    (flood_score 0..1, max_jump, dil_ratio).
    """
    max_jump = 1.0
    for a, b in zip(pts, pts[1:]):
        pa, pb = a["players"], b["players"]
        if pa and pb and pa > 0:
            max_jump = max(max_jump, pb / pa)
    dil_jump = 1.0
    for a, b in zip(pts, pts[1:]):
        pa, pb = a["players"], b["players"]
        if not pa or not pb or pa <= 0:
            continue
        gp = pb / pa
        if gp <= 1.0:
            continue
        dils = []
        for ra, rb in zip(a["rates"], b["rates"]):
            if ra is None or rb is None or ra <= 0 or rb <= 0:
                continue
            ea, eb = ra * pa, rb * pb
            if ea > 0:
                dils.append(gp / (eb / ea))
        if dils:
            dils.sort()
            dil_jump = max(dil_jump, dils[len(dils) // 2])
    dil = 1.0
    if len(pts) >= 2:
        p0, p1 = pts[0]["players"], pts[-1]["players"]
        e0, e1 = pts[0]["earned"], pts[-1]["earned"]
        if p0 and p1 and e0 and e1 and e0 > 0 and p0 > 0:
            gp = p1 / p0
            ge = e1 / e0
            if ge > 0:
                dil = gp / ge
    s_jump = max(0.0, min(1.0, math.log(dil_jump) / math.log(3.0))) if dil_jump > 1 else 0.0
    s_dil = max(0.0, min(1.0, math.log(dil) / math.log(3.0))) if dil > 1 else 0.0
    flood = max(s_jump, s_dil)
    return flood, max_jump, dil


def forecast_case(case, cfg=None):
    cfg = dict(CFG if cfg is None else cfg)
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = case.get("horizon_minutes", 0) or 0
    horizon_days = max(horizon_min / 1440.0, 0.0)
    t_now = parse_time(case["at"])
    t_target = parse_time(case["target"])
    release = RELEASE
    try:
        rd = (case.get("game") or {}).get("release_date")
        if rd:
            release = datetime.fromisoformat(rd + "T00:00:00+00:00")
    except Exception:
        release = RELEASE

    pts = collapse(history)
    n = len(trophies)
    last_rates = []
    for i in range(n):
        r = None
        for p in reversed(pts):
            if i < len(p["rates"]) and p["rates"][i] is not None:
                r = p["rates"][i]
                break
        last_rates.append(r if r is not None else 1.0)

    all_series = [trophy_series(pts, i) for i in range(n)]
    groups = [group_of(t, last_rates[i]) for i, t in enumerate(trophies)]

    group_w = None
    global_w = {"pers": cfg["def_pers"], "logtime": cfg["def_logtime"],
                "decomp": cfg["def_decomp"], "realtime": cfg["def_realtime"]}
    if pts and len(pts) >= 4:
        span_h = (pts[-1]["t"] - pts[0]["t"]).total_seconds() / 3600.0
        if span_h >= cfg["bt_window_h"] + 2:
            t_cut = datetime.fromtimestamp(
                t_now.timestamp() - cfg["bt_window_h"] * 3600.0,
                tz=timezone.utc)
            gw, gl = backtest_weights(all_series, t_cut, t_now, release, cfg, groups)
            if gw is not None:
                group_w = gw
                global_w = gl

    flood, max_jump, dil = flood_signals(pts)
    flood_active = flood if flood >= cfg["flood_trig"] else 0.0

    clamp_eff = max(0.35, cfg["clamp"] * math.sqrt(max(horizon_days, 0.01) / 4.52))

    medians = []
    sig_lo = []
    sig_hi = []
    for i, trophy in enumerate(trophies):
        series = all_series[i]
        last = last_rates[i]
        last_lg = logit(last)
        w = dict(global_w)
        if group_w is not None:
            w = dict(group_w.get(groups[i], global_w))
        if "pers" in w:
            w["pers"] = max(w["pers"], cfg["pers_floor"])
            s = sum(w.values())
            w = {k: v / s for k, v in w.items()}
        est = fit_estimators(series, t_now, t_target, release, cfg)
        # Clamp every trend prediction to a sane move from the last rate:
        # a linear-in-real-time fit over 4.5 days can otherwise run away.
        for name in ("logtime", "decomp", "realtime"):
            if name in est:
                p, s = est[name]
                est[name] = (max(last_lg - cfg["est_clamp"],
                                 min(last_lg + cfg["est_clamp"], p)), s)
        num, den, preds = 0.0, 0.0, []
        for name, wt in w.items():
            if name in est:
                num += wt * est[name][0]
                den += wt
                preds.append(est[name][0])
        med = num / den if den > 0 else last_lg
        # Anchor the median toward persistence: rates can crash 6x in one
        # refresh (dilution) but can only climb as fast as players play.
        if "pers" in est:
            med = est["pers"][0] + cfg["trend_shrink"] * (med - est["pers"][0])
        if flood_active > 0 and "pers" in est:
            pull = cfg["flood_w"] * flood_active
            med = (1 - pull) * med + pull * est["pers"][0]
        med = max(last_lg - clamp_eff, min(last_lg + clamp_eff, med))
        sig_base = cfg["sigma0"] * math.sqrt(horizon_days)
        disag = 0.0
        if len(preds) >= 2:
            m = sum(preds) / len(preds)
            disag = math.sqrt(sum((p - m) ** 2 for p in preds) / len(preds))
            disag = min(disag, 0.8)
        se = 0.0
        for name in ("logtime", "decomp", "realtime"):
            if name in est:
                se = max(se, est[name][1])
        se = min(se, 0.8)
        resid = 0.25 if group_w is not None else 0.0
        sig = math.sqrt(sig_base ** 2 + disag ** 2 + se ** 2 + resid ** 2)
        sig = max(cfg["sigma_floor"], min(cfg["sigma_cap"], sig))
        lo = sig * cfg["low_tail"] * (1.0 + cfg["low_mult"] * flood_active)
        hi = sig * cfg["hi_tail"]
        if last >= 99.9:
            med = min(med, logit(99.99))
            med = max(med, logit(99.0))
            hi = 0.15
            lo = max(lo, 0.8)
        if last <= 0.01:
            hi = max(hi, 0.8)
        medians.append(med)
        sig_lo.append(min(lo, cfg["sigma_cap"] * 1.6))
        sig_hi.append(hi)

    rows_q = []
    for i in range(n):
        qs = [inv_logit(medians[i] + z * (sig_lo[i] if z < 0 else sig_hi[i]))
              for z in Z]
        qs = [round4(min(99.99, max(0.01, q))) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows_q.append(qs)

    plat_idx = [i for i, t in enumerate(trophies)
                if t.get("tier") == "platinum"]
    for pi in plat_idx:
        for k in range(5):
            cap = min(rows_q[j][k] for j in range(n) if j != pi) if n > 1 else rows_q[pi][k]
            if rows_q[pi][k] > cap:
                rows_q[pi][k] = cap
    by_id = {t.get("id"): i for i, t in enumerate(trophies)}
    for chain in ([3, 4, 5, 6, 7], [7, 14]):
        idxs = [by_id[c] for c in chain if c in by_id]
        for k in range(5):
            run = None
            for j in idxs:
                if run is None:
                    run = rows_q[j][k]
                else:
                    run = min(run, rows_q[j][k])
                    rows_q[j][k] = run
    for i in range(n):
        for k in range(1, 5):
            if rows_q[i][k] < rows_q[i][k - 1]:
                rows_q[i][k] = rows_q[i][k - 1]

    rows = [{"id": t["id"], "q": rows_q[i]} for i, t in enumerate(trophies)]
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        fc = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(fc, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
