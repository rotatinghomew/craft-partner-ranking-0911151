"""候选排序：小样本收缩、共同承办方借力与筛选后扩容的回看。

该模块只依赖快照内容，不读取网络或本地路径。所有推断口径、验证折、随机
种子和资料指纹都随成果保存，便于复现。主分析以 rounds.supported 为暴露
量；nominations.additional_slots 是筛选后的扩容，单列回看与敏感性分析，
只有样本外表现明确更好时才并入暴露量。
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp

METHOD = "candidate-yield-v1"
VERSION = "1"

DEFAULT_CONFIG = {
    "draws": 4000,
    "seed": 20261115,
    "portfolio_size": None,
    "slots_per_candidate": None,
    "tie_threshold": 0.8,
    "permutations": 2000,
    "validation_folds": 5,
    "validation_tolerance": 0.01,
    "prior_sd": 2.5,
    "conversion_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
}

# 只有在既有届次上验证过才使用这些结构。名称写入成果，不能复用旧方法名。
STRUCTURES = (
    {"name": "region", "project": False, "organizer": False, "region": True, "category": False},
    {"name": "project", "project": True, "organizer": False, "region": False, "category": False},
    {"name": "organizer", "project": False, "organizer": True, "region": False, "category": False},
    {"name": "project-organizer", "project": True, "organizer": True, "region": False, "category": False},
    {"name": "project-organizer-region", "project": True, "organizer": True, "region": True, "category": False},
    {"name": "project-organizer-region-category", "project": True, "organizer": True, "region": True, "category": True},
)

_FIT_CACHE = {}
_FIT_CACHE_LIMIT = 48


class _Design:
    __slots__ = ("X", "Zp", "Zo", "names", "unseen_project", "unseen_organizer")

    def __init__(self, X, Zp, Zo, names, unseen_project, unseen_organizer):
        self.X = X
        self.Zp = Zp
        self.Zo = Zo
        self.names = names
        self.unseen_project = unseen_project
        self.unseen_organizer = unseen_organizer


def _r(value, digits=6):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("分析结果必须是有限数值")
    return round(value, digits)


def _levels(rows, key):
    return sorted({row[key] for row in rows})


def _design(rows, levels, config, center):
    """按训练时固定的层级生成设计矩阵；未见层级取先验均值。"""
    n = len(rows)
    cols = [np.ones(n)]
    names = ["intercept"]
    if config["trend"] and center is not None:
        cols.append(np.array([row["season"] - center for row in rows], dtype=float))
        names.append("season_trend")
    for group in (levels["region"][1:] if config["region"] else ()):
        cols.append(np.array([1.0 if row["region"] == group else 0.0 for row in rows]))
        names.append("region:" + group)
    for group in (levels["category"][1:] if config["category"] else ()):
        cols.append(np.array([1.0 if row["category"] == group else 0.0 for row in rows]))
        names.append("category:" + group)
    X = np.column_stack(cols) if cols else np.zeros((n, 0))

    project_index = {name: index for index, name in enumerate(levels["project"])}
    organizer_index = {name: index for index, name in enumerate(levels["organizer"])}
    if config["project"]:
        Zp = np.zeros((n, len(project_index)))
        unseen_project = np.zeros(n)
        for row_index, row in enumerate(rows):
            column = project_index.get(row["project"])
            if column is None:
                unseen_project[row_index] = 1.0
            else:
                Zp[row_index, column] = 1.0
    else:
        Zp = np.zeros((n, 0))
        unseen_project = np.zeros(n)
    if config["organizer"]:
        Zo = np.zeros((n, len(organizer_index)))
        unseen_organizer = np.zeros(n)
        for row_index, row in enumerate(rows):
            column = organizer_index.get(row["organizer"])
            if column is None:
                unseen_organizer[row_index] = 1.0
            else:
                Zo[row_index, column] = 1.0
    else:
        Zo = np.zeros((n, 0))
        unseen_organizer = np.zeros(n)
    return _Design(X, Zp, Zo, names, unseen_project, unseen_organizer)


def _panel(snapshot, exposure="supported", additional_weight=None):
    """把 rounds、projects、nominations 对齐成项目-届次面板。

    主口径 exposure="supported"；exposure="expanded" 把全部追加名额并入
    暴露量；additional_weight 用于回看时按比例折算扩容，取值 [0,1]。
    achieved 始终对应原始 supported，追加名额不能改变已观测达成数。
    """
    rounds = {}
    for row in snapshot["tables"]["rounds"]:
        rounds[row["id"]] = dict(row)
    corrections = sorted(
        snapshot["tables"]["corrections"], key=lambda row: (row["released_at"], row["revision"])
    )
    applied = []
    for correction in corrections:
        if correction["target"] in rounds:
            if correction["field"] not in {"supported", "achieved", "budget"}:
                raise ValueError("不支持的复核字段")
            rounds[correction["target"]][correction["field"]] = correction["value"]
            applied.append(correction["id"])
    projects = {row["id"]: row for row in snapshot["tables"]["projects"]}
    nominations = {(row["project"], row["season"]): row for row in snapshot["tables"]["nominations"]}
    if exposure not in {"supported", "expanded"}:
        raise ValueError("未知暴露口径")
    if additional_weight is None:
        additional_weight = 1.0 if exposure == "expanded" else 0.0
    if not 0.0 <= float(additional_weight) <= 1.0:
        raise ValueError("扩容折算比例必须在 [0,1]")

    rows = []
    for round_row in sorted(rounds.values(), key=lambda row: (row["season"], row["project"])):
        project = projects.get(round_row["project"])
        if project is None:
            raise ValueError("届次记录缺少对应项目")
        if round_row["supported"] <= 0 or not 0 <= round_row["achieved"] <= round_row["supported"]:
            raise ValueError("项目达成数与支持数矛盾")
        nomination = nominations.get((round_row["project"], round_row["season"]))
        additional = int(nomination["additional_slots"]) if nomination else 0
        if additional < 0:
            raise ValueError("追加名额不能为负")
        exposure_value = float(round_row["supported"]) + float(additional_weight) * additional
        if exposure_value <= 0:
            raise ValueError("暴露量必须为正")
        rows.append(
            {
                "project": round_row["project"],
                "season": int(round_row["season"]),
                "organizer": round_row["organizer"],
                "region": project["region"],
                "category": project["category"],
                "supported": int(round_row["supported"]),
                "additional_slots": additional,
                "achieved": int(round_row["achieved"]),
                "chosen": None if nomination is None else bool(nomination["chosen"]),
                "known_before": None if nomination is None else nomination["known_before"],
                "expanded": int(round_row["supported"]) + additional,
                "N": exposure_value,
            }
        )
    return rows, applied


def _fit(rows, config, exposure):
    """带弱先验的二项分层模型；方差分量用边际似然选择。"""
    if not rows:
        raise ValueError("没有可用于估计的届次资料")
    levels = {key: _levels(rows, key) for key in ("project", "organizer", "region", "category")}
    season_values = {row["season"] for row in rows}
    center = float(np.mean([row["season"] for row in rows])) if len(season_values) > 1 else None
    config = {**config, "trend": bool(config.get("trend", True)) and center is not None}
    key = (
        config["name"],
        bool(config["project"]),
        bool(config["organizer"]),
        bool(config["region"]),
        bool(config["category"]),
        config["trend"],
        float(config["prior_sd"]),
        exposure,
        tuple(levels["project"]),
        tuple(levels["organizer"]),
        tuple(levels["region"]),
        tuple(levels["category"]),
        tuple(
            (
                row["project"],
                row["season"],
                row["organizer"],
                row["region"],
                row["category"],
                round(row["N"], 6),
                row["achieved"],
            )
            for row in rows
        ),
    )
    cached = _FIT_CACHE.get(key)
    if cached is not None:
        return cached

    y = np.array([row["achieved"] for row in rows], dtype=float)
    n = np.array([row["N"] for row in rows], dtype=float)
    design = _design(rows, levels, config, center)
    X, Zp, Zo = design.X, design.Zp, design.Zo
    fixed = X.shape[1]
    project_count = Zp.shape[1]
    organizer_count = Zo.shape[1]
    total = fixed + project_count + organizer_count
    project_start = fixed
    organizer_start = fixed + project_count
    prior_precision = 1.0 / float(config["prior_sd"]) ** 2

    def unpack(w):
        return w[:fixed], w[project_start:organizer_start], w[organizer_start:]

    def objective(w, project_sd, organizer_sd):
        beta, project_effect, organizer_effect = unpack(w)
        eta = X @ beta
        if project_count:
            eta = eta + Zp @ project_effect
        if organizer_count:
            eta = eta + Zo @ organizer_effect
        value = float(np.sum(n * np.logaddexp(0.0, eta) - y * eta))
        value += 0.5 * prior_precision * float(beta @ beta)
        residual = n * expit(eta) - y
        gradient = np.concatenate(
            [X.T @ residual, Zp.T @ residual if project_count else np.zeros(0), Zo.T @ residual if organizer_count else np.zeros(0)]
        )
        gradient[:fixed] += prior_precision * beta
        if project_count:
            value += 0.5 * float(project_effect @ project_effect) / project_sd**2
            gradient[project_start:organizer_start] += project_effect / project_sd**2
        if organizer_count:
            value += 0.5 * float(organizer_effect @ organizer_effect) / organizer_sd**2
            gradient[organizer_start:] += organizer_effect / organizer_sd**2
        return value, gradient

    def hessian(w, project_sd, organizer_sd):
        beta, project_effect, organizer_effect = unpack(w)
        eta = X @ beta
        if project_count:
            eta = eta + Zp @ project_effect
        if organizer_count:
            eta = eta + Zo @ organizer_effect
        weight = n * expit(eta) * expit(-eta)
        D = np.column_stack([X, Zp, Zo])
        H = D.T @ (D * weight[:, None])
        H[np.diag_indices(fixed)] += prior_precision
        if project_count:
            index = np.arange(project_start, organizer_start)
            H[index, index] += 1.0 / project_sd**2
        if organizer_count:
            index = np.arange(organizer_start, total)
            H[index, index] += 1.0 / organizer_sd**2
        return H

    def solve(project_sd, organizer_sd, warm=None):
        start = np.zeros(total) if warm is None else warm
        result = minimize(
            lambda w: objective(w, project_sd, organizer_sd),
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-8},
        )
        w = result.x
        H = hessian(w, project_sd, organizer_sd)
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            H = H + np.eye(total) * 1e-8
            try:
                L = np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                return None
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        value = objective(w, project_sd, organizer_sd)[0]
        return value, w, H, L, log_det

    def marginal(project_sd, organizer_sd, warm=None):
        solved = solve(project_sd, organizer_sd, warm)
        if solved is None:
            return math.inf, None
        value, w, H, L, log_det = solved
        # 随机效应先验的归一化项 0.5*log|Sigma| = k*log(sigma)。
        penalty = 0.5 * log_det
        if project_count:
            penalty += project_count * math.log(project_sd)
        if organizer_count:
            penalty += organizer_count * math.log(organizer_sd)
        return value + penalty, (value, w, H, L, log_det)

    project_sd = 0.0
    organizer_sd = 0.0
    best_warm = None
    if project_count or organizer_count:
        grid = [0.03, 0.08, 0.2, 0.5, 1.0]
        best = None
        for candidate_project in (grid if project_count else [0.0]):
            for candidate_organizer in (grid if organizer_count else [0.0]):
                value, solved = marginal(candidate_project, candidate_organizer, best_warm)
                if solved is None:
                    continue
                if best is None or value < best[0]:
                    best = (value, candidate_project, candidate_organizer, solved[1])
                    best_warm = solved[1]
        if best is None:
            raise ValueError("分层模型无法估计")

        def profile(theta):
            values = [
                math.exp(theta[0]) if project_count else 0.0,
                math.exp(theta[1]) if organizer_count else 0.0,
            ]
            value, _ = marginal(values[0], values[1], best[3])
            return value

        start = [math.log(best[1]) if project_count else 0.0, math.log(best[2]) if organizer_count else 0.0]
        refined = minimize(
            profile,
            np.array(start),
            method="L-BFGS-B",
            bounds=[(-5.5, 1.6) if project_count else (0, 0), (-5.5, 1.6) if organizer_count else (0, 0)],
            options={"maxiter": 80, "ftol": 1e-8},
        )
        project_sd = math.exp(refined.x[0]) if project_count else 0.0
        organizer_sd = math.exp(refined.x[1]) if organizer_count else 0.0
        value, solved = marginal(project_sd, organizer_sd, best[3])
        if solved is None:
            project_sd, organizer_sd = best[1], best[2]
            value, solved = marginal(project_sd, organizer_sd)
    else:
        value, solved = marginal(0.0, 0.0)
    if solved is None:
        raise ValueError("分层模型无法估计")
    penalized, w, H, L, log_det = solved
    model = {
        "config": config,
        "exposure": exposure,
        "levels": levels,
        "center": center,
        "fixed": fixed,
        "project_start": project_start,
        "organizer_start": organizer_start,
        "total": total,
        "project_sd": project_sd,
        "organizer_sd": organizer_sd,
        "w": w,
        "H": H,
        "L": L,
        "log_det": log_det,
        "penalized": penalized,
        "marginal": value,
        "names": design.names,
    }
    if len(_FIT_CACHE) >= _FIT_CACHE_LIMIT:
        _FIT_CACHE.clear()
    _FIT_CACHE[key] = model
    return model


def _draw_latent(model, rows, draws, seed):
    """从 Laplace 后验抽取线性预测值；保留候选之间的相关性。"""
    config = model["config"]
    design = _design(rows, model["levels"], config, model["center"])
    X, Zp, Zo = design.X, design.Zp, design.Zo
    H = model["H"]
    L = model["L"]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((draws, H.shape[0]))
    delta = solve_triangular(L, noise.T, lower=True, trans="T").T
    weights = model["w"][None, :] + delta
    beta = weights[:, : model["fixed"]]
    eta = beta @ X.T
    if model["project_start"] < model["organizer_start"]:
        project_effect = weights[:, model["project_start"] : model["organizer_start"]]
        eta = eta + project_effect @ Zp.T
        unseen = design.unseen_project > 0
        if np.any(unseen) and model["project_sd"] > 0:
            extra = rng.standard_normal((draws, int(unseen.sum()))) * model["project_sd"]
            eta[:, unseen] += extra
    if model["organizer_start"] < model["total"]:
        organizer_effect = weights[:, model["organizer_start"] :]
        eta = eta + organizer_effect @ Zo.T
        unseen = design.unseen_organizer > 0
        if np.any(unseen) and model["organizer_sd"] > 0:
            extra = rng.standard_normal((draws, int(unseen.sum()))) * model["organizer_sd"]
            eta[:, unseen] += extra
    return eta, design


def _log_binomial(y, n, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return (
        gammaln(n + 1.0)
        - gammaln(y + 1.0)
        - gammaln(n - y + 1.0)
        + y * np.log(p)
        + (n - y) * np.log1p(-p)
    )


def _heldout_score(model, rows, draws=800, seed=17):
    eta, _ = _draw_latent(model, rows, draws, seed)
    y = np.array([row["achieved"] for row in rows], dtype=float)
    n = np.array([row["N"] for row in rows], dtype=float)
    p = expit(eta)
    log_likelihood = _log_binomial(y[None, :], n[None, :], p)
    score = float(np.mean(logsumexp(log_likelihood, axis=0) - math.log(draws)))
    posterior_mean = p.mean(axis=0)
    mae = float(np.mean(np.abs(n * posterior_mean - y)))
    return score, mae, posterior_mean


def _prior_snapshot(snapshot, as_of):
    tables = {}
    for name, rows in snapshot["tables"].items():
        tables[name] = [dict(row) for row in rows if row["released_at"] <= as_of]
    return {"as_of": as_of, "tables": tables, "snapshot_id": snapshot.get("snapshot_id")}


def _structure(name):
    for structure in STRUCTURES:
        if structure["name"] == name:
            return dict(structure)
    raise ValueError("未知模型结构")


def _folds(snapshot, folds_limit):
    """按届次向前验证的折：训练只用该届决策前可见的届次。"""
    base_rows, _ = _panel(snapshot, "supported")
    seasons = sorted({row["season"] for row in base_rows})
    known = defaultdict(list)
    for row in snapshot["tables"]["nominations"]:
        known[int(row["season"])].append(row["known_before"])
    folds = []
    for season in seasons:
        prior = [value for value in seasons if value < season]
        if len(prior) < 2 or season not in known:
            continue
        cutoff = min(known[season])
        train_rows, _ = _panel(_prior_snapshot(snapshot, cutoff), "supported")
        train_rows = [row for row in train_rows if row["season"] < season]
        test_rows = [row for row in base_rows if row["season"] == season]
        if len(train_rows) < 8 or not test_rows:
            continue
        folds.append((season, cutoff, train_rows, test_rows))
    if folds_limit is not None and len(folds) > int(folds_limit):
        folds = folds[-int(folds_limit) :]
    return folds


def _validation(snapshot, config, draws):
    folds = _folds(snapshot, config["validation_folds"])
    if not folds:
        return {
            "status": "insufficient_history",
            "criterion": "按届次向前验证的平均二项对数预测分（越高越好）",
            "folds": [],
            "tolerance": float(config["validation_tolerance"]),
            "candidates": [],
            "selected": "region",
            "tie_break": "历史不足以验证借力层级，退回只含地区固定效应的结构。",
            "note": "没有可用的向前验证折；不启用项目或承办方借力，避免未经检验的收缩。",
        }
    results = []
    for structure in STRUCTURES:
        per_fold = []
        for season, cutoff, train_rows, test_rows in folds:
            fitted = _fit(train_rows, {**config, **structure}, "supported")
            score, mae, _ = _heldout_score(
                fitted, test_rows, draws=min(800, draws), seed=int(config["seed"]) + season
            )
            per_fold.append(
                {
                    "season": season,
                    "cutoff": cutoff,
                    "train_rounds": len(train_rows),
                    "test_rounds": len(test_rows),
                    "log_score": _r(score),
                    "mae": _r(mae),
                }
            )
        results.append(
            {
                "structure": structure,
                "mean_log_score": float(np.mean([fold["log_score"] for fold in per_fold])),
                "mean_mae": float(np.mean([fold["mae"] for fold in per_fold])),
                "folds": per_fold,
            }
        )
    best_score = max(item["mean_log_score"] for item in results)
    tolerance = float(config["validation_tolerance"])
    tied = [item for item in results if item["mean_log_score"] >= best_score - tolerance]
    # 分数接近时优先保留地区层级，避免把已验证的地区差异抹平；其余按模型从简。
    tied.sort(
        key=lambda item: (
            not item["structure"]["region"],
            not item["structure"]["organizer"],
            not item["structure"]["project"],
            item["structure"]["category"],
        )
    )
    selected = tied[0]
    return {
        "status": "validated",
        "criterion": "按届次向前验证的平均二项对数预测分（越高越好）",
        "folds": [
            {"season": season, "cutoff": cutoff, "train_rounds": len(train), "test_rounds": len(test)}
            for season, cutoff, train, test in folds
        ],
        "tolerance": tolerance,
        "candidates": [
            {
                "structure": item["structure"]["name"],
                "configuration": {
                    key: item["structure"][key] for key in ("project", "organizer", "region", "category")
                },
                "mean_log_score": _r(item["mean_log_score"]),
                "mean_mae": _r(item["mean_mae"]),
                "folds": item["folds"],
            }
            for item in results
        ],
        "selected": selected["structure"]["name"],
        "tie_break": "分数差距不超过容差时，优先保留 region，其次 organizer，再次 project，最后 category",
    }


def _exposure_check(snapshot, config, structure, draws):
    """同一结构下比较 supported 与 supported+additional_slots 的样本外表现。"""
    folds = _folds(snapshot, config["validation_folds"])
    scores = {"supported": [], "expanded": []}
    for season, cutoff, _, _ in folds:
        for exposure in ("supported", "expanded"):
            train_rows, _ = _panel(_prior_snapshot(snapshot, cutoff), exposure)
            train_rows = [row for row in train_rows if row["season"] < season]
            test_rows, _ = _panel(snapshot, exposure)
            test_rows = [row for row in test_rows if row["season"] == season]
            if len(train_rows) < 8 or not test_rows:
                continue
            fitted = _fit(train_rows, {**config, **structure}, exposure)
            score, _, _ = _heldout_score(
                fitted,
                test_rows,
                draws=min(800, draws),
                seed=int(config["seed"]) + season + (0 if exposure == "supported" else 1000),
            )
            scores[exposure].append(score)
    summary = {}
    for exposure in ("supported", "expanded"):
        values = scores[exposure]
        summary[exposure] = _r(float(np.mean(values))) if values else None
    if summary["supported"] is None and summary["expanded"] is None:
        summary["selected"] = "supported"
        summary["status"] = "insufficient_history"
    elif summary["expanded"] is not None and summary["supported"] is not None and summary["expanded"] > summary["supported"] + float(config["validation_tolerance"]):
        summary["selected"] = "expanded"
        summary["status"] = "expanded_preferred"
    else:
        summary["selected"] = "supported"
        summary["status"] = "supported_preferred"
    summary["note"] = (
        "主口径默认使用 supported；追加名额是筛选后的扩容，只有样本外预测明确更好时才并入暴露量。"
    )
    return summary


def _permutation_test(project_rows, chosen_map, stratum_map, permutations, seed):
    """在承办方层内置换入选标签，检验决策前差异是否超出随机波动。"""
    projects = sorted(project_rows)
    chosen = np.array([1.0 if chosen_map.get(project) else 0.0 for project in projects])
    achieved = np.array([project_rows[project][0] for project in projects], dtype=float)
    supported = np.array([project_rows[project][1] for project in projects], dtype=float)

    def statistic(labels):
        treated = labels > 0.5
        if treated.sum() == 0 or (~treated).sum() == 0:
            return 0.0
        return float(
            achieved[treated].sum() / supported[treated].sum()
            - achieved[~treated].sum() / supported[~treated].sum()
        )

    observed = statistic(chosen)
    strata = defaultdict(list)
    for index, project in enumerate(projects):
        strata[stratum_map.get(project)].append(index)
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(int(permutations)):
        labels = chosen.copy()
        for indices in strata.values():
            if len(indices) < 2:
                continue
            values = labels[indices].copy()
            rng.shuffle(values)
            labels[indices] = values
        if abs(statistic(labels)) >= abs(observed) - 1e-12:
            more_extreme += 1
    return observed, (more_extreme + 1.0) / (int(permutations) + 1.0)


def _conversion_profile(snapshot, config, structure, identifiable, reasons):
    """扩容折算系数的样本外轮廓；仅当入选状态或名额在项目内变化时可识别。"""
    if not identifiable:
        return {
            "status": "not_identified",
            "coefficient": None,
            "interval": None,
            "reasons": reasons,
            "note": "入选项目与未入选项目的差异同时包含项目身份、承办方与扩容，不能解释为扩容的因果效果；主分析不把追加名额计入暴露量。",
        }
    folds = _folds(snapshot, config["validation_folds"])
    grid = [float(value) for value in config["conversion_grid"]]
    profile = []
    for weight in grid:
        scores = []
        for season, cutoff, _, _ in folds:
            train_rows, _ = _panel(_prior_snapshot(snapshot, cutoff), "supported", additional_weight=weight)
            train_rows = [row for row in train_rows if row["season"] < season]
            test_rows, _ = _panel(snapshot, "supported", additional_weight=weight)
            test_rows = [row for row in test_rows if row["season"] == season]
            if len(train_rows) < 8 or not test_rows:
                continue
            fitted = _fit(train_rows, {**config, **structure}, "conversion-%.3f" % weight)
            score, _, _ = _heldout_score(
                fitted,
                test_rows,
                draws=min(800, config["draws"]),
                seed=int(config["seed"]) + season + int(weight * 1000),
            )
            scores.append(score)
        profile.append({"weight": weight, "mean_log_score": _r(float(np.mean(scores))) if scores else None})
    usable = [item for item in profile if item["mean_log_score"] is not None]
    if not usable:
        return {
            "status": "not_identified",
            "coefficient": None,
            "interval": None,
            "reasons": reasons + ["验证折不足"],
            "profile": profile,
            "note": "没有足够验证折估计扩容折算。",
        }
    best = max(usable, key=lambda item: item["mean_log_score"])
    tolerance = float(config["validation_tolerance"])
    plausible = [item["weight"] for item in usable if item["mean_log_score"] >= best["mean_log_score"] - tolerance]
    return {
        "status": "estimated",
        "coefficient": best["weight"],
        "interval": [min(plausible), max(plausible)],
        "profile": profile,
        "note": "系数表示追加名额中可计入有效暴露的比例；区间为样本外分数与最优值相差不超过容差的取值。",
    }


def _audit(snapshot, panel_rows, config, structure):
    nominations = snapshot["tables"]["nominations"]
    chosen_by_season = defaultdict(dict)
    known_by_season = defaultdict(list)
    slots_by_season = defaultdict(dict)
    for row in nominations:
        season = int(row["season"])
        chosen_by_season[season][row["project"]] = bool(row["chosen"])
        slots_by_season[season][row["project"]] = int(row["additional_slots"])
        known_by_season[season].append(row["known_before"])
    seasons = sorted(chosen_by_season)

    patterns = defaultdict(set)
    for season in seasons:
        for project, chosen in chosen_by_season[season].items():
            patterns[project].add(chosen)
    switchers = sum(1 for values in patterns.values() if len(values) > 1)
    slot_variation = 0
    for project in patterns:
        if len({slots_by_season[season].get(project) for season in seasons}) > 1:
            slot_variation += 1

    round_by_season = defaultdict(list)
    for row in panel_rows:
        round_by_season[row["season"]].append(row)
    retrospective = []
    for season in seasons:
        if season == 0 or season not in known_by_season:
            continue
        cutoff = min(known_by_season[season])
        prior_rows, _ = _panel(_prior_snapshot(snapshot, cutoff), "supported")
        prior_rows = [row for row in prior_rows if row["season"] < season]
        if not prior_rows:
            continue
        totals = defaultdict(lambda: [0, 0])
        strata = {}
        for row in prior_rows:
            totals[row["project"]][0] += row["achieved"]
            totals[row["project"]][1] += row["supported"]
            strata[row["project"]] = row["organizer"]
        chosen_map = chosen_by_season[season]
        observed_gap, p_value = _permutation_test(
            totals, chosen_map, strata, config["permutations"], int(config["seed"]) + season
        )
        chosen_ids = {project for project, chosen in chosen_map.items() if chosen}
        control_ids = {project for project, chosen in chosen_map.items() if not chosen}

        def aggregate(ids):
            achieved = sum(value[0] for project, value in totals.items() if project in ids)
            supported = sum(value[1] for project, value in totals.items() if project in ids)
            return achieved / supported if supported else None

        prior_chosen = aggregate(chosen_ids)
        prior_control = aggregate(control_ids)
        current = round_by_season.get(season, [])
        current_chosen = [row for row in current if row["project"] in chosen_ids]
        current_control = [row for row in current if row["project"] in control_ids]

        def season_rate(rows, key):
            if not rows:
                return None
            return sum(row["achieved"] for row in rows) / sum(row[key] for row in rows)

        post_chosen = season_rate(current_chosen, "supported")
        post_control = season_rate(current_control, "supported")
        post_chosen_expanded = season_rate(current_chosen, "expanded")
        post_control_expanded = season_rate(current_control, "expanded")
        retrospective.append(
            {
                "season": season,
                "decision_cutoff": cutoff,
                "chosen_projects": len(chosen_ids),
                "control_projects": len(control_ids),
                "prior_rate_chosen": None if prior_chosen is None else _r(prior_chosen),
                "prior_rate_control": None if prior_control is None else _r(prior_control),
                "prior_rate_gap": None if prior_chosen is None or prior_control is None else _r(prior_chosen - prior_control),
                "prior_gap_permutation_p": _r(p_value),
                "additional_slots_chosen": sum(slots_by_season[season].get(project, 0) for project in chosen_ids),
                "additional_slots_control": sum(slots_by_season[season].get(project, 0) for project in control_ids),
                "post_rate_chosen_supported": None if post_chosen is None else _r(post_chosen),
                "post_rate_control_supported": None if post_control is None else _r(post_control),
                "post_rate_chosen_expanded": None if post_chosen_expanded is None else _r(post_chosen_expanded),
                "post_rate_control_expanded": None if post_control_expanded is None else _r(post_control_expanded),
                "change_chosen_supported": None
                if post_chosen is None or prior_chosen is None
                else _r(post_chosen - prior_chosen),
                "change_control_supported": None
                if post_control is None or prior_control is None
                else _r(post_control - prior_control),
                "difference_in_differences_supported": None
                if post_chosen is None or post_control is None or prior_chosen is None or prior_control is None
                else _r((post_chosen - prior_chosen) - (post_control - prior_control)),
                "change_chosen_expanded": None
                if post_chosen_expanded is None or prior_chosen is None
                else _r(post_chosen_expanded - prior_chosen),
                "change_control_expanded": None
                if post_control_expanded is None or prior_control is None
                else _r(post_control_expanded - prior_control),
            }
        )

    identifiable = bool(switchers > 0 and slot_variation > 0)
    reasons = []
    if switchers == 0:
        reasons.append("入选状态在既有届次中按项目固定，未出现项目内切换")
    if slot_variation == 0:
        reasons.append("追加名额在项目内没有变化")
    if switchers > 0 and slot_variation == 0:
        reasons.append("入选状态有项目内切换，但追加名额始终随入选状态固定，扩容与入选效果仍不可分离")
    if slot_variation > 0 and switchers == 0:
        reasons.append("追加名额有项目内变化，但入选状态始终固定，扩容与项目身份仍不可分离")
    conversion = _conversion_profile(snapshot, config, structure, identifiable, reasons)

    chosen_sets = [
        frozenset(project for project, chosen in mapping.items() if chosen)
        for mapping in chosen_by_season.values()
    ]
    stability = None
    if chosen_sets:
        counts = defaultdict(int)
        for value in chosen_sets:
            counts[value] += 1
        stability = max(counts.values()) / len(chosen_sets)

    return {
        "seasons": seasons,
        "projects_switching_selection": switchers,
        "projects_with_slot_variation": slot_variation,
        "selection_identified": identifiable,
        "chosen_set_stability": None if stability is None else _r(stability),
        "additional_slots_conversion": conversion,
        "retrospective": retrospective,
        "interpretation": [
            "回看先比较决策前可见的届次，再展示筛选后的结果，避免只用入选后的表现证明筛选有效。",
            "入选状态与追加名额若在项目内不变，则入选项目与未入选项目的差异无法与项目身份分开。",
            "主分析使用 supported 作为暴露量；expanded 只作敏感性对照，并在成果中单列。",
            "difference_in_differences 比较入选组与对照组从决策前到当届的变化，仍需结合识别条件解读。",
        ],
    }


def _borrowing(model, panel_rows, candidates, draws, seed):
    """报告借力层级、收缩程度与地区差异，避免把地区效应抹平。"""
    levels = model["levels"]
    w = model["w"]
    fixed = model["fixed"]
    names = model["names"]
    beta = w[:fixed]
    project_start = model["project_start"]
    organizer_start = model["organizer_start"]
    total = model["total"]
    project_effect = (
        w[project_start:organizer_start] if project_start < organizer_start else np.zeros(len(levels["project"]))
    )
    organizer_effect = (
        w[organizer_start:total] if organizer_start < total else np.zeros(len(levels["organizer"]))
    )

    eta, _ = _draw_latent(model, candidates, draws, seed)
    p_draws = expit(eta)
    posterior_mean = p_draws.mean(axis=0)
    posterior_sd = p_draws.std(axis=0, ddof=1) if draws > 1 else np.zeros(len(candidates))
    totals = defaultdict(lambda: [0, 0])
    for row in panel_rows:
        totals[row["project"]][0] += row["achieved"]
        totals[row["project"]][1] += row["supported"]

    peers = defaultdict(lambda: {"organizer": [], "region": []})
    attribute_source = {}
    for row in panel_rows:
        attribute_source[row["project"]] = (row["organizer"], row["region"])
    for row in candidates:
        attribute_source.setdefault(row["project"], (row["organizer"], row["region"]))
    for project, (organizer, region) in attribute_source.items():
        for other, (other_organizer, other_region) in attribute_source.items():
            if other == project:
                continue
            if other_organizer == organizer:
                peers[project]["organizer"].append(other)
            if other_region == region:
                peers[project]["region"].append(other)

    shrinkage = []
    for index, row in enumerate(candidates):
        achieved, supported = totals.get(row["project"], (0, 0))
        raw_rate = achieved / supported if supported else None
        if raw_rate is not None and 0 < raw_rate < 1 and supported > 0:
            raw_logit_se = 1.0 / math.sqrt(supported * raw_rate * (1.0 - raw_rate))
        else:
            raw_logit_se = None
        column = p_draws[:, index]
        posterior_logit = np.log(np.clip(column, 1e-9, 1 - 1e-9) / (1 - np.clip(column, 1e-9, 1 - 1e-9)))
        posterior_logit_sd = float(posterior_logit.std(ddof=1)) if draws > 1 else None
        borrowed_fraction = None
        if raw_logit_se and posterior_logit_sd is not None:
            borrowed_fraction = max(0.0, min(1.0, 1.0 - posterior_logit_sd / raw_logit_se))
        shrinkage.append(
            {
                "project": row["project"],
                "observed_supported": supported,
                "observed_rate": None if raw_rate is None else _r(raw_rate),
                "posterior_rate": _r(posterior_mean[index]),
                "posterior_rate_sd": _r(posterior_sd[index]),
                "raw_logit_se": None if raw_logit_se is None else _r(raw_logit_se),
                "posterior_logit_sd": None if posterior_logit_sd is None else _r(posterior_logit_sd),
                "borrowed_fraction": None if borrowed_fraction is None else _r(borrowed_fraction),
                "shares_organizer_with": sorted(peers[row["project"]]["organizer"]),
                "shares_region_with": sorted(peers[row["project"]]["region"]),
            }
        )

    def effect_table(prefix, values):
        table = {group: 0.0 for group in values}
        for name, value in zip(names, beta):
            if name.startswith(prefix):
                table[name.split(":", 1)[1]] = float(value)
        return {key: _r(value) for key, value in table.items()}

    project_variance = float(model["project_sd"]) ** 2
    organizer_variance = float(model["organizer_sd"]) ** 2
    total_variance = project_variance + organizer_variance
    return {
        "project_sd_logit": _r(model["project_sd"]),
        "organizer_sd_logit": _r(model["organizer_sd"]),
        "organizer_variance_share": None
        if total_variance <= 0
        else _r(organizer_variance / total_variance),
        "region_effects_logit": effect_table("region:", levels["region"]),
        "category_effects_logit": effect_table("category:", levels["category"]),
        "organizer_effects_logit": {
            organizer: _r(organizer_effect[index]) if index < len(organizer_effect) else 0.0
            for index, organizer in enumerate(levels["organizer"])
        },
        "intercept": _r(beta[0]),
        "season_trend_logit": _r(beta[names.index("season_trend")]) if "season_trend" in names else None,
        "shrinkage": shrinkage,
        "interpretation": (
            "项目与承办方的方差分量决定借力强度：organizer 方差占比越高，项目估计越多地向同承办方收缩；"
            "region 固定效应保留地区差异，不并入随机效应。shares_* 列出各候选实际借到信息的同层项目。"
        ),
    }


def _tiers(sorted_candidates, p_draws, threshold):
    """按后验优势概率合并相邻候选；证据不足时保留并列而不是强行拆分。"""
    tiers = []
    representative = None
    for index, candidate in enumerate(sorted_candidates):
        column = candidate["_column"]
        if representative is None:
            tiers.append([index])
            representative = column
            continue
        advantage = float(np.mean(p_draws[:, representative] > p_draws[:, column]))
        if advantage < threshold:
            tiers[-1].append(index)
        else:
            tiers.append([index])
            representative = column
    return tiers


def _candidate_rows(snapshot):
    projects = {row["id"]: row for row in snapshot["tables"]["projects"]}
    nominations = snapshot["tables"]["nominations"]
    latest = None
    candidate_ids = []
    if nominations:
        latest = max(int(row["season"]) for row in nominations)
        candidate_ids = sorted(
            {row["project"] for row in nominations if int(row["season"]) == latest and row["candidate"]}
        )
    source = "nominations-season-%s" % latest if candidate_ids else "all-visible-projects"
    if not candidate_ids:
        candidate_ids = sorted(projects)
    season = (latest + 1) if latest is not None else 1
    rows = []
    for project in candidate_ids:
        attributes = projects.get(project)
        if attributes is None:
            raise ValueError("候选项目缺少项目资料")
        rows.append(
            {
                "project": project,
                "season": season,
                "organizer": attributes["organizer"],
                "region": attributes["region"],
                "category": attributes["category"],
                "supported": 0,
                "additional_slots": 0,
                "achieved": 0,
                "chosen": None,
                "known_before": None,
                "expanded": 0,
                "N": 1.0,
            }
        )
    return rows, source, season


def _latest_exposure(panel_rows):
    latest = {}
    for row in panel_rows:
        if row["project"] not in latest or row["season"] >= latest[row["project"]][0]:
            latest[row["project"]] = (row["season"], row["N"])
    return {project: value[1] for project, value in latest.items()}


def _default_slots(candidates, latest_exposure):
    values = [latest_exposure.get(row["project"], 0) for row in candidates]
    values = [value for value in values if value > 0]
    if not values:
        return 1
    return max(1, int(round(float(np.median(values)))))


def _portfolio(sorted_candidates, p_draws, slots, size, seed):
    values = p_draws * float(slots)
    selected = np.arange(min(size, len(sorted_candidates)))
    order = np.argsort(-values, axis=1)
    ideal = np.take_along_axis(values, order[:, : len(selected)], axis=1).sum(axis=1)
    chosen = values[:, selected].sum(axis=1)
    regret = ideal - chosen
    top_sets = order[:, : len(selected)]
    rng = np.random.default_rng(seed + 1)
    predictive = rng.binomial(int(slots), p_draws[:, selected]).sum(axis=1)

    def interval(array, level):
        return [_r(np.quantile(array, (1 - level) / 2)), _r(np.quantile(array, 1 - (1 - level) / 2))]

    return {
        "portfolio_size": len(selected),
        "slots_per_candidate": int(slots),
        "total_slots": int(len(selected) * slots),
        "selected": [sorted_candidates[index]["project"] for index in selected],
        "expected_achieved": _r(chosen.mean()),
        "expected_achieved_interval_80": interval(chosen, 0.8),
        "expected_achieved_interval_95": interval(chosen, 0.95),
        "predictive_interval_80": interval(predictive, 0.8),
        "predictive_interval_95": interval(predictive, 0.95),
        "oracle_expected_achieved": _r(ideal.mean()),
        "expected_regret": _r(regret.mean()),
        "probability_selected_matches_oracle_set": _r(float(np.mean(regret <= 1e-9))),
        "efficiency": _r(float(chosen.mean() / ideal.mean())) if ideal.mean() > 0 else None,
        "top_set_uncertainty": [
            _r(float(np.mean(~np.any(top_sets == selected[index], axis=1))))
            for index in range(len(selected))
        ],
    }


def candidate_analysis(snapshot, config=None):
    """生成候选排序、组合收益、借力结构与筛选回看。

    返回可直接作为成果 result/model 的内容；不修改快照。
    """
    settings = dict(DEFAULT_CONFIG)
    if config:
        settings.update(config)
    config = settings
    draws = int(config["draws"])
    if draws < 200:
        raise ValueError("后验抽样次数过少")
    if not 0 < float(config["tie_threshold"]) <= 1:
        raise ValueError("并列阈值必须在 (0,1]")
    if int(config["validation_folds"]) < 1:
        raise ValueError("至少需要一折验证")
    if int(config["permutations"]) < 1:
        raise ValueError("置换次数必须为正")

    panel_rows, applied_corrections = _panel(snapshot, "supported")
    if len({row["project"] for row in panel_rows}) < 2:
        raise ValueError("至少需要两个项目的届次资料")
    validation = _validation(snapshot, config, draws)
    structure = _structure(validation["selected"])
    structure_config = {**config, **structure, "trend": True}
    exposure_check = _exposure_check(snapshot, config, structure, draws)
    exposure = exposure_check["selected"]
    panel_rows, applied_corrections = _panel(snapshot, exposure)
    model = _fit(panel_rows, structure_config, exposure)

    candidates, candidate_source, next_season = _candidate_rows(snapshot)
    latest_exposure = _latest_exposure(panel_rows)
    if config["slots_per_candidate"] is None:
        slots = _default_slots(candidates, latest_exposure)
        slots_source = "latest-season-median-exposure"
    else:
        slots = int(config["slots_per_candidate"])
        slots_source = "committee-supplied"
    if slots < 1:
        raise ValueError("未来支持名额必须为正")
    if config["portfolio_size"] is None:
        nominations = snapshot["tables"]["nominations"]
        latest_season = max((int(row["season"]) for row in nominations), default=None)
        if latest_season is None:
            size = min(10, len(candidates))
        else:
            size = sum(
                1 for row in nominations if int(row["season"]) == latest_season and row["chosen"]
            )
            if size < 1:
                size = min(10, len(candidates))
        size_source = "latest-nomination-chosen"
        if size > len(candidates):
            size_source += "-clipped-to-candidate-count"
    else:
        size = int(config["portfolio_size"])
        if size < 1:
            raise ValueError("入选名额必须为正")
        size_source = "committee-supplied"
        if size > len(candidates):
            size_source += "-clipped-to-candidate-count"
    size = min(len(candidates), size)

    eta, _ = _draw_latent(model, candidates, draws, int(config["seed"]))
    p_draws = expit(eta)
    count = p_draws.shape[1]
    ranks = np.empty_like(p_draws)
    order = np.argsort(-p_draws, axis=1)
    for row_index in range(p_draws.shape[0]):
        ranks[row_index, order[row_index]] = np.arange(1, count + 1)
    posterior_mean = p_draws.mean(axis=0)

    totals = defaultdict(lambda: [0, 0])
    exposures = defaultdict(float)
    seasons_seen = defaultdict(set)
    for row in panel_rows:
        totals[row["project"]][0] += row["achieved"]
        totals[row["project"]][1] += row["supported"]
        exposures[row["project"]] += row["N"]
        seasons_seen[row["project"]].add(row["season"])
    raw_rates = {
        project: (achieved / supported if supported else None)
        for project, (achieved, supported) in totals.items()
    }
    raw_order = sorted(totals, key=lambda project: (-(raw_rates[project] if raw_rates[project] is not None else -1.0), project))
    raw_rank = {project: index + 1 for index, project in enumerate(raw_order)}

    sorted_candidates = []
    for column in np.argsort(-posterior_mean):
        row = candidates[column]
        achieved, supported = totals.get(row["project"], (0, 0))
        rank_draws = ranks[:, column]
        character = p_draws[:, column]
        sorted_candidates.append(
            {
                "project": row["project"],
                "organizer": row["organizer"],
                "region": row["region"],
                "category": row["category"],
                "observed_achieved": achieved,
                "observed_supported": supported,
                "observed_exposure": _r(exposures.get(row["project"], 0.0), 4),
                "observed_rate": None if raw_rates.get(row["project"]) is None else _r(raw_rates[row["project"]]),
                "observed_seasons": len(seasons_seen.get(row["project"], set())),
                "expected_rate": _r(posterior_mean[column]),
                "expected_rate_interval_80": [
                    _r(np.quantile(character, 0.1)),
                    _r(np.quantile(character, 0.9)),
                ],
                "expected_rate_interval_95": [
                    _r(np.quantile(character, 0.025)),
                    _r(np.quantile(character, 0.975)),
                ],
                "expected_achieved": _r(slots * posterior_mean[column]),
                "expected_achieved_interval_80": [
                    _r(slots * np.quantile(character, 0.1)),
                    _r(slots * np.quantile(character, 0.9)),
                ],
                "expected_achieved_interval_95": [
                    _r(slots * np.quantile(character, 0.025)),
                    _r(slots * np.quantile(character, 0.975)),
                ],
                "rank_mean": _r(rank_draws.mean(), 4),
                "rank_interval_80": [
                    int(np.quantile(rank_draws, 0.1)),
                    int(np.quantile(rank_draws, 0.9)),
                ],
                "rank_interval_95": [
                    int(np.quantile(rank_draws, 0.025)),
                    int(np.quantile(rank_draws, 0.975)),
                ],
                "top_k_probability": None,
                "selection_risk": None,
                "raw_rank": raw_rank.get(row["project"]),
                "rank_gain_vs_raw": None
                if raw_rank.get(row["project"]) is None
                else _r(raw_rank[row["project"]] - rank_draws.mean(), 4),
                "tier": None,
                "_column": column,
            }
        )

    rng = np.random.default_rng(int(config["seed"]) + 2)
    columns = [candidate["_column"] for candidate in sorted_candidates]
    predictive = rng.binomial(int(slots), p_draws[:, columns])
    for index, candidate in enumerate(sorted_candidates):
        candidate["predictive_interval_80"] = [
            int(np.quantile(predictive[:, index], 0.1)),
            int(np.quantile(predictive[:, index], 0.9)),
        ]
        candidate["predictive_interval_95"] = [
            int(np.quantile(predictive[:, index], 0.025)),
            int(np.quantile(predictive[:, index], 0.975)),
        ]

    tiers = _tiers(sorted_candidates, p_draws, float(config["tie_threshold"]))
    tier_rows = []
    for tier_index, indices in enumerate(tiers, start=1):
        for index in indices:
            sorted_candidates[index]["tier"] = tier_index
        rates = [sorted_candidates[index]["expected_rate"] for index in indices]
        tier_rows.append(
            {
                "tier": tier_index,
                "projects": [sorted_candidates[index]["project"] for index in indices],
                "expected_rate_min": min(rates),
                "expected_rate_max": max(rates),
            }
        )

    top_sets = np.argsort(-p_draws, axis=1)[:, :size]
    for index, candidate in enumerate(sorted_candidates):
        probability = float(np.mean(np.any(top_sets == candidate["_column"], axis=1)))
        candidate["top_k_probability"] = _r(probability, 4)
        candidate["selection_risk"] = _r(1.0 - probability, 4)

    p_for_portfolio = p_draws[:, columns]
    portfolio = _portfolio(sorted_candidates, p_for_portfolio, slots, size, int(config["seed"]))
    portfolio["selection_risk"] = {
        candidate["project"]: candidate["selection_risk"] for candidate in sorted_candidates[:size]
    }
    boundary_tier = sorted_candidates[size - 1]["tier"]
    portfolio["boundary_tie"] = {
        "tier": boundary_tier,
        "last_selected": sorted_candidates[size - 1]["project"],
        "first_not_selected": None if size >= len(sorted_candidates) else sorted_candidates[size]["project"],
        "shares_tier": (
            False
            if size >= len(sorted_candidates)
            else sorted_candidates[size - 1]["tier"] == sorted_candidates[size]["tier"]
        ),
        "note": "若 shares_tier 为真，最后一个入选名额与紧邻落选者在后验上不可区分，组合边界不稳。",
    }

    borrowing = _borrowing(model, panel_rows, candidates, draws, int(config["seed"]) + 3)
    audit = _audit(snapshot, panel_rows, config, structure)
    for candidate in sorted_candidates:
        candidate.pop("_column", None)

    return {
        "result": {
            "candidates": sorted_candidates,
            "tiers": tier_rows,
            "portfolio": portfolio,
            "borrowing": borrowing,
            "audit": audit,
            "config": {
                "portfolio_size": size,
                "portfolio_size_source": size_source,
                "slots_per_candidate": slots,
                "slots_source": slots_source,
                "candidate_source": candidate_source,
                "next_season": next_season,
                "draws": draws,
                "seed": int(config["seed"]),
                "tie_threshold": float(config["tie_threshold"]),
            },
            "interpretation": [
                "排序对象是同一未来支持名额下的后验预期达成，不是历史比例。",
                "小样本项目向同承办方/同地区收缩，并保留后验区间与名次区间。",
                "区间重叠且优势概率不足时显示并列层级；top-k 概率、选择风险与组合边界说明稳定性。",
                "回看把筛选前信息、筛选后结果和追加名额分开，不能把入选后的表现当作筛选收益。",
            ],
        },
        "model": {
            "method": METHOD,
            "version": VERSION,
            "likelihood": "binomial(achieved | exposure, p)",
            "link": "logit",
            "structure": structure["name"],
            "structure_configuration": {
                key: structure[key] for key in ("project", "organizer", "region", "category")
            },
            "exposure": exposure,
            "season_trend": bool(structure_config["trend"]),
            "priors": {
                "fixed_effects_sd": float(config["prior_sd"]),
                "project_sd_logit": _r(model["project_sd"]),
                "organizer_sd_logit": _r(model["organizer_sd"]),
            },
            "estimation": "Laplace marginal likelihood for variance components; posterior draws from Laplace approximation",
            "validation": validation,
            "exposure_check": exposure_check,
            "applied_corrections": sorted(set(applied_corrections)),
            "limitations": [
                "后验抽样基于 Laplace 近似，方差分量的抽样不确定性未并入区间。",
                "承办方与项目层级来自资料字段，不代表已经证明的因果机制。",
                "地区差异保留在固定效应中；若某地区样本过少，其效应会被先验收缩。",
                "入选状态与追加名额若在项目内不变，两者及项目身份的效果不可分离。",
                "未来届次的季节趋势按训练期线性外推，超出既有范围时不确定性更大。",
            ],
        },
        "data_basis": {
            "as_of": snapshot.get("as_of"),
            "snapshot_id": snapshot.get("snapshot_id"),
            "rounds": len(snapshot["tables"]["rounds"]),
            "projects": len(snapshot["tables"]["projects"]),
            "nominations": len(snapshot["tables"]["nominations"]),
            "corrections": len(snapshot["tables"]["corrections"]),
            "applied_corrections": sorted(set(applied_corrections)),
        },
    }
