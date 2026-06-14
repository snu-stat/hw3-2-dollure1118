# Problem 2-3
# 1. 포아송회귀

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.discrete.discrete_model import NegativeBinomial



Teams = pd.read_csv("data/Teams.csv")

# R의 dplyr 기능은 pandas로 대체함.
# R의 MASS::glm.nb, MASS::stepAIC는 Python에 동일 함수가 없으므로,
# statsmodels의 NegativeBinomial과 직접 구현한 stepwise AIC 함수로 대체함.


#데이터 전처리
predictors_2_3 = [
    "logRS", "logRA",
    "H", "X2B", "X3B", "HR", "BB", "SO", "CS", "HBP", "SF",
    "ERA", "CG", "SHO", "IPouts", "HA", "HRA", "BBA", "SOA",
    "E", "DP", "FP", "SV"
]

needed_raw_vars = [
    "yearID", "franchID", "teamID", "W", "L", "R", "RA",
    "H", "X2B", "X3B", "HR", "BB", "SO", "CS", "HBP", "SF",
    "ERA", "CG", "SHO", "IPouts", "HA", "HRA", "BBA", "SOA",
    "E", "DP", "FP", "SV"
]

# Lahman CSV에서는 2루타, 3루타 변수가 "2B", "3B"일 수 있음.
# R에서는 숫자로 시작하는 변수명이 자동으로 X2B, X3B처럼 바뀌는 경우가 있음.
if "X2B" not in Teams.columns and "2B" in Teams.columns:
    Teams = Teams.rename(columns={"2B": "X2B"})
if "X3B" not in Teams.columns and "3B" in Teams.columns:
    Teams = Teams.rename(columns={"3B": "X3B"})

missing_vars = sorted(set(needed_raw_vars) - set(Teams.columns))
if len(missing_vars) > 0:
    raise ValueError(
        "These variables are missing from Lahman::Teams: "
        + ", ".join(missing_vars)
    )

teams_2_3 = Teams[
    (Teams["yearID"] >= 2010) &
    (Teams["yearID"] <= 2025) &
    (Teams["yearID"] != 2020) &
    Teams["W"].notna() &
    Teams["L"].notna() &
    Teams["R"].notna() &
    Teams["RA"].notna() &
    (Teams["W"] + Teams["L"] > 0) &
    (Teams["R"] > 0) &
    (Teams["RA"] > 0)
].copy()

teams_2_3["G"] = teams_2_3["W"] + teams_2_3["L"]
teams_2_3["WPct"] = teams_2_3["W"] / teams_2_3["G"]
teams_2_3["RS"] = teams_2_3["R"]
teams_2_3["logRS"] = np.log(teams_2_3["R"])
teams_2_3["logRA"] = np.log(teams_2_3["RA"])
teams_2_3["log_ratio"] = np.log(teams_2_3["R"] / teams_2_3["RA"])

teams_2_3 = teams_2_3[
    [
        "yearID", "franchID", "teamID",
        "W", "L", "G", "WPct", "RS", "RA",
        "logRS", "logRA", "log_ratio",
        "H", "X2B", "X3B", "HR", "BB", "SO", "CS", "HBP", "SF",
        "ERA", "CG", "SHO", "IPouts", "HA", "HRA", "BBA", "SOA",
        "E", "DP", "FP", "SV"
    ]
].copy()

teams_2_3 = teams_2_3[
    (teams_2_3["WPct"] > 0) &
    (teams_2_3["WPct"] < 1)
].copy()

model_vars = ["W", "G", "WPct", "log_ratio"] + predictors_2_3
teams_2_3 = teams_2_3.dropna(subset=model_vars).copy()


# 모형식 일반화
def count_formula_text(response, predictors):
    if len(predictors) == 0:
        return f"{response} ~ offset(log(G))"
    return f"{response} ~ offset(log(G)) + " + " + ".join(predictors)


def logistic_formula_text(response, predictors, intercept=True):
    if len(predictors) == 0:
        return f"{response} ~ 1"
    rhs = " + ".join(predictors)
    if intercept:
        return f"{response} ~ {rhs}"
    return f"{response} ~ 0 + {rhs}"


count_full_formula = count_formula_text("W", predictors_2_3)
count_null_formula = "W ~ offset(log(G))"

logistic_full_formula = logistic_formula_text("WPct", predictors_2_3)
logistic_null_formula = "WPct ~ 1"


def selected_terms(model):
    return getattr(model, "_selected_predictors", [])


def pearson_dispersion(model):
    return np.sum(model.resid_pearson ** 2) / model.df_resid


# 행렬 다중공선성 계산용 조건수 함수
def condition_number(predictors, data):
    mm = data[predictors].copy()
    return np.linalg.cond(mm.to_numpy())


# 경고와 에러 저장
def fit_capture(func):
    warn = []

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = func()

            warn = [str(w.message) for w in caught]

        return {
            "value": value,
            "warnings": sorted(set(warn)),
            "error": None
        }

    except Exception as e:
        return {
            "value": e,
            "warnings": sorted(set(warn)),
            "error": e
        }


def fit_logistic_glm(data, predictors, intercept=True, maxiter=100):
    X = data[predictors].copy()

    if intercept:
        X = sm.add_constant(X, has_constant="add")

    y = data["WPct"]

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=data["G"]
    ).fit(maxiter=maxiter)

    model._selected_predictors = list(predictors)
    model._model_kind = "logistic"
    model._intercept = intercept

    return model


def fit_poisson_glm(data, predictors, maxiter=100):
    X = data[predictors].copy()
    X = sm.add_constant(X, has_constant="add")

    y = data["W"]
    offset = np.log(data["G"])

    model = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=offset
    ).fit(maxiter=maxiter)

    model._selected_predictors = list(predictors)
    model._model_kind = "poisson"

    return model


def fit_negative_binomial(data, predictors, maxiter=100):
    X = data[predictors].copy()
    X = sm.add_constant(X, has_constant="add")

    y = data["W"]
    offset = np.log(data["G"])

    model = NegativeBinomial(
        y,
        X,
        offset=offset,
        loglike_method="nb2"
    ).fit(maxiter=maxiter, disp=0)

    model._selected_predictors = list(predictors)
    model._model_kind = "negative_binomial"

    return model


def stepwise_aic(data, all_predictors, start_predictors, fit_func, trace=1):
    current_predictors = list(start_predictors)
    current_model = fit_func(data, current_predictors)
    current_aic = current_model.aic

    while True:
        candidates = []

        # drop step
        for var in current_predictors:
            new_predictors = [x for x in current_predictors if x != var]
            fit_result = fit_capture(lambda p=new_predictors: fit_func(data, p))

            if fit_result["error"] is None:
                model = fit_result["value"]
                candidates.append(("drop", var, new_predictors, model.aic, model))

        # add step
        remaining_predictors = [
            x for x in all_predictors if x not in current_predictors
        ]

        for var in remaining_predictors:
            new_predictors = current_predictors + [var]
            fit_result = fit_capture(lambda p=new_predictors: fit_func(data, p))

            if fit_result["error"] is None:
                model = fit_result["value"]
                candidates.append(("add", var, new_predictors, model.aic, model))

        if len(candidates) == 0:
            break

        best_action, best_var, best_predictors, best_aic, best_model = min(
            candidates,
            key=lambda x: x[3]
        )

        if best_aic < current_aic - 1e-8:
            if trace:
                print(
                    f"{best_action} {best_var}, "
                    f"AIC: {current_aic:.6f} -> {best_aic:.6f}"
                )

            current_predictors = best_predictors
            current_model = best_model
            current_aic = best_aic
        else:
            break

    return current_model


# 비교용 문제2-2 모형 생성
logistic_full = fit_logistic_glm(
    teams_2_3,
    predictors_2_3,
    intercept=True,
    maxiter=100
)

logistic_step = stepwise_aic(
    data=teams_2_3,
    all_predictors=predictors_2_3,
    start_predictors=predictors_2_3,
    fit_func=lambda data, predictors: fit_logistic_glm(
        data,
        predictors,
        intercept=True,
        maxiter=100
    ),
    trace=0
)

print(logistic_formula_text("WPct", selected_terms(logistic_step)))
print(logistic_step.summary())


# 포아송회귀
poisson_full = fit_poisson_glm(
    teams_2_3,
    predictors_2_3,
    maxiter=100
)

print("Poisson full model summary")
print(poisson_full.summary())
print()

poisson_step = stepwise_aic(
    data=teams_2_3,
    all_predictors=predictors_2_3,
    start_predictors=predictors_2_3,
    fit_func=lambda data, predictors: fit_poisson_glm(
        data,
        predictors,
        maxiter=100
    ),
    trace=1
)

print("\nPoisson stepwise AIC selected model")
print(count_formula_text("W", selected_terms(poisson_step)))
print(poisson_step.summary())
print()


print("Poisson post-stepwise drop1 check")


def poisson_drop1_chisq(data, current_model):
    current_predictors = selected_terms(current_model)
    rows = []

    rows.append({
        "term": "<none>",
        "Df": np.nan,
        "Deviance": current_model.deviance,
        "AIC": current_model.aic,
        "LRT": np.nan,
        "Pr(>Chi)": np.nan
    })

    for var in current_predictors:
        reduced_predictors = [x for x in current_predictors if x != var]
        reduced_model = fit_poisson_glm(data, reduced_predictors)

        lrt = reduced_model.deviance - current_model.deviance
        df_diff = reduced_model.df_resid - current_model.df_resid
        p_value = chi2.sf(lrt, df=df_diff)

        rows.append({
            "term": var,
            "Df": df_diff,
            "Deviance": reduced_model.deviance,
            "AIC": reduced_model.aic,
            "LRT": lrt,
            "Pr(>Chi)": p_value
        })

    return pd.DataFrame(rows).set_index("term")


poisson_drop_check = poisson_drop1_chisq(teams_2_3, poisson_step)
print(poisson_drop_check)
print()

poisson_disp_full = pearson_dispersion(poisson_full)
poisson_disp_step = pearson_dispersion(poisson_step)

print("Poisson overdispersion diagnostics")
print("Full model Pearson dispersion:", poisson_disp_full)
print("Stepwise model Pearson dispersion:", poisson_disp_step)
print("Rule of thumb: values much larger than 1 suggest overdispersion.\n")


# 음이항회귀
# 우선 모두 적합
nb_full_fit = fit_capture(
    lambda: fit_negative_binomial(
        teams_2_3,
        predictors_2_3,
        maxiter=100
    )
)

nb_model = None
nb_step_model = None
nb_fit_status = "not fitted"

# 음이항분포로 적합 실패 시 포아송회귀 단계적 선택모형 변수로 재적합
if nb_full_fit["error"] is not None:
    nb_fit_status = "full model failed"
    print("Negative binomial full model failed.")
    print("Error message:")
    print(str(nb_full_fit["value"]), "\n")

    print("Trying a negative binomial model using the Poisson-selected formula.")
    nb_fallback_fit = fit_capture(
        lambda: fit_negative_binomial(
            teams_2_3,
            selected_terms(poisson_step),
            maxiter=100
        )
    )

    if nb_fallback_fit["error"] is not None:
        nb_fit_status = "full and fallback models failed"
        print("Negative binomial fallback model also failed.")
        print("Error message:")
        print(str(nb_fallback_fit["value"]), "\n")

        if len(nb_fallback_fit["warnings"]) > 0:
            print("Fallback warnings:")
            print(nb_fallback_fit["warnings"])
            print()
    else:
        nb_model = nb_fallback_fit["value"]
        nb_step_model = nb_model
        nb_fit_status = "fallback model fitted from Poisson-selected formula"
        print("Negative binomial fallback model fitted.")

        if len(nb_fallback_fit["warnings"]) > 0:
            print("Fallback warnings:")
            print(nb_fallback_fit["warnings"])
            print()
else:
    nb_model = nb_full_fit["value"]
    print("Negative binomial full model fitted.")

    if len(nb_full_fit["warnings"]) > 0:
        print("Full-model warnings:")
        print(nb_full_fit["warnings"])
        print()

    nb_step_fit = fit_capture(
        lambda: stepwise_aic(
            data=teams_2_3,
            all_predictors=predictors_2_3,
            start_predictors=predictors_2_3,
            fit_func=lambda data, predictors: fit_negative_binomial(
                data,
                predictors,
                maxiter=100
            ),
            trace=1
        )
    )

    if nb_step_fit["error"] is not None:
        nb_fit_status = "full model fitted but stepAIC failed"
        print("Negative binomial stepAIC failed.")
        print("Error message:")
        print(str(nb_step_fit["value"]), "\n")
        nb_step_model = nb_model
    else:
        nb_fit_status = "stepwise model fitted"
        nb_step_model = nb_step_fit["value"]

        if len(nb_step_fit["warnings"]) > 0:
            print("StepAIC warnings:")
            print(nb_step_fit["warnings"])
            print()


print("Negative binomial fit status:", nb_fit_status, "\n")


# 음이항 적합에서의 문제 진단
design_kappa = condition_number(predictors_2_3, teams_2_3)

print("Diagnostics for possible negative-binomial fitting issues")
print("Poisson full-model Pearson dispersion:", poisson_disp_full)
print("Poisson step-model Pearson dispersion:", poisson_disp_step)
print("Full count-model condition number:", design_kappa)

if design_kappa > 1000:
    print(
        "The condition number is large, suggesting strong multicollinearity among predictors. "
        "This can make the full negative-binomial model or stepwise updates unstable."
    )

print()


def nb_theta(model):
    alpha = model.params.get("alpha", np.nan)

    if pd.isna(alpha) or alpha <= 0:
        return np.nan

    return 1 / alpha


def nb_theta_se(model):
    alpha = model.params.get("alpha", np.nan)

    if pd.isna(alpha) or alpha <= 0:
        return np.nan

    if hasattr(model, "bse") and "alpha" in model.bse.index:
        alpha_se = model.bse["alpha"]
        return alpha_se / (alpha ** 2)

    return np.nan


if nb_step_model is not None:
    print("Negative binomial selected/final model")
    print(count_formula_text("W", selected_terms(nb_step_model)))
    print(nb_step_model.summary())

    theta_hat = nb_theta(nb_step_model)
    theta_se = nb_theta_se(nb_step_model)

    print("Estimated theta:", theta_hat)
    print("SE(theta):", theta_se)

    if theta_hat > 1000:
        print("Theta is very large, so the negative binomial model is nearly Poisson.")

    print()


# 예측성능평가
def prediction_metrics(model, data, model_name, model_type):
    predictors = selected_terms(model)

    if model_type == "logistic":
        X = data[predictors].copy()
        X = sm.add_constant(X, has_constant="add")

        pred_wpct = model.predict(X)
        pred_w = pred_wpct * data["G"]

    else:
        X = data[predictors].copy()
        X = sm.add_constant(X, has_constant="add")
        offset = np.log(data["G"])

        if model_type == "negative_binomial":
            pred_w = model.predict(X, offset=offset)
        else:
            pred_w = model.predict(X, offset=offset)

        pred_wpct = pred_w / data["G"]

    if hasattr(model, "deviance"):
        residual_deviance = model.deviance
    else:
        residual_deviance = np.nan

    if hasattr(model, "df_resid"):
        residual_df = model.df_resid
    else:
        residual_df = len(data) - len(model.params)

    return pd.DataFrame({
        "model": [model_name],
        "selected_variables": [", ".join(predictors)],
        "df": [len(model.params)],
        "logLik": [model.llf],
        "AIC": [model.aic],
        "residual_deviance": [residual_deviance],
        "residual_df": [residual_df],
        "W_MAE": [np.mean(np.abs(data["W"] - pred_w))],
        "W_RMSE": [np.sqrt(np.mean((data["W"] - pred_w) ** 2))],
        "WPct_MAE": [
            np.average(np.abs(data["WPct"] - pred_wpct), weights=data["G"])
        ],
        "WPct_RMSE": [
            np.sqrt(
                np.average((data["WPct"] - pred_wpct) ** 2, weights=data["G"])
            )
        ]
    })


# 로지스틱 ~ 포아송 비교
comparison_table = pd.concat(
    [
        prediction_metrics(
            logistic_step,
            teams_2_3,
            "Problem 2-2 logistic stepwise",
            "logistic"
        ),
        prediction_metrics(
            poisson_step,
            teams_2_3,
            "Problem 2-3 Poisson stepwise",
            "count"
        )
    ],
    ignore_index=True
)

# 음이항회귀 적합 성공 시 함께 비교
if nb_step_model is not None:
    comparison_table = pd.concat(
        [
            comparison_table,
            prediction_metrics(
                nb_step_model,
                teams_2_3,
                "Problem 2-3 negative binomial",
                "negative_binomial"
            )
        ],
        ignore_index=True
    )

print("서로 다른 모형 계열을 비교할 때는 AIC보다는 W_MAE 등의 예측오차 지표를 더 중요하게 보는 것이 적절하다.")
print(comparison_table)