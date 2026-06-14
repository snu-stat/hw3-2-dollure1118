# Problem 2-1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import t, chi2



Teams=pd.read_csv("data/Teams.csv")


# 데이터 전처리: 팀-연도 데이터 생성
teams_2_1 = Teams[
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

teams_2_1["G"] = teams_2_1["W"] + teams_2_1["L"]
teams_2_1["RS"] = teams_2_1["R"]
teams_2_1["WPct"] = teams_2_1["W"] / teams_2_1["G"]
teams_2_1["log_ratio"] = np.log(teams_2_1["RS"] / teams_2_1["RA"])
teams_2_1["logRS"] = np.log(teams_2_1["RS"])
teams_2_1["logRA"] = np.log(teams_2_1["RA"])
teams_2_1["logit_WPct"] = np.log(
    teams_2_1["WPct"] / (1 - teams_2_1["WPct"])
)

teams_2_1 = teams_2_1[
    [
        "yearID", "franchID", "teamID", "W", "L", "G",
        "RS", "RA", "WPct", "log_ratio",
        "logRS", "logRA", "logit_WPct"
    ]
]

teams_2_1 = teams_2_1[
    (teams_2_1["WPct"] > 0) &
    (teams_2_1["WPct"] < 1)
].copy()


# 1. Bill James 공식
def bill_james_formula(x, k):
    RS, RA = x
    return 1 / (1 + (RA / RS) ** k)


popt, pcov = curve_fit(
    bill_james_formula,
    (teams_2_1["RS"], teams_2_1["RA"]),
    teams_2_1["WPct"],
    p0=[2],
    bounds=(0, np.inf)
)

k_hat = popt[0]
print("k의 점추정량:", k_hat)

# 95% 신뢰구간
n = len(teams_2_1)
p = 1
df = n - p
se_k = np.sqrt(np.diag(pcov))[0]

k_ci = np.array([
    k_hat - t.ppf(0.975, df=df) * se_k,
    k_hat + t.ppf(0.975, df=df) * se_k
])

print("95% 신뢰구간:", k_ci)


# 2. 절편이 없는 로지스틱 회귀
X_ratio = teams_2_1[["log_ratio"]]
y = teams_2_1["WPct"]

glm_ratio = sm.GLM(
    y,
    X_ratio,
    family=sm.families.Binomial(),
    freq_weights=teams_2_1["G"]
).fit()

beta1_hat = glm_ratio.params["log_ratio"]
beta1_ci = glm_ratio.conf_int().loc["log_ratio"]

print("beta의 점추정량, 95% 신뢰구간:")
print(beta1_hat)
print(beta1_ci)
print("1의 결과와 매우 유사")


# 3. 모형적합결과 진단
teams_2_1["eta_ratio"] = glm_ratio.predict(X_ratio, linear=True)
teams_2_1["pred_ratio"] = glm_ratio.predict(X_ratio)
teams_2_1["pred_ratio_manual"] = 1 / (1 + np.exp(-teams_2_1["eta_ratio"]))

# Deviance residuals
teams_2_1["deviance_resid"] = glm_ratio.resid_deviance


# i. Residual deviance에 대한 해석
resid_dev = glm_ratio.deviance
resid_df = glm_ratio.df_resid
resid_dev_p = chi2.sf(resid_dev, df=resid_df)

print("3-i. Residual deviance diagnostic")
print("Residual deviance:", resid_dev)
print("Residual df:", resid_df)
print("Chi-square upper-tail p-value:", resid_dev_p)
print(
    "Small p-value means the model has lack of fit "
    "relative to the binomial GLM assumption.\n"
)


# ii. Deviance residuals vs linear predictors
plt.figure(figsize=(8, 5))
plt.axhline(0, color="gray", linewidth=0.5)
plt.scatter(
    teams_2_1["eta_ratio"],
    teams_2_1["deviance_resid"],
    alpha=0.75
)
plt.title("Deviance residuals vs linear predictors")
plt.xlabel(r"$\eta = \hat{\beta}_1 \log(RS / RA)$")
plt.ylabel("Deviance residual")
plt.tight_layout()
plt.show()

print(
    "그래프가 곡선 패턴이 뚜렷하거나 잔차가 한쪽으로 몰리지 않고 "
    "0 주변에 무작위로 흩어져 있는 모형이 된다."
)


# logRA와 logRS를 따로 넣은 모형
X_logs = teams_2_1[["logRA", "logRS"]]

glm_logs = sm.GLM(
    y,
    X_logs,
    family=sm.families.Binomial(),
    freq_weights=teams_2_1["G"]
).fit()

coef_logs = glm_logs.params
ci_logs = glm_logs.conf_int()

print("logRA, logRS 점추정량, 95% 신뢰구간:")
print(coef_logs)
print(ci_logs)


# 1, 2항의 추정 계수와 비교
estimate_compare_all = pd.DataFrame(
    {
        "estimate": [
            k_hat,
            beta1_hat,
            coef_logs["logRS"],
            -coef_logs["logRA"]
        ],
        "lower": [
            k_ci[0],
            beta1_ci[0],
            ci_logs.loc["logRS", 0],
            -ci_logs.loc["logRA", 1]
        ],
        "upper": [
            k_ci[1],
            beta1_ci[1],
            ci_logs.loc["logRS", 1],
            -ci_logs.loc["logRA", 0]
        ]
    },
    index=[
        "Bill James nls: k",
        "Logistic glm: beta1",
        "Logistic glm: logRS coef",
        "Logistic glm: -logRA coef"
    ]
)

print(estimate_compare_all)

print(
    "logRS coefficient + logRA coefficient:",
    coef_logs["logRS"] + coef_logs["logRA"]
)

print(
    "log(RS)와 log(RA)를 따로 넣어도, "
    "log(RS/RA) 하나를 넣은 모형과 거의 다르지 않다."
)


# Problem 2-2

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2



Teams = pd.read_csv("data/Teams.csv")



# 데이터 전처리
predictors_2_2 = [
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

teams_2_2 = Teams[
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

teams_2_2["G"] = teams_2_2["W"] + teams_2_2["L"]
teams_2_2["WPct"] = teams_2_2["W"] / teams_2_2["G"]
teams_2_2["RS"] = teams_2_2["R"]
teams_2_2["logRS"] = np.log(teams_2_2["R"])
teams_2_2["logRA"] = np.log(teams_2_2["RA"])
teams_2_2["log_ratio"] = np.log(teams_2_2["R"] / teams_2_2["RA"])

teams_2_2 = teams_2_2[
    [
        "yearID", "franchID", "teamID",
        "W", "L", "G", "WPct", "RS", "RA",
        "logRS", "logRA", "log_ratio",
        "H", "X2B", "X3B", "HR", "BB", "SO", "CS", "HBP", "SF",
        "ERA", "CG", "SHO", "IPouts", "HA", "HRA", "BBA", "SOA",
        "E", "DP", "FP", "SV"
    ]
].copy()

teams_2_2 = teams_2_2[
    (teams_2_2["WPct"] > 0) &
    (teams_2_2["WPct"] < 1)
].copy()

model_vars = ["WPct", "G", "log_ratio"] + predictors_2_2
teams_2_2 = teams_2_2.dropna(subset=model_vars).copy()


def fit_glm(data, predictors, intercept=True, maxiter=100):
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

    return model


def formula_text(response, predictors, intercept=True):
    if len(predictors) == 0:
        return f"{response} ~ 1"

    rhs = " + ".join(predictors)

    if intercept:
        return f"{response} ~ {rhs}"
    else:
        return f"{response} ~ 0 + {rhs}"


# 일단 모든 변수 적합
full_formula = formula_text("WPct", predictors_2_2, intercept=True)
null_formula = "WPct ~ 1"

full_model = fit_glm(teams_2_2, predictors_2_2, intercept=True, maxiter=100)

print("Full model summary")
print(full_model.summary())


# AIC를 기준으로 하는 단계별(stepwise) 변수선택
# Python statsmodels에는 R의 step() 함수가 기본 제공되지 않으므로,
# AIC 기준 양방향 stepwise selection을 직접 구현함.
def stepwise_aic(data, all_predictors, start_predictors, trace=1):
    current_predictors = list(start_predictors)
    current_model = fit_glm(data, current_predictors, intercept=True)
    current_aic = current_model.aic

    while True:
        candidates = []

        # drop step
        for var in current_predictors:
            new_predictors = [x for x in current_predictors if x != var]
            model = fit_glm(data, new_predictors, intercept=True)
            candidates.append(("drop", var, new_predictors, model.aic, model))

        # add step
        remaining_predictors = [
            x for x in all_predictors if x not in current_predictors
        ]
        for var in remaining_predictors:
            new_predictors = current_predictors + [var]
            model = fit_glm(data, new_predictors, intercept=True)
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

    return current_model, current_predictors


step_model, step_predictors = stepwise_aic(
    data=teams_2_2,
    all_predictors=predictors_2_2,
    start_predictors=predictors_2_2,
    trace=1
)

print("\nStepwise AIC selected model")
print(formula_text("WPct", step_predictors, intercept=True))
print(step_model.summary())


print("AIC of full model and stepwise model")
aic_table = pd.DataFrame(
    {
        "df": [
            full_model.df_model + 1,
            step_model.df_model + 1
        ],
        "AIC": [
            full_model.aic,
            step_model.aic
        ]
    },
    index=["full_model", "step_model"]
)
print(aic_table)


# 남은 변수들을 모두 모형에 남길지 일부를 제거할지
# R의 drop1(step_model, test = "Chisq")에 해당하는 계산을 직접 구현함.
def drop1_chisq(data, current_predictors, current_model):
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
        reduced_model = fit_glm(data, reduced_predictors, intercept=True)

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


drop_check = drop1_chisq(teams_2_2, step_predictors, step_model)

current_aic = step_model.aic
drop_check_terms = drop_check.loc[drop_check.index != "<none>"].copy()

candidate_drops = drop_check_terms[
    drop_check_terms["AIC"].notna() &
    drop_check_terms["Pr(>Chi)"].notna() &
    (drop_check_terms["AIC"] <= current_aic + 2) &
    (drop_check_terms["Pr(>Chi)"] > 0.05)
].index.tolist()

print("Variables to consider dropping after stepwise selection:")
if len(candidate_drops) == 0:
    print(
        "None by the rule: AIC within 2 of selected model "
        "and LRT p-value > 0.05.\n"
    )
else:
    print(candidate_drops)
    print()


# 최종모형 설정
final_model = step_model
final_predictors = step_predictors

print(formula_text("WPct", final_predictors, intercept=True))


# 문제1의 모형과 비교
problem_2_1_model = fit_glm(
    teams_2_2,
    ["log_ratio"],
    intercept=False,
    maxiter=100
)


def model_metrics(model, data, model_name, predictors, intercept=True):
    X = data[predictors].copy()

    if intercept:
        X = sm.add_constant(X, has_constant="add")

    pred = model.predict(X)

    return pd.DataFrame({
        "model": [model_name],
        "df": [model.df_model + 1],
        "logLik": [model.llf],
        "AIC": [model.aic],
        "residual_deviance": [model.deviance],
        "residual_df": [model.df_resid],
        "weighted_MAE": [
            np.average(np.abs(data["WPct"] - pred), weights=data["G"])
        ],
        "weighted_RMSE": [
            np.sqrt(
                np.average((data["WPct"] - pred) ** 2, weights=data["G"])
            )
        ]
    })


comparison_table = pd.concat(
    [
        model_metrics(
            problem_2_1_model,
            teams_2_2,
            "Problem 1",
            ["log_ratio"],
            intercept=False
        ),
        model_metrics(
            final_model,
            teams_2_2,
            "AIC final model",
            final_predictors,
            intercept=True
        )
    ],
    ignore_index=True
)

print("Model comparison on the same observations")
print(comparison_table)
print("모형의 복잡도가 올라갔음에도 가중 MAE, RMSE 모두 감소했다. 특히 AIC가 유의미하게 감소했다. ")
print("따라서 final model이 더 좋은 성능을 보인다. ")



# ::: {.panel-tabset}

# ## R

# ```{r}

# ```

# ## Python

# ```{python}

# ```

# :::