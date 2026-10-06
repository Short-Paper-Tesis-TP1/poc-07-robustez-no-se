"""
PoC de robustez — "No sé" en el formulario
==========================================
Objetivo único: medir cuánto empeora el modelo cuando la usuaria responde "No sé"
en las preguntas que pueden ser difíciles de conocer:
  4. ¿Cuántos otros créditos tienes vigentes?      -> n_otros_creditos
  5. ¿Cuánto debes hoy en esos otros créditos?     -> ratio_deuda_ingreso
  6. ¿Alguna vez te atrasaste en un pago?           -> atraso_historial

Criterio de aceptación (fijado ANTES de correr):
  - Caída de AUC-ROC <= 0.02 con "No sé" en UNA pregunta.
  - Caída de AUC-ROC <= 0.03 con "No sé" en las TRES a la vez.
  - Caída de AUC-ROC <= 0.02 con "No sé" en 5 y 6 (diseño final: la 4 es obligatoria).
  - Se exige en todas las mujeres Y en el subgrupo emprendedoras.

Control: "No sé" en edad, variable que NUNCA falta en el entrenamiento.
Sirve para mostrar por qué "No sé" solo se ofrece en las preguntas 4, 5 y 6:
LightGBM trata como 0 un valor faltante en una variable que no tuvo faltantes al entrenar.

Datos: Home Credit (application_train, bureau, bureau_balance), solo mujeres.
Uso:   python poc_robustez.py "C:\\ruta\\home-credit-default-risk"
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

RUTA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\alexa\Escritorio\UPC\TP1\home-credit-default-risk"
SEMILLA = 42


def csv(nombre):
    return os.path.join(RUTA, f"dataset-{nombre}.csv")


# Diccionario v1: nombres, orden y restricción monotónica (+1 = nunca baja el riesgo; 0 = libre)
COLUMNAS = ["ratio_cuota_ingreso", "ratio_deuda_ingreso", "ratio_monto_ingreso", "n_otros_creditos",
            "atraso_historial", "edad", "nivel_educativo", "n_hijos"]
MONOTONIA = [1, 1, 1, 1, 1, 0, 0, 0]
EDUCACION = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
             "Higher education": 3, "Academic degree": 3}

t0 = time.time()

# ---------------------------------------------------------------- 1. Datos
app = pd.read_csv(csv("application_train"), usecols=[
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT",
    "DAYS_BIRTH", "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE"])
w = app[app.CODE_GENDER == "F"].copy()

bur = pd.read_csv(csv("bureau"), usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT"])
activos = (bur[bur.CREDIT_ACTIVE == "Active"]
           .assign(deuda=lambda d: d.AMT_CREDIT_SUM_DEBT.fillna(0))
           .groupby("SK_ID_CURR").agg(n=("SK_ID_BUREAU", "count"), deuda=("deuda", "sum")))
w = w.merge(activos, left_on="SK_ID_CURR", right_index=True, how="left")
en_buro = w.SK_ID_CURR.isin(bur.SK_ID_CURR)

bb = pd.read_csv(csv("bureau_balance"), usecols=["SK_ID_BUREAU", "STATUS"],
                 dtype={"SK_ID_BUREAU": "int64", "STATUS": "category"})  # category: menos RAM
ids_bb = bb.SK_ID_BUREAU.unique()
ids_atraso = bb.loc[bb.STATUS.isin(["1", "2", "3", "4", "5"]), "SK_ID_BUREAU"].unique()
del bb
clientes_con_bb = bur.loc[bur.SK_ID_BUREAU.isin(ids_bb), "SK_ID_CURR"].unique()
clientes_con_atraso = bur.loc[bur.SK_ID_BUREAU.isin(ids_atraso), "SK_ID_CURR"].unique()

# ---------------------------------------------------------------- 2. Variables del diccionario v1
X = pd.DataFrame(index=w.index)
X["ratio_cuota_ingreso"] = w.AMT_ANNUITY / w.AMT_INCOME_TOTAL
deuda_otros = np.where(en_buro, w.deuda.fillna(0), np.nan)            # sin buró = desconocido
X["ratio_deuda_ingreso"] = (w.AMT_CREDIT + deuda_otros) / w.AMT_INCOME_TOTAL
X["ratio_monto_ingreso"] = w.AMT_CREDIT / w.AMT_INCOME_TOTAL
X["n_otros_creditos"] = np.where(en_buro, w.n.fillna(0), np.nan)      # en buró sin activos = 0
X["atraso_historial"] = np.where(w.SK_ID_CURR.isin(clientes_con_atraso), 1.0,
                                 np.where(w.SK_ID_CURR.isin(clientes_con_bb), 0.0, np.nan))
X["edad"] = -w.DAYS_BIRTH / 365.25
X["nivel_educativo"] = w.NAME_EDUCATION_TYPE.map(EDUCACION)
X["n_hijos"] = w.CNT_CHILDREN
X = X[COLUMNAS]
y = w.TARGET.values
emprendedora = ((w.NAME_INCOME_TYPE == "Commercial associate") | (w.ORGANIZATION_TYPE == "Self-employed")).values

faltantes = {c: round(float(X[c].isna().mean() * 100), 2) for c in COLUMNAS}

# ---------------------------------------------------------------- 3. Partición 70 / 15 / 15 estratificada
idx = np.arange(len(X))
i_tr, i_resto = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEMILLA)
i_cal, i_te = train_test_split(i_resto, test_size=0.50, stratify=y[i_resto], random_state=SEMILLA)

# ---------------------------------------------------------------- 4. Entrenamiento (con restricciones monotónicas)
t1 = time.time()
modelo = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                            monotone_constraints=MONOTONIA, random_state=SEMILLA, verbose=-1)
modelo.fit(X.iloc[i_tr], y[i_tr])
seg_entrenamiento = time.time() - t1
calibrador = IsotonicRegression(out_of_bounds="clip").fit(modelo.predict_proba(X.iloc[i_cal])[:, 1], y[i_cal])

# ---------------------------------------------------------------- 5. Escenarios de "No sé" sobre el conjunto de prueba
escenarios = {
    "Completo (sin 'No sé')": [],
    "No sé en 4 (otros créditos)": ["n_otros_creditos"],
    "No sé en 5 (deuda)": ["ratio_deuda_ingreso"],
    "No sé en 6 (atrasos)": ["atraso_historial"],
    "No sé en 5 y 6 (4 obligatoria)": ["ratio_deuda_ingreso", "atraso_historial"],
    "No sé en 4, 5 y 6": ["n_otros_creditos", "ratio_deuda_ingreso", "atraso_historial"],
    "CONTROL: No sé en edad": ["edad"],
}


def medir(Xp, yp):
    p = modelo.predict_proba(Xp)[:, 1]
    return {"auc_roc": roc_auc_score(yp, p), "auc_pr": average_precision_score(yp, p),
            "brier_calibrado": brier_score_loss(yp, calibrador.predict(p))}


Xte, yte, emp_te = X.iloc[i_te], y[i_te], emprendedora[i_te]
resultados = {}
for nombre, vars_nan in escenarios.items():
    Xs = Xte.copy()
    Xs[vars_nan] = np.nan
    resultados[nombre] = {"todas": medir(Xs, yte), "emprendedoras": medir(Xs[emp_te], yte[emp_te])}

base = resultados["Completo (sin 'No sé')"]
for nombre, r in resultados.items():
    for grupo in ("todas", "emprendedoras"):
        r[grupo]["caida_auc"] = base[grupo]["auc_roc"] - r[grupo]["auc_roc"]


def cumple(nombre, limite):
    r = resultados[nombre]
    return all(r[g]["caida_auc"] <= limite for g in ("todas", "emprendedoras"))


veredicto = {
    "No sé en 4": cumple("No sé en 4 (otros créditos)", 0.02),
    "No sé en 5": cumple("No sé en 5 (deuda)", 0.02),
    "No sé en 6": cumple("No sé en 6 (atrasos)", 0.02),
    "No sé en 5 y 6": cumple("No sé en 5 y 6 (4 obligatoria)", 0.02),
    "No sé en 4, 5 y 6": cumple("No sé en 4, 5 y 6", 0.03),
}

# ---------------------------------------------------------------- 6. Reporte
print(f"\nMujeres: {len(X):,} | emprendedoras: {emprendedora.sum():,} | TARGET=1: {y.mean()*100:.2f} %")
print(f"Prueba: {len(i_te):,} filas ({emp_te.sum():,} emprendedoras)")
print(f"Entrenamiento: {seg_entrenamiento:.1f} s | total: {time.time()-t0:.1f} s")
print("\n% de faltantes que el modelo YA VIO al entrenar:")
for c, v in faltantes.items():
    print(f"  {c:22s} {v:6.2f} %")
print(f"\n{'Escenario':32s} {'AUC todas':>10s} {'caída':>7s} {'AUC empr.':>10s} {'caída':>7s} {'AUC-PR':>7s} {'Brier':>7s}")
for nombre, r in resultados.items():
    t, e = r["todas"], r["emprendedoras"]
    print(f"{nombre:32s} {t['auc_roc']:10.4f} {t['caida_auc']:7.4f} {e['auc_roc']:10.4f} {e['caida_auc']:7.4f} "
          f"{t['auc_pr']:7.4f} {t['brier_calibrado']:7.4f}")
print("\nVeredicto (criterio fijado antes de correr):")
for k, v in veredicto.items():
    print(f"  {k:20s} {'CUMPLE' if v else 'NO CUMPLE'}")

salida = {"poblacion": {"mujeres": int(len(X)), "emprendedoras": int(emprendedora.sum()),
                        "tasa_target": float(y.mean())},
          "faltantes_entrenamiento_pct": faltantes,
          "segundos_entrenamiento": round(seg_entrenamiento, 1),
          "criterio": {"una_pregunta": 0.02, "tres_preguntas": 0.03},
          "resultados": resultados, "veredicto": veredicto}
with open("resultado_poc_robustez.json", "w", encoding="utf-8") as fh:
    json.dump(salida, fh, indent=2, ensure_ascii=False)
print("\nGuardado: resultado_poc_robustez.json")
