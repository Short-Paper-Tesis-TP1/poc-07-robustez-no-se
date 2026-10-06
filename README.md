# PoC 07: robustez del modelo ante respuestas "No sé"

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Medir cuánto cae el AUC-ROC cuando la usuaria responde "No sé" en tres preguntas: otros créditos vigentes (4), deuda en otros créditos (5) y atrasos (6).

## Datos
Home Credit Default Risk (Kaggle, 2018): `application_train`, `bureau` y `bureau_balance`. Los CSV no se incluyen en el repositorio porque las reglas de la competencia no permiten redistribuirlos. Se descargan de https://www.kaggle.com/competitions/home-credit-default-risk/data.

Población: mujeres (`CODE_GENDER == "F"`). El subgrupo emprendedoras son las clientas con `NAME_INCOME_TYPE == "Commercial associate"` u `ORGANIZATION_TYPE == "Self-employed"`, y se reporta por separado.

## Método
- Modelo preliminar de 8 variables, entrenado con LGBMClassifier.
- Partición estratificada 70/15/15 con semilla 42.
- Sobre el conjunto de prueba, cada respuesta se reemplaza por un valor desconocido: una pregunta a la vez y luego las tres juntas.
- Control: el mismo reemplazo en la edad, una variable que nunca tuvo faltantes en el entrenamiento.

**Criterio, fijado antes de correr:** la caída del AUC-ROC debe ser de 0.02 como máximo por pregunta y de 0.03 como máximo con las tres juntas, en ambos grupos.

## Cómo correrla
```
pip install -r requirements.txt
python poc_robustez.py "C:\ruta\home-credit-default-risk"
```

## Resultado
| Escenario | AUC-ROC todas | Caída | AUC-ROC emprendedoras | Caída | Criterio |
|---|---|---|---|---|---|
| Completo (sin 'No sé') | 0.6412 | +0.0000 | 0.6277 | +0.0000 | - |
| No sé en 4 (otros créditos) | 0.6172 | +0.0240 | 0.6024 | +0.0252 | No cumple |
| No sé en 5 (deuda) | 0.6420 | -0.0008 | 0.6252 | +0.0025 | Cumple |
| No sé en 6 (atrasos) | 0.6375 | +0.0037 | 0.6225 | +0.0052 | Cumple |
| No sé en 5 y 6 (4 obligatoria) | 0.6387 | +0.0025 | 0.6206 | +0.0071 | Cumple |
| No sé en 4, 5 y 6 | 0.6061 | +0.0351 | 0.5859 | +0.0417 | No cumple |
| CONTROL: No sé en edad | 0.5962 | +0.0450 | 0.5792 | +0.0484 | (control) |

Caída = AUC-ROC del escenario completo menos el del escenario indicado (positivo = empeora). Muestra: 202,448 mujeres, 62,091 emprendedoras. Criterio: caída <= 0.02 por pregunta y <= 0.03 con las tres juntas, en ambos grupos.

## Decisión de diseño
- La pregunta 4 (número de otros créditos vigentes) es obligatoria, con la ayuda "Puedes verlo gratis en tu Reporte de Deudas de la SBS".
- "No sé" se permite solo en las preguntas 5 y 6.

## Archivos
- `poc_robustez.py`: script.
- `resultado_poc_robustez.json`: salida estructurada.
- `salida_robustez.txt`: salida completa de la consola.
