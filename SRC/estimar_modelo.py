import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# 1. Asegurar directorios de salida
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# 2. Cargar datos limpios
df = pd.read_csv("data/processed/enemdu_limpia.csv")

# 3. Especificación de la fórmula econométrica
formula = "pobreza_ing ~ anios_educ + edad + es_mujer + es_rural + es_informal + horas_trabajo"

# 4. Estimación de Modelo LOGIT
modelo_logit = smf.logit(formula, data=df).fit()
print("="*60)
print("             RESULTADOS MODELO LOGIT")
print("="*60)
print(modelo_logit.summary())

# 5. Estimación de Modelo PROBIT
modelo_probit = smf.probit(formula, data=df).fit()
print("\n" + "="*60)
print("             RESULTADOS MODELO PROBIT")
print("="*60)
print(modelo_probit.summary())

# 6. Efectos Marginales Promedio (AME) para LOGIT
ame_logit = modelo_logit.get_margeff(at='overall', method='dydx')
print("\n" + "="*60)
print("   EFECTOS MARGINALES PROMEDIO (AME) - LOGIT")
print("="*60)
print(ame_logit.summary())

# 7. Guardar Resumen en Texto para el Minipaper
with open("outputs/tables/resumen_modelos.txt", "w", encoding="utf-8") as f:
    f.write("=== MODELO LOGIT ===\n")
    f.write(str(modelo_logit.summary()))
    f.write("\n\n=== EFECTOS MARGINALES (AME) ===\n")
    f.write(str(ame_logit.summary()))

# 8. Gráfico de Efectos Marginales
fig, ax = plt.subplots(figsize=(8, 5))
summary_ame = ame_logit.summary_frame()
variables = ['Años Educ.', 'Edad', 'Es Mujer', 'Es Rural', 'Es Informal', 'Horas Trab.']
valores_ame = summary_ame['dy/dx']

# Identificar automáticamente la columna de error estándar
col_err = 'Std. Err.' if 'Std. Err.' in summary_ame.columns else summary_ame.columns[1]
errores = summary_ame[col_err]

colors = ['green' if x < 0 else 'red' for x in valores_ame]
ax.barh(variables, valores_ame, xerr=errores*1.96, capsize=5, color=colors, alpha=0.7)
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_title('Efectos Marginales Promedio (AME) sobre la Pobreza', fontsize=12, fontweight='bold')
ax.set_xlabel('Cambio en la Probabilidad de Pobreza (puntos porcentuales)')
plt.tight_layout()
plt.savefig("outputs/figures/efectos_marginales.png", dpi=300)
plt.close()

print("\n" + "="*60)
print(" ¡PROCESO COMPLETO! Modelos estimados y gráficos guardados en outputs/")
print("="*60)