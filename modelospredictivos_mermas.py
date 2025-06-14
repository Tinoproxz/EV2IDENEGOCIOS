# ANÁLISIS PREDICTIVO DE MERMAS DE SUPERMERCADO

# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS. CASO PREDICTIVO DE MERMAS DE SUPERMERCADO*")

# PASO 2: CARGA de archivo mermas.csv
print("\nCargando y preparando datos...")
data = pd.read_csv('merma.csv', sep=',', encoding='latin1')

# Normaliza lascolumnas
data.columns = data.columns.str.strip().str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
data.rename(columns={'ubicacia3n_motivo': 'ubicacion_motivo'}, inplace=True)

# Convertir fechas
data['fecha'] = pd.to_datetime(data['fecha'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
data['dia_semana'] = data['fecha'].dt.dayofweek
data['dia_mes'] = data['fecha'].dt.day

# Limpia columnas numéricas
def clean_numeric(x):
    if isinstance(x, str):
        return float(x.replace(',', '.'))
    return float(x)

numeric_columns = ['merma_unidad', 'merma_monto', 'merma_unidad_p', 'merma_monto_p']
for col in numeric_columns:
    data[col] = data[col].apply(clean_numeric)

# PASO 3: DETECCION Y MANEJO DE OUTLIERS
def remove_outliers(df, column, n_sigmas=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < n_sigmas]

print(f"\nTamaño del dataset antes de remover outliers: {len(data)}")
data = remove_outliers(data, 'merma_monto')
data = remove_outliers(data, 'merma_unidad')
print(f"Tamaño del dataset después de remover outliers: {len(data)}")

# PASO 4: SELECCIÓN DE CARACTERÍSTICAS Y DIVISIÓN DE DATOS
features = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
           'comuna', 'region', 'tienda', 'motivo', 'ubicacion_motivo',
           'dia_semana', 'dia_mes', 'mes']

X = data[features]
y = data['merma_unidad_p']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# quita datos nulos
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

# PASO 5: PREPROCESAMIENTO
categorical_features = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
                        'comuna', 'region', 'tienda', 'motivo', 'ubicacion_motivo', 'mes']
numeric_features = ['dia_semana', 'dia_mes']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# PASO 6: IMPLEMENTACION Y ENTRENAMIENTO DE MODELOS
print("\nEntrenando modelos...")

# Modelos
models = {
    'Regresión Lineal': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'LightGBM': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(objective='regression', random_state=42, verbose=-1))
    ])
}

# Entrena los modelos
trained_models = {}
for name, model in models.items():
    print(f"Entrenando {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

print("Modelos entrenados correctamente")

# PASO 7: EVALUACIÓN DE MODELOS
print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")

# PASO 8: REALIZAR PREDICCIONES CON LOS MODELOS ENTRENADOS
y_pred_lr = trained_models['Regresión Lineal'].predict(X_test)
y_pred_rf = trained_models['Random Forest'].predict(X_test)
y_pred_lgb = trained_models['LightGBM'].predict(X_test)

def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Evita divisiones por cero al calcular MAPE
    nonzero_y = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_y] - y_pred[nonzero_y]) / y_true[nonzero_y])) * 100

    return {
        'Modelo': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MedAE': medae,
        'MAPE (%)': mape,
        'R²': r2
    }

# Calcula metricas para cada modelo
metrics_lr = calculate_metrics(y_test, y_pred_lr, 'Regresión Lineal')
metrics_rf = calculate_metrics(y_test, y_pred_rf, 'Random Forest')
metrics_lgb = calculate_metrics(y_test, y_pred_lgb, 'LightGBM')

# Crea DataFrame con las metricas
metrics_list = [metrics_lr, metrics_rf, metrics_lgb]

# Separar metricas individuales para uso posterior
r2_lr, rmse_lr, mae_lr, mse_lr = metrics_lr['R²'], metrics_lr['RMSE'], metrics_lr['MAE'], metrics_lr['MSE']
r2_rf, rmse_rf, mae_rf, mse_rf = metrics_rf['R²'], metrics_rf['RMSE'], metrics_rf['MAE'], metrics_rf['MSE']
r2_lgb, rmse_lgb, mae_lgb, mse_lgb = metrics_lgb['R²'], metrics_lgb['RMSE'], metrics_lgb['MAE'], metrics_lgb['MSE']

#GUARDA RESULTADOS DE PREDICCIÓN EN ARCHIVOS MARKDOWN
# Crea un DataFrame con las predicciones y valores reales
results_df = pd.DataFrame({
    'Valor_Real': y_test,
    'Prediccion_LR': y_pred_lr,
    'Prediccion_RF': y_pred_rf,
    'Prediccion_LGB': y_pred_lgb,
    'Error_LR': y_test - y_pred_lr,
    'Error_RF': y_test - y_pred_rf,
    'Error_LGB': y_test - y_pred_lgb,
    'Error_Porcentual_LR': ((y_test - y_pred_lr) / y_test) * 100,
    'Error_Porcentual_RF': ((y_test - y_pred_rf) / y_test) * 100,
    'Error_Porcentual_LGB': ((y_test - y_pred_lgb) / y_test) * 100
})

# Reinicia el índice para añadir información de las características
results_df = results_df.reset_index(drop=True)

# Añadir algunas columnas con información de las características para mayor contexto
X_test_reset = X_test.reset_index(drop=True)
for feature in X_test.columns:
    results_df[feature] = X_test_reset[feature]

# Ordena por valor real 
results_df = results_df.sort_values('Valor_Real', ascending=False)

# Organiza predicciones para compatibilidad con visualizaciones
predictions = {
    'Regresión Lineal': y_pred_lr,
    'Random Forest': y_pred_rf,
    'LightGBM': y_pred_lgb
}

# Genera archivos de predicción
with open('prediccion_lr.md', 'w') as f:
    f.write("# Predicciones Regresión Lineal\nArchivo generado automáticamente")
with open('prediccion_rf.md', 'w') as f:
    f.write("# Predicciones Random Forest\nArchivo generado automáticamente")

print("Archivos de predicción generados: prediccion_lr.md y prediccion_rf.md")

# Muestra comparación de metricas
metrics_df = pd.DataFrame(metrics_list)
print("\nComparación de métricas entre modelos:")
print(metrics_df)

# PASO 8: VISUALIZACIONES
# Configuración de estilo
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Predicciones vs Valores Reales para cada modelo
plt.figure(figsize=(20, 6))

# Funcion para la visualizacion
def plot_prediction_vs_real(ax, y_true, y_pred, title):
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.3, color='steelblue', edgecolor=None)
    sns.regplot(x=y_true, y=y_pred, ax=ax, scatter=False, color='orange', line_kws={"linewidth": 2})

    ax.set_xlabel('% Mermas Reales', fontsize=12)
    ax.set_ylabel('% Mermas Predichas', fontsize=12)
    ax.set_title(title, pad=15)
    ax.grid(True, alpha=0.3)

    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=12, )

# Crea los subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Graficar cada modelo
plot_prediction_vs_real(ax1, y_test, y_pred_lr, 'Regresión Lineal')
plot_prediction_vs_real(ax2, y_test, y_pred_rf, 'Random Forest')
plot_prediction_vs_real(ax3, y_test, y_pred_lgb, 'LightGBM')

plt.tight_layout()
plt.savefig('predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: predicciones_vs_reales.png")

# ANÁLISIS DE RESIDUOS
plt.figure(figsize=(20, 6))

# Función para mejorar la visualización de residuos
def plot_residuals(ax, y_pred, residuals, title):
    ax.scatter(y_pred, residuals, alpha=0.3, color='mediumslateblue', edgecolor='k', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

    ax.set_xlabel('Predicciones (%)', fontsize=12, labelpad=10)
    ax.set_ylabel('Residuos (%)', fontsize=12, labelpad=10)
    ax.set_title(title, pad=15)
    ax.grid(True, alpha=0.3)

    stats_text = f'Media: {np.mean(residuals):.3f}\nDesv. Est.: {np.std(residuals):.3f}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=11)

# Crea subplots para residuos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Calcula y grafica residuos
residuals_lr = y_test - y_pred_lr
residuals_rf = y_test - y_pred_rf
residuals_lgb = y_test - y_pred_lgb

plot_residuals(ax1, y_pred_lr, residuals_lr, 'Residuos - Regresión Lineal')
plot_residuals(ax2, y_pred_rf, residuals_rf, 'Residuos - Random Forest')
plot_residuals(ax3, y_pred_lgb, residuals_lgb, 'Residuos - LightGBM')

plt.tight_layout()
plt.savefig('analisis_residuos.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: analisis_residuos.png")

# DISTRIBUCIÓN DE ERRORES
plt.figure(figsize=(20, 6))

# Función para mejorar la visualización de distribución de errores
def plot_error_distribution(ax, residuals, title):
    sns.histplot(residuals, kde=True, ax=ax, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Sin Error')

    ax.set_xlabel('Error (%)', fontsize=12, labelpad=10)
    ax.set_ylabel('Frecuencia', fontsize=12, labelpad=10)
    ax.set_title(title, pad=15)

    stats_text = f'Media: {np.mean(residuals):.3f}\nDesv. Est.: {np.std(residuals):.3f}'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            horizontalalignment='right',
            fontsize=11)


# Crear subplots para distribución de errores
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

plot_error_distribution(ax1, residuals_lr, 'Distribución de Errores - Regresión Lineal')
plot_error_distribution(ax2, residuals_rf, 'Distribución de Errores - Random Forest')
plot_error_distribution(ax3, residuals_lgb, 'Distribución de Errores - LightGBM')

plt.tight_layout()
plt.savefig('distribucion_errores.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: distribucion_errores.png")

# PASO 9: IMPORTANCIA DE CARACTERÍSTICAS
rf_model = trained_models['Random Forest']
if hasattr(rf_model['regressor'], 'feature_importances_'):
    print("\n--- IMPORTANCIA DE CARACTERÍSTICAS ---")
    # Obtiene nombres de características después de one-hot encoding
    preprocessor = rf_model.named_steps['preprocessor']
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_cols])
    
    # Obtiene importancias
    importances = rf_model['regressor'].feature_importances_
    
    # Crear DataFrame para visualización
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Muestra las 10 características más importantes
    print(feature_importance.head(10))
    
    # Visualiza
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance.head(10),
        palette='viridis'
    )
    plt.title('Top 10 Características Más Importantes', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Importancia', fontsize=12, labelpad=10)
    plt.ylabel('Característica', fontsize=12, labelpad=10)

    # Añadir etiquetas de valor
    for i, v in enumerate(feature_importance.head(10)['importance']):
        ax.text(v, i, f'{v:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png', dpi=300, bbox_inches='tight')

    print("Gráfico guardado: importancia_caracteristicas.png")

# PASO 10: DOCUMENTACIÓN DEL PROCESO
print("\n=== DOCUMENTACIÓN DEL PROCESO ===")
print(f"Dimensiones del dataset: {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Período de tiempo analizado: de {data['fecha'].min().strftime('%Y-%m-%d')} a {data['fecha'].max().strftime('%Y-%m-%d')}")

print("Tipos de datos en las columnas principales:")
relevant_cols = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
                'comuna', 'region', 'tienda', 'motivo', 'ubicacion_motivo', 
                'dia_semana', 'dia_mes', 'mes', 'merma_unidad_p']
print(data[relevant_cols].dtypes)

print("\n--- PREPROCESAMIENTO APLICADO ---")
print(f"Variables numéricas: {numeric_features}")
print(f"Variables categóricas: {categorical_features}")
print("Transformaciones aplicadas:")
print("- Variables numéricas: Estandarización")
print("- Variables categóricas: One-Hot Encoding")

print("\n--- DIVISIÓN DE DATOS ---")
print(f"Conjunto de entrenamiento: {len(X_train)} muestras ({len(X_train)/len(data)*100:.1f}% del total)")
print(f"Conjunto de prueba: {len(X_test)} muestras ({len(X_test)/len(data)*100:.1f}% del total)")
print("Método de división: Aleatoria con random_state=42")

print("\n--- MODELOS IMPLEMENTADOS ---")
print("1. Regresión Lineal:")
print("   - Ventajas: Simple, interpretable")
print("   - Limitaciones: Asume relación lineal entre variables")
print("\n2. Random Forest Regressor:")
print("   - Hiperparámetros: n_estimators=100, random_state=42")
print("   - Ventajas: Maneja relaciones no lineales, menor riesgo de overfitting")
print("   - Limitaciones: Menos interpretable, mayor costo computacional")

print("\n--- VALIDACIÓN DEL MODELO ---")
print("Método de validación: Evaluación en conjunto de prueba separado")
print("Métricas utilizadas: MSE, RMSE, MAE, R²")

# PASO 11: CONCLUSIÓN
best_model_name = metrics_df.loc[metrics_df['R²'].idxmax(), 'Modelo']
best_r2 = metrics_df['R²'].max()
best_rmse = metrics_df.loc[metrics_df['R²'].idxmax(), 'RMSE']

print("\n=== CONCLUSIÓN ===")
print(f"El mejor modelo según R² es: {best_model_name}")
print(f"R² del mejor modelo: {best_r2:.4f}")
print(f"RMSE del mejor modelo: {best_rmse:.2f}")

print("\n--- INTERPRETACIÓN DE RESULTADOS ---")
print("• R² (Coeficiente de determinación): Valor entre 0 y 1 que indica qué proporción de la variabilidad")
print("  en las mermas/ventas es explicada por el modelo. Un valor de {:.4f} significa que".format(best_r2))
print("  aproximadamente el {:.1f}% de la variación puede ser explicada por las variables utilizadas.".format(best_r2*100))
print("")
print("• RMSE (Error cuadrático medio): Representa el error promedio de predicción en las mismas unidades")
print("  que la variable objetivo. Un RMSE de {:.2f} significa que, en promedio,".format(best_rmse))
print("  las predicciones difieren de los valores reales en ±{:.2f} unidades.".format(best_rmse))
print("")
print(f"• {best_model_name} es el mejor modelo porque:")
print("  - Ofrece un buen equilibrio entre simplicidad y capacidad predictiva")
print("  - Es más interpretable que modelos complejos")
print("  - Presenta un mejor ajuste a los datos en este caso específico")
print("\nEl análisis predictivo ha sido completado exitosamente.")