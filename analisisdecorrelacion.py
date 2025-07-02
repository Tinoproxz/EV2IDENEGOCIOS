import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos desde CSV
# Cambiar la ruta por tu archivo CSV
df = pd.read_csv('merma.csv', encoding='utf-8')

# Opcional: Mostrar las primeras filas para verificar la carga
print("Primeras filas del dataset:")
print(df.head())
print("\nColumnas disponibles:", df.columns.tolist())
print("Forma del dataset:", df.shape)

# 2. Agrupar productos con merma por tienda y fecha
basket = df.groupby(['fecha', 'tienda'])['descripcion'].apply(list).reset_index(name='items')

print(f"\nNúmero de transacciones (combinaciones fecha-tienda): {len(basket)}")

# 3. Crear lista de transacciones
transactions = basket['items'].tolist()

# Mostrar ejemplo de transacciones
print("\nEjemplo de transacciones:")
for i, trans in enumerate(transactions[:3]):
    print(f"Transacción {i+1}: {trans}")

# 4. Codificar en formato booleano
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

print(f"\nProductos únicos encontrados: {len(te.columns_)}")
print("Primeros productos:", list(te.columns_)[:10])

# 5. Encontrar combinaciones frecuentes (mínimo soporte 0.1)
print("\nBuscando patrones frecuentes...")
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

if len(frequent_itemsets) == 0:
    print("No se encontraron patrones frecuentes con soporte >= 0.1")
    print("Intentando con soporte más bajo (0.05)...")
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

print(f"Patrones frecuentes encontrados: {len(frequent_itemsets)}")

# 6. Generar reglas de asociación (confianza mínima 0.5)
if len(frequent_itemsets) > 0:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    if len(rules) == 0:
        print("No se encontraron reglas con confianza >= 0.5")
        print("Intentando con confianza más baja (0.3)...")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    
    print(f"Reglas de asociación encontradas: {len(rules)}")
    
    # 7. Filtrar por soporte ≥ 0.4 y lift > 1.5 para encontrar más correlación
    filtered_rules = rules[(rules['support'] >= 0.4) & (rules['lift'] > 1.0)]
    
    # Si no hay reglas con esos criterios, relajar los filtros
    if len(filtered_rules) == 0:
        print("No se encontraron reglas con soporte >= 0.4 y lift > 1.0")
        print("Mostrando reglas con lift > 1.0...")
        filtered_rules = rules[rules['lift'] > 1.0]
        
        if len(filtered_rules) == 0:
            print("Mostrando todas las reglas encontradas...")
            filtered_rules = rules
    
    # 8. Mostrar las reglas ordenadas por lift (más correlacionadas arriba)
    filtered_rules = filtered_rules.sort_values(by='lift', ascending=False)
    print(f"\nReglas con mayor correlacion (Top {min(15, len(filtered_rules))}):")
    
    # Mostrar las reglas de forma más legible con nombres truncados
    for idx, rule in filtered_rules.head(15).iterrows():
        # Truncar nombres largos para mejor legibilidad
        antecedents = [item[:30] + '...' if len(item) > 30 else item for item in list(rule['antecedents'])]
        consequents = [item[:30] + '...' if len(item) > 30 else item for item in list(rule['consequents'])]
        
        antecedents_str = ', '.join(antecedents)
        consequents_str = ', '.join(consequents)
        
        print(f"\nRegla: {antecedents_str} => {consequents_str}")
        print(f"  Soporte: {rule['support']:.3f} | Confianza: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}")
    
    # 9. Gráfico: Confianza vs Lift para esas reglas más correlacionadas
    if len(filtered_rules) > 0:
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(
            data=filtered_rules,
            x='confidence',
            y='lift',
            size='support',
            hue='support',
            palette='viridis',
            sizes=(100, 500),
            alpha=0.7
        )
        
        plt.title('Reglas de Asociación - Análisis de Mermas\n(Confianza vs Lift)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Confianza', fontsize=12)
        plt.ylabel('Lift', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Añadir línea horizontal en lift = 1 para mostrar umbral de independencia
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, 
                   label='Lift = 1 (Independencia)')
        
        plt.legend(title='Soporte', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('reglas_asociacion_confianza_vs_lift.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico adicional: Distribución de soporte
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_rules['support'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribución del Soporte en las Reglas de Asociación', fontsize=14)
        plt.xlabel('Soporte', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('distribucion_soporte_reglas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Mostrar resumen estadístico
        print("\nResumen estadístico de las reglas:")
        print(filtered_rules[['support', 'confidence', 'lift']].describe())
        
    else:
        print("No hay reglas para graficar.")
        
else:
    print("No se pudieron generar reglas de asociación.")

# Información adicional sobre el dataset
print(f"\nResumen del análisis:")
print(f"- Total de transacciones analizadas: {len(transactions)}")
print(f"- Productos únicos: {len(te.columns_) if 'te' in locals() else 'N/A'}")
print(f"- Patrones frecuentes: {len(frequent_itemsets) if 'frequent_itemsets' in locals() else 'N/A'}")
print(f"- Reglas de asociación: {len(filtered_rules) if 'filtered_rules' in locals() else 'N/A'}")