import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
df = pd.read_excel('/content/mermas_actividad_unidad_2.xlsx')

# 2. Agrupar productos con merma por tienda y fecha
basket = df.groupby(['fecha', 'tienda'])['descripcion'].apply(list).reset_index(name='items')

# 3. Crear lista de transacciones
transactions = basket['items'].tolist()

# 4. Codificar en formato booleano
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# 5. Encontrar combinaciones frecuentes (mínimo soporte 0.1)
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# 6. Generar reglas de asociación (confianza mínima 0.5)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# 7. Filtrar por soporte ≥ 0.4 y lift > 1.5 para encontrar más correlación
filtered_rules = rules[(rules['support'] >= 0.4) & (rules['lift'] > 1.0)]

# 8. Mostrar las reglas ordenadas por lift (más correlacionadas arriba)
filtered_rules = filtered_rules.sort_values(by='lift', ascending=False)
print("Reglas con mayor correlación (lift alto):")
display(filtered_rules)

# 9. Gráfico: Confianza vs Lift para esas reglas más correlacionadas
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=filtered_rules,
    x='confidence',
    y='lift',
    size='support',
    hue='support',
    palette='viridis',
    sizes=(100, 300)
)
plt.title('Reglas de Asociación con Alta Correlación (lift > 1.0 y soporte ≥ 0.4)')
plt.xlabel('Confianza')
plt.ylabel('Lift')
plt.grid(True)
plt.legend(title='Soporte', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()