import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, normaltest, ttest_ind, t, ttest_1samp, pearsonr, spearmanr
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
file_path = 'C:/Users/Gabriel/Desktop/Proyectos/Estadistica/gym_members_exercise_tracking.csv'
data = pd.read_csv(file_path)

print("Columnas disponibles en el dataset:")
print(data.columns)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# 1. Contingency Table
contingency_table = pd.crosstab(data['Gender'], data['Workout_Type'])
print("\nContingency Table (Gender vs Workout_Type):")
print(contingency_table)

# 2. Correlation Matrix (only numeric columns)
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Test of Independence (Chi-Square Test)
chi2, p, dof, ex = chi2_contingency(contingency_table)
print("\nChi-Square Test of Independence:")
print(f"Chi2 Statistic: {chi2}")
print(f"p-value: {p}")

# 4. Goodness of Fit Test (Normality Test)
normality_test = normaltest(data['Calories_Burned'])
print("\nGoodness of Fit (Normality Test for Calories_Burned):")
print(f"Statistic: {normality_test.statistic}")
print(f"p-value: {normality_test.pvalue}")

# 5. Hypothesis Testing (t-test for Avg_BPM based on Gender)
male_bpm = data[data['Gender'] == 'Male']['Avg_BPM']
female_bpm = data[data['Gender'] == 'Female']['Avg_BPM']
t_stat, t_pvalue = ttest_ind(male_bpm, female_bpm, nan_policy='omit')
print("\nHypothesis Testing (t-test for Avg_BPM by Gender):")
print(f"t-statistic: {t_stat}")
print(f"p-value: {t_pvalue}")

# Additional Visualization: Distribution of Calories Burned
plt.figure(figsize=(10, 6))
sns.histplot(data['Calories_Burned'], kde=True, bins=30, color='blue')
plt.title("Distribution of Calories Burned")
plt.xlabel("Calories Burned")
plt.ylabel("Frequency")
plt.show()

# Additional Visualization: Boxplot of Avg_BPM by Workout Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Workout_Type', y='Avg_BPM', data=data)
plt.title("Boxplot of Avg_BPM by Workout Type")
plt.xlabel("Workout Type")
plt.ylabel("Average BPM")
plt.show()



#Estimacion de Parametros
def confidence_interval(data_column, confidence=0.95):
    """
    Calcula el intervalo de confianza para la media de una columna numérica.
    Retorna la media, el límite inferior y el límite superior.
    """
    n = len(data_column.dropna())  # Número de observaciones sin valores nulos
    mean = data_column.mean()  # Media muestral
    std_err = data_column.std(ddof=1) / (n ** 0.5)  # Error estándar de la media
    t_critical = t.ppf((1 + confidence) / 2, df=n-1)  # Valor crítico de t
    margin_of_error = t_critical * std_err  # Margen de error

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return mean, lower_bound, upper_bound

# Seleccionar variables para análisis
columns_to_analyze = ['Calories_Burned', 'Avg_BPM', 'Workout_Frequency (days/week)']

# Almacenar resultados
intervals = {}

print("\nEstimación de Parámetros (Intervalos de Confianza):")
for column in columns_to_analyze:
    mean, lower, upper = confidence_interval(data[column])
    print(f"{column}: Media = {mean:.2f}, Intervalo de Confianza ({lower:.2f}, {upper:.2f})")
    intervals[column] = (mean, lower, upper)

# Extraer datos para graficar
variables = list(intervals.keys())
means = [intervals[var][0] for var in variables]
lowers = [intervals[var][1] for var in variables]
uppers = [intervals[var][2] for var in variables]
errors = [(means[i] - lowers[i]) for i in range(len(means))]  # Margen de error para la gráfica

# Crear la gráfica de intervalos de confianza
plt.figure(figsize=(10, 6))
plt.errorbar(variables, means, yerr=errors, fmt='o', capsize=5, capthick=2, color='blue', label='IC 95%')
plt.xlabel('Variables')
plt.ylabel('Valores')
plt.title('Intervalos de Confianza para la Media')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()



# Definir pares de variables a analizar
correlation_pairs = [
    ('Calories_Burned', 'Avg_BPM'),
    ('Workout_Frequency (days/week)', 'Calories_Burned')
]

print("\nCoeficiente de Correlación de Pearson:")

# Calcular y mostrar la correlación
for var1, var2 in correlation_pairs:
    if var1 in data.columns and var2 in data.columns:
        if pd.api.types.is_numeric_dtype(data[var1]) and pd.api.types.is_numeric_dtype(data[var2]):
            corr, p_value = pearsonr(data[var1].dropna(), data[var2].dropna())
            print(f"{var1} vs {var2}: r = {corr:.4f}, p-value = {p_value:.4f}")
            if p_value < 0.05:
                print(f"✅ La correlación es estadísticamente significativa.")
            else:
                print(f"❌ No hay suficiente evidencia para afirmar que la correlación es significativa.")
        else:
            print(f"⚠️ Advertencia: {var1} o {var2} no son numéricas.")
    else:
        print(f"⚠️ Advertencia: {var1} o {var2} no existen en el dataset.")

for var1, var2 in correlation_pairs:
    if var1 in data.columns and var2 in data.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=data[var1], y=data[var2])
        plt.title(f"Relación entre {var1} y {var2}")
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.grid(True)
        plt.show()



print("\nCoeficiente de Correlación de Spearman:")

for var1, var2 in correlation_pairs:
    if var1 in data.columns and var2 in data.columns:
        corr, p_value = spearmanr(data[var1].dropna(), data[var2].dropna())
        print(f"{var1} vs {var2}: ρ = {corr:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print(f"✅ La correlación es estadísticamente significativa.")
        else:
            print(f"❌ No hay suficiente evidencia para afirmar que la correlación es significativa.")



# Definir variables
X = data['Avg_BPM']
y = data['Calories_Burned']

# Agregar una constante para el término b0 (intercepto)
X = sm.add_constant(X)

# Ajustar el modelo de regresión
modelo = sm.OLS(y, X).fit()

# Mostrar resultados
print(modelo.summary())

# Obtener predicciones del modelo
data['Predicted_Calories'] = modelo.predict(X)

# Graficar los datos originales y la línea de regresión
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Avg_BPM'], y=data['Calories_Burned'], label='Datos Reales')
sns.lineplot(x=data['Avg_BPM'], y=data['Predicted_Calories'], color='red', label='Regresión Lineal')
plt.xlabel('Avg BPM')
plt.ylabel('Calories Burned')
plt.title('Regresión Lineal: Calories Burned vs. Avg BPM')
plt.legend()
plt.grid(True)
plt.show()



# Calcular residuos
data['Residuos'] = modelo.resid

# Gráfico de Residuos vs. Valores Ajustados
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Predicted_Calories'], y=data['Residuos'])
plt.axhline(0, color='red', linestyle='dashed')
plt.xlabel('Calories Burned (Predichas)')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos vs. Valores Ajustados')
plt.grid(True)
plt.show()



# Definir variables predictoras (X) y la variable dependiente (y)
predictors = ['Avg_BPM', 'Workout_Frequency (days/week)', 'Session_Duration (hours)', 'Weight (kg)', 'Experience_Level']
X = data[predictors]
y = data['Calories_Burned']

# Agregar constante para el término b0
X = sm.add_constant(X)

# Ajustar el modelo de regresión múltiple
modelo_multiple = sm.OLS(y, X).fit()

# Mostrar los resultados del modelo
print(modelo_multiple.summary())

# Obtener predicciones del modelo
data['Predicted_Calories_Multiple'] = modelo_multiple.predict(X)

# Gráfico de comparación entre valores reales y predichos
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Calories_Burned'], y=data['Predicted_Calories_Multiple'])
plt.xlabel('Calories Burned (Reales)')
plt.ylabel('Calories Burned (Predichos)')
plt.title('Comparación entre Valores Reales y Predichos')
plt.grid(True)
plt.show()

# Calcular residuos
data['Residuos_Multiple'] = modelo_multiple.resid

# Gráfico de Residuos vs. Valores Ajustados
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Predicted_Calories_Multiple'], y=data['Residuos_Multiple'])
plt.axhline(0, color='red', linestyle='dashed')
plt.xlabel('Calories Burned (Predichos)')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos vs. Valores Ajustados')
plt.grid(True)
plt.show()



# ANOVA para Calories_Burned según Workout_Type
anova_workout = stats.f_oneway(
    data[data['Workout_Type'] == 'Cardio']['Calories_Burned'],
    data[data['Workout_Type'] == 'HIIT']['Calories_Burned'],
    data[data['Workout_Type'] == 'Strength']['Calories_Burned'],
    data[data['Workout_Type'] == 'Yoga']['Calories_Burned']
)

print("\nANOVA - Calories Burned vs. Workout Type")
print(f"F-Statistic: {anova_workout.statistic:.4f}")
print(f"P-Value: {anova_workout.pvalue:.4f}")

# ANOVA para Calories_Burned según Experience_Level
anova_experience = stats.f_oneway(
    data[data['Experience_Level'] == 1]['Calories_Burned'],
    data[data['Experience_Level'] == 2]['Calories_Burned'],
    data[data['Experience_Level'] == 3]['Calories_Burned']
)

print("\nANOVA - Calories Burned vs. Experience Level")
print(f"F-Statistic: {anova_experience.statistic:.4f}")
print(f"P-Value: {anova_experience.pvalue:.4f}")


# Boxplot para Calories_Burned vs. Workout_Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='Workout_Type', y='Calories_Burned', data=data)
plt.title('Distribución de Calories Burned según Workout Type')
plt.xlabel('Workout Type')
plt.ylabel('Calories Burned')
plt.grid(True)
plt.savefig("Figure_ANOVA_Workout.png", dpi=300)
plt.show()

# Boxplot para Calories_Burned vs. Experience_Level
plt.figure(figsize=(8, 5))
sns.boxplot(x='Experience_Level', y='Calories_Burned', data=data)
plt.title('Distribución de Calories Burned según Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Calories Burned')
plt.grid(True)
plt.savefig("Figure_ANOVA_Experience.png", dpi=300)
plt.show()



# Seleccionar solo las columnas numéricas
numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                      'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
                      'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
                      'Experience_Level', 'BMI']

# Escalar los datos (PCA es sensible a la escala)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_features])

# Aplicar PCA
pca = PCA(n_components=len(numerical_features))  # Mantener todas las componentes para análisis inicial
principal_components = pca.fit_transform(scaled_data)

# Varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_

# Mostrar resultados
print("\nVarianza explicada por cada componente:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")

# Calcular la varianza acumulada
cumulative_variance = np.cumsum(explained_variance)
print("\nVarianza acumulada:")
for i, var in enumerate(cumulative_variance):
    print(f"PC{i+1}: {var:.4f}")

# Selección del número óptimo de componentes (cuando la varianza acumulada alcanza ~95%)
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNúmero óptimo de componentes: {optimal_components}")

# Gráfico de Varianza Explicada
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(numerical_features) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianza Explicada')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada')
plt.title('Gráfico de Varianza Explicada por PCA')
plt.legend()
plt.grid(True)
plt.savefig("Figure_PCA_Variance.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Visualización de los Datos en 2 Componentes Principales')
plt.grid(True)
plt.savefig("Figure_PCA_2D.png", dpi=300)
plt.show()



# Seleccionar las variables para el clustering
clustering_features = ['Avg_BPM', 'Calories_Burned', 'Workout_Frequency (days/week)',
                       'Session_Duration (hours)', 'Weight (kg)', 'Experience_Level']

# Escalar los datos para normalizar las variables
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[clustering_features])

# Determinar el número óptimo de clusters usando el método del codo
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Selección de k')
plt.grid(True)
plt.savefig("Figure_Elbow_Method.png", dpi=300)
plt.show()

# Aplicar K-Means con el número óptimo de clusters
optimal_k = 3  # Ajusta esto según el método del codo
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizar los centroides
print("Centroides de los clusters:")
print(kmeans.cluster_centers_)

# Contar cuántos individuos hay en cada cluster
print("\nDistribución de Clusters:")
print(data['Cluster'].value_counts())

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
data['PC1'] = pca_components[:, 0]
data['PC2'] = pca_components[:, 1]

# Graficar los clusters en el espacio PCA
plt.figure(figsize=(8, 5))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=data, palette='viridis')
plt.title('Clustering en 2D usando PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig("Figure_Clusters_PCA.png", dpi=300)
plt.show()

print("Analysis complete.")
