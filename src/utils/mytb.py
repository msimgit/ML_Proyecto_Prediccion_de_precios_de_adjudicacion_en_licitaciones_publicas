from itertools import combinations
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from scipy.stats import shapiro
from scipy.stats import boxcox
from scipy import stats

#----------------------------------------------------------------------------------------------------------------
# Función data_report(dataf)
#----------------------------------------------------------------------------------------------------------------

def data_report(dataf):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(dataf.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(dataf.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(dataf.isnull().sum() * 100 / len(dataf), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(dataf.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(dataf), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)

    return concatenado.T


#----------------------------------------------------------------------------------------------------------------
# Función analisis_univariante_ultrarrapidoanalisis_univariante_ultrarrapido(dataf, vars_cat, vars_num, bins=12, sample_size=100_000, max_barras=10)
#----------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

def analisis_univariante_ultrarrapido(dataf, vars_cat, vars_num, bins=12, sample_size=100_000, max_barras=10):
    """
    Análisis univariante ultrarrápido:
    - Barras para categóricas (máx 20 categorías)
    - Histogramas para numéricas
    - Todas las etiquetas del eje X truncadas a 5 caracteres
    """
    total_vars = vars_cat + vars_num
    n = len(total_vars)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    for i, var in enumerate(total_vars):
        ax = axes[i]
        serie = dataf[var].dropna()

        if var in vars_cat:
            freqs = serie.value_counts().head(max_barras)
            labels = [str(label)[:5] for label in freqs.index]
            ax.bar(labels, freqs.values)

        elif var in vars_num:
            if len(serie) > sample_size:
                serie = serie.sample(sample_size, random_state=42)
            ax.hist(serie, bins=bins, edgecolor='black')

            # Recortar etiquetas numéricas del eje X
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([str(label)[:5] for label in ax.get_xticks()], rotation=45, fontsize=6)

        ax.set_title(var[:30], fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=6)

    # Ocultamos parte de los ejes para mejorar la visualizacion
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


#----------------------------------------------------------------------------------------------------------------
# Función analisis_bivariante_ultrarrapido(dataf, vars_cat, vars_num, target, bins=30, sample_size=100_000, max_barras=10)
#----------------------------------------------------------------------------------------------------------------

def analisis_bivariante_ultrarrapido(dataf, vars_cat, vars_num, target, bins=30, sample_size=100_000, max_barras=10):
    """
    Análisis bivariante ultrarrápido:
    - Para variables categóricas: boxplots del target por categoría (máx 10 categorías)
    - Para variables numéricas: scatterplot contra target
    """
    total_vars = vars_cat + vars_num
    n = len(total_vars)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    for i, var in enumerate(total_vars):
        ax = axes[i]
        df = dataf[[var, target]].dropna()

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

        if var in vars_cat:
            # Limita a las categorías más frecuentes
            top_cats = df[var].value_counts().index[:max_barras]
            df = df[df[var].isin(top_cats)]
            sns.boxplot(x=var, y=target, data=df, ax=ax)
            ax.set_xticklabels([str(label.get_text())[:5] for label in ax.get_xticklabels()], rotation=45, fontsize=6)

        elif var in vars_num:
            ax.scatter(df[var], df[target], alpha=0.3, s=5)
            ax.set_xlabel(var[:10], fontsize=6)
            ax.set_ylabel(target[:10], fontsize=6)

            # Recortar etiquetas numéricas del eje X
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([str(label)[:5] for label in ax.get_xticks()], rotation=45, fontsize=6)

        ax.set_title(var[:30], fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=6)

    # Ocultamos parte de los ejes para mejorar la visualizacion
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------------------------------------------
# Función analizar_transformaciones(dataf, vars_continuas, sample_size=10000)
#----------------------------------------------------------------------------------------------------------------

def analizar_transformaciones(dataf, vars_continuas, sample_size=10000):
    """
    Evalúa distribución original, logarítmica y Box-Cox de variables numéricas continuas.
    Muestra histogramas y p-values del test de Shapiro para cada transformación.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from numpy import log1p
    from scipy.stats import boxcox, shapiro

    for var in vars_continuas:
        print(f"\nVariable: {var}")

        serie = dataf[var].dropna()

        # Muestreamos si la serie es muy grande
        if len(serie) > sample_size:
            serie = serie.sample(sample_size, random_state=42)

        # Evitamos ceros o negativos para log y boxcox
        if (serie <= 0).any():
            print(f"La variable '{var}' contiene valores <= 0. No se puede aplicar log ni Box-Cox.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        # Original
        sns.histplot(serie, kde=False, ax=axes[0])
        pval_original = shapiro(serie).pvalue
        axes[0].set_title(f"Original\nShapiro p={pval_original:.4f}")
        axes[0].set_xlabel(var)

        # Log(x+1)
        log1p_data = log1p(serie)
        sns.histplot(log1p_data, kde=False, ax=axes[1])
        pval_log1p = shapiro(log1p_data).pvalue
        axes[1].set_title(f"Log(x+1)\nShapiro p={pval_log1p:.4f}")
        axes[1].set_xlabel(var)

        # Box-Cox
        boxcox_data, _ = boxcox(serie)
        sns.histplot(boxcox_data, kde=False, ax=axes[2])
        pval_boxcox = shapiro(boxcox_data).pvalue
        axes[2].set_title(f"Box-Cox\nShapiro p={pval_boxcox:.4f}")
        axes[2].set_xlabel(var)

        plt.suptitle(f"Transformaciones para '{var}'", fontsize=12)
        plt.tight_layout()
        plt.show()


#----------------------------------------------------------------------------------------------------------------
# Función auto_encoding(df_enc, n_features_hash=6)
#----------------------------------------------------------------------------------------------------------------

from sklearn.feature_extraction import FeatureHasher
import pandas as pd

def auto_encoding(df_enc, n_features_hash=6):
    """
    Codifica automáticamente las variables categóricas del DataFrame según su cardinalidad.

    - Usa One-Hot Encoding para baja cardinalidad.
    - Usa Hashing Encoding (FeatureHasher) para alta cardinalidad.

    Parámetros:
    ------------
    - df_enc : pd.DataFrame
        DataFrame con las variables a codificar.
    - n_features_hash : int
        Número de dimensiones para el hasher (más = menor colisión, más columnas).

    Retorna:
    --------
    - df_enc : pd.DataFrame
        DataFrame codificado.
    """
    df_encoded = df_enc.copy()
    df_freq, lista_cat, lista_num = card_tipo(df_enc)

    for feature in df_encoded.columns:
        print(f"Procesando: {feature}")

        if df_encoded[feature].dtype == object:
            tipo = df_freq.loc[feature, 'tipo_sugerido'].lower()

            if 'numerica' in tipo:  # Alta cardinalidad ➜ Hashing
                print(f"  Hashing Encoding (alta cardinalidad) para '{feature}'")

                hasher = FeatureHasher(n_features=n_features_hash, input_type='string')
                
                # Corrección clave aquí
                hashed = hasher.transform(df_encoded[feature].astype(str).apply(lambda x: [x])).toarray()
                
                hashed_df = pd.DataFrame(
                    hashed,
                    columns=[f"{feature}_hash_{i}" for i in range(n_features_hash)],
                    index=df_encoded.index
                )

                df_encoded = pd.concat([df_encoded.drop(columns=[feature]), hashed_df], axis=1)


            elif 'categorica' in tipo:  # Baja cardinalidad ➜ One-Hot
                print(f"  One-Hot Encoding para '{feature}'")
                dummies = pd.get_dummies(df_encoded[feature], prefix=feature)
                df_encoded = pd.concat([df_encoded.drop(columns=[feature]), dummies], axis=1)

    return df_encoded



#----------------------------------------------------------------------------------------------------------------
# Función controlar_tipos(dataf)
#----------------------------------------------------------------------------------------------------------------

def controlar_tipos(dataf):
    """
    Forzamos los tipos de cada una de las features de nuestro dataframe
    Hemos decidido hacerlo con una función para favorecer la limpieza del Jupyter Notebbok y no alargar la redaccion

    Parámetros:
    ------------
    - dataf : pd.DataFrame
        DataFrame con las variables a codificar.

    Retorna:
    --------
    - dataf : pd.DataFrame
        DataFrame con los tipos modificados
    """

    new_types = {
        int: ['Número de expediente', 'CPV', 'CPV licitación/lote', 'Tipo de contrato',
            'Número de ofertas recibidas por licitación/lote', 'Forma de presentación de la oferta',
            'Resultado licitación/lote', 'Número de contrato'],
        float: ['Valor estimado del contrato', 'Presupuesto base con impuestos/lote',
                'Presupuesto base sin impuestos/lote', 'Valor estimado licitación/lote',
                'Presupuesto base con impuestos', 'Presupuesto base sin impuestos', 
                'Importe adjudicación con impuestos', 'Importe adjudicación sin impuestos',
                'Precio de la oferta más baja', 'Precio de la oferta más alta',],
        str: ['Link licitación', 'Objeto del Contrato', 'Objeto de licitación/lote',
            'Vigente/Anulada/Archivada', 'Contrato mixto',
            'Lugar de ejecución', 'Lugar de ejecución licitación/lote', 'Órgano de contratación', 
            'NIF OC', 'ID OC en PLACSP', 'Enlace al Perfil de Contratante del OC',
            'Tipo de Procedimiento', 'Sistema de contratación', 'Tramitación',
            'Adjudicatario', 'Identificador adjudicatario', 'Tipo de identificador de adjudicatario'],
        bool: ['Subasta electrónica', 'Se han excluido ofertas anormalmente bajas',
            'Contrato mixto', 'Vigente/Anulada/Archivada',
            ],
        'datetime64[ns]': ['Fecha de actualización', 'Fecha formalización del contrato', 'Fecha del acuerdo licitación/lote'],    
    }

    for key, value in new_types.items():
        for i in value:
            dataf[i] = dataf[i].astype(key, errors='ignore')
    return dataf






#----------------------------------------------------------------------------------------------------------------
# Función card_tipo(df)
#----------------------------------------------------------------------------------------------------------------

def card_tipo(df):
    """
    Análisis cardinalidad de de las variables de un DataFrame de pandas.

    Autor: MS
    Fecha: Julio 2025

    Descripción:
    -------------
    Esta función realiza un análisis de la frecuencia de cada una de las variables
    proponiendo un tipo de categorización; categorica, numercica discreta o continua.

    Para cada variable:
    - Numéricas: se calculan estadísticas descriptivas y se generan histogramas, boxplots y violin plots.
    - Categóricas: se muestran frecuencias absolutas y relativas, y se visualizan con gráficos de barras y pastel.

    Parámetros:
    ------------
    - df : pd.DataFrame
        El DataFrame a analizar.

    Requisitos:
    ------------
    - N/A
    """

    n_rows = len(df)

    # Umbrales dinámicos
    if n_rows <= 500:
        umbral_categoria = 10
        umbral_continua = 50
    elif n_rows <= 5000:
        umbral_categoria = 15
        umbral_continua = 75
    else:
        umbral_categoria = 20
        umbral_continua = 100

    df_temp = pd.DataFrame({
        "Card": df.nunique(),
        "%_Card": df.nunique() / len(df) * 100,
        "Tipo": df.dtypes
    })
    df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00

    df_temp["tipo_sugerido"] = "Categorica"  # Valor por defecto

    # Reglas de clasificación
    df_temp.loc[df_temp["Card"] == 2, "tipo_sugerido"] = "Binaria"
    df_temp.loc[(df_temp["Card"] >= umbral_categoria) & (df_temp["%_Card"] < umbral_continua), "tipo_sugerido"] = "Numerica discreta"
    df_temp.loc[(df_temp["%_Card"] >= umbral_continua) & (df_temp["Tipo"].isin(['int64', 'float64'])), "tipo_sugerido"] = "Numerica continua"

    # Crear listas agrupadas
    list_categoricas = df_temp[df_temp["tipo_sugerido"].isin(["Categorica", "Binaria"])].index.tolist()
    list_numericas = df_temp[df_temp["tipo_sugerido"].isin(["Numerica continua", "Numerica discreta"])].index.tolist()

    return df_temp, list_categoricas, list_numericas


#----------------------------------------------------------------------------------------------------------------
# Función analysis_uni(df, display_units=False)
#----------------------------------------------------------------------------------------------------------------

def analysis_uni(df_uni, display_units=False):
    """
    Análisis univariante completo de variables numéricas y categóricas.
    Guarda todos los gráficos en PDF (solo gráficos) y muestra por pantalla.
    Resultados estadísticos se imprimen por consola.

    Parámetros:
    - df_uni: DataFrame
    - display_units: bool (si True, se muestran etiquetas en gráficos)
    """

    # Crear carpeta de salida si no existe
    os.makedirs("./src/reports", exist_ok=True)
    pdf = PdfPages("./src/reports/Analisis_univariante.pdf")

    df_uni.info()

    # Clasificación de variables
    df_tipo, categoricas, numericas = card_tipo(df_uni)
    print("Lista de variables categóricas:\n", categoricas)
    print("Lista de variables numéricas:\n", numericas)
    linea = '-' * 100
    print(linea)
    print("Propuesta de categorización del dataset:")
    print(linea)
    print(df_tipo)

# Análisis de variables NUMÉRICAS

    print("\nComenzamos nuestro análisis con las variables numéricas")

    # Histogramas + KDE
    fig, axes = plt.subplots(math.ceil(len(numericas)/5), 5, figsize=(20, 4 * math.ceil(len(numericas)/5)))
    for ax, col in zip(axes.flatten(), numericas):
        sns.histplot(df_uni[col], kde=True, bins=50, ax=ax)
        ax.set_title(col)
    for ax in axes.flatten()[len(numericas):]: ax.axis('off')
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

    # Boxplots
    fig, axes = plt.subplots(math.ceil(len(numericas)/5), 5, figsize=(20, 4 * math.ceil(len(numericas)/5)))
    for ax, col in zip(axes.flatten(), numericas):
        sns.boxplot(x=df_uni[col], ax=ax)
        ax.set_title(col)
    for ax in axes.flatten()[len(numericas):]: ax.axis('off')
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

    # Estadísticas numéricas
    for col in numericas:
        data = df_uni[col].dropna()
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.astype('int64') / 1e9

        print(linea)
        print(f"Análisis univariante para la variable numérica: {col}")
        print(linea)

        media = data.mean()
        mediana = data.median()
        moda = data.mode().values
        cuartiles = np.quantile(data, [0.25, 0.5, 0.75])
        percentiles = np.percentile(data, [10, 25, 50, 75, 90])
        varianza = data.var()
        desviacion = data.std()
        rango = data.max() - data.min()
        minimo = data.min()
        maximo = data.max()
        asimetria = data.skew()
        curtosis = data.kurt()

        print("\nEstadísticos de centralidad:")
        print(f"\t{'Media:':<25}{media:.4f}")
        print(f"\t{'Mediana:':<25}{mediana:.4f}")
        print(f"\t{'Moda:':<25}{moda}")
        print(f"\t{'Cuartiles(25,50,75):':<25}{cuartiles}")
        print(f"\t{'Percent.(10,25,50,75,90):':<25}{percentiles}")

        print("\nEstadísticos de dispersión:")
        print(f"\t{'Varianza:':<25}{varianza:.4f}")
        print(f"\t{'Desviación estándar:':<25}{desviacion:.4f}")
        print(f"\t{'Rango:':<25}{rango:.4f}")
        print(f"\t{'Mínimo:':<25}{minimo:.4f}")
        print(f"\t{'Máximo:':<25}{maximo:.4f}")
        print(f"\t{'Asimetría:':<25}{asimetria:.4f}")
        print(f"\t{'Curtosis:':<25}{curtosis:.4f}")


# Análisis de variables CATEGÓRICAS

    print("\nMostramos ahora un resumen visual de las variables categóricas")

    # Impresión por pantalla de frecuencia por variable
    for col in categoricas:
        data = df_uni[col].astype(str).fillna("Desconocido")
        print(linea)
        print(f"Análisis univariante para la variable categórica: {col}")
        print(linea)

        freq_abs = data.value_counts()
        if len(freq_abs) > 10:
            top_9 = freq_abs.nlargest(9)
            otros = freq_abs.iloc[9:].sum()
            freq_abs = pd.concat([top_9, pd.Series({'Otros': otros})])
        freq_rel = freq_abs / freq_abs.sum() * 100

        tabla = pd.DataFrame({
            'Frecuencia absoluta': freq_abs,
            'Frecuencia relativa (%)': freq_rel.round(2)
        })
        print(tabla)

    # Gráfico agrupado final de barras normalizadas
    fig, axes = plt.subplots(math.ceil(len(categoricas)/5), 5, figsize=(20, 4 * math.ceil(len(categoricas)/5)))

    for ax, col in zip(axes.flatten(), categoricas):
        data = df_uni[col].astype(str).fillna("Desconocido")
        freq = data.value_counts(normalize=True)
        if len(freq) > 10:
            top_9 = freq.nlargest(9)
            otros = freq.iloc[9:].sum()
            freq = pd.concat([top_9, pd.Series({'Otros': otros})])
        sns.barplot(x=freq.values, y=freq.index, ax=ax, orient='h')
        ax.set_title(col)
        if display_units:
            for i, (val, label) in enumerate(zip(freq.values, freq.index)):
                ax.text(val, i, f"{label}", va='center', ha='left', fontsize=8)

    for ax in axes.flatten()[len(categoricas):]:
        ax.axis('off')

    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

    # Cierre del PDF
    pdf.close()
    print("\n Todos los gráficos han sido guardados en: ./src/reports/Analisis_univariante.pdf")


#----------------------------------------------------------------------------------------------------------------
# Función analysis_biv((df, target, display_labels=False)
#----------------------------------------------------------------------------------------------------------------

def analysis_biv(df, target, display_labels=False):
    """
    Análisis bivariante con target numérico frente a variables numéricas y categóricas.

    Parámetros:
    ------------
    - df : pd.DataFrame
        DataFrame con las variables a analizar.
    - target : str
        Nombre de la variable objetivo (debe ser numérica).
    - display_labels : bool
        Si True, muestra etiquetas en gráficos categóricos.
    """

    os.makedirs("./src/reports", exist_ok=True)
    pdf = PdfPages("./src/reports/Analisis_bivariante.pdf")

    numericas = df.select_dtypes(include=np.number).columns.drop(target, errors='ignore')
    categoricas = df.select_dtypes(include='object').columns

    print(f"Target numérico: {target}")
    print(f"Variables numéricas:\n{list(numericas)}")
    print(f"Variables categóricas:\n{list(categoricas)}")
    print("-" * 100)

# Análsis features NUMÉRICAS contra target NUMÉRICO

    print("\nAnálisis con variables numéricas:")
    correlaciones = df[numericas].corrwith(df[target]).dropna()
    print("Correlaciones de Pearson:")
    print(correlaciones.sort_values(ascending=False))

    # Mapa de calor
    corr_matrix = df[[target] + list(numericas)].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa de calor de correlaciones")
    pdf.savefig(); plt.show(); plt.close()

    # Scatterplots agrupados
    filas = math.ceil(len(numericas) / 3)
    fig, axes = plt.subplots(filas, 3, figsize=(18, 5 * filas))
    axes = axes.flatten()
    for i, col in enumerate(numericas):
        sns.regplot(data=df, x=col, y=target, ax=axes[i], scatter_kws={'s': 10, 'alpha': 0.3}, line_kws={'color': 'red'})
        axes[i].set_title(f"{target} vs {col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    pdf.savefig(); plt.show(); plt.close()

# Análsis features CATEGÓRICAS contra target NUMÉRICO

    print("\nAnálisis con variables categóricas:")
    for col in categoricas:
        top_cats = df[col].value_counts().nlargest(30).index
        df_cat = df[df[col].isin(top_cats)]
        resumen = df_cat.groupby(col)[target].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False)
        print(f"\nResumen para {col} (top 30):")
        print(resumen)

    # Boxplots agrupados
    filas = math.ceil(len(categoricas) / 3)
    fig, axes = plt.subplots(filas, 3, figsize=(18, 5 * filas))
    axes = axes.flatten()
    for i, col in enumerate(categoricas):
        top_cats = df[col].value_counts().nlargest(30).index
        sns.boxplot(data=df[df[col].isin(top_cats)], x=col, y=target, ax=axes[i])
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_title(f"Boxplot: {target} vs {col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    pdf.savefig(); plt.show(); plt.close()

    # Violin plots agrupados
    fig, axes = plt.subplots(filas, 3, figsize=(18, 5 * filas))
    axes = axes.flatten()
    for i, col in enumerate(categoricas):
        top_cats = df[col].value_counts().nlargest(30).index
        sns.violinplot(data=df[df[col].isin(top_cats)], x=col, y=target, ax=axes[i], inner="quartile")
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_title(f"Violin plot: {target} vs {col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    pdf.savefig(); plt.show(); plt.close()

    # Cierre PDF
    pdf.close()
    print("\n Gráficos guardados en ./src/reports/Analisis_bivariante.pdf")


def visualizar_distribuciones(train_set, target_col):
    resumen = analizar_columnas(train_set)

    # --- Distribución del target
    plt.figure(figsize=(8, 4))
    sns.histplot(train_set[target_col].dropna(), kde=True)
    plt.title(f'Distribución del target: {target_col}')
    plt.show()

    # --- Distribuciones numéricas
    num_cols = resumen[resumen['clasificacion'].isin(['numerica_discreta', 'numerica_continua'])]['columna']
    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(train_set[col].dropna(), kde=True)
        plt.title(f'Distribución de la variable numérica: {col}')
        plt.show()

    # --- Distribuciones categóricas
    cat_cols = resumen[resumen['clasificacion'] == 'categorica']['columna']
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        train_set[col].value_counts(normalize=True).plot(kind='bar')
        plt.title(f'Distribución de la variable categórica: {col}')
        plt.ylabel('Frecuencia relativa')
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return resumen


#----------------------------------------------------------------------------------------------------------------
# visualizar_distribuciones(train_set, target_col)
#----------------------------------------------------------------------------------------------------------------


def visualizar_distribuciones(train_set, target_col):
    # Llamamos a tu función de cardinalidad
    resumen, list_categoricas, list_numericas = card_tipo(train_set.drop(columns=[target_col]))

    # Añadimos el target al resumen manualmente
    target_tipo = 'Numerica continua' if train_set[target_col].nunique() > 20 else 'Numerica discreta'
    resumen.loc[target_col] = {
        'Card': train_set[target_col].nunique(),
        '%_Card': train_set[target_col].nunique() / len(train_set) * 100,
        'Tipo': train_set[target_col].dtype,
        'tipo_sugerido': target_tipo
    }
    
    # --- Distribución del target
    plt.figure(figsize=(8, 4))
    sns.histplot(train_set[target_col].dropna(), kde=True)
    plt.title(f'Distribución del target: {target_col}')
    plt.xlabel(target_col)
    plt.tight_layout()
    plt.show()

    # --- Variables numéricas
    for col in list_numericas:
        plt.figure(figsize=(8, 4))
        sns.histplot(train_set[col].dropna(), kde=True)
        plt.title(f'Distribución numérica: {col}')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    # --- Variables categóricas
    for col in list_categoricas:
        plt.figure(figsize=(10, 4))
        valores = train_set[col].value_counts(normalize=True)[:20]  # Top 20 categorías
        valores.plot(kind='bar')
        plt.title(f'Distribución categórica: {col}')
        plt.ylabel('Frecuencia relativa')
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return resumen



#----------------------------------------------------------------------------------------------------------------
# Función imputar_y_transformar_fechas(dataf)
#----------------------------------------------------------------------------------------------------------------

def imputar_y_transformar_fechas(dataf):
    """
    Imputa fechas faltantes en 'Fecha formalización del contrato' basándose en la diferencia
    con 'Fecha del acuerdo licitación/lote' y anualiza las variables de fecha.

    Parámetros:
    -----------
    dataf : pd.DataFrame
            DataFrame con las columnas necesarias.
    
    features_fechas : list
            Lista de columnas tipo fecha a transformar (e.g. anualizar).

    Retorna:
    --------
    dataf : pd.DataFrame
            DataFrame modificado con fechas imputadas y anualizadas.
    """
    features_fechas = ['Fecha del acuerdo licitación/lote', 'Fecha formalización del contrato']

    # Resolvemos Missings de forma manual
    # 'Fecha del acuerdo licitación/lote'
    # Nos aseguramos que estén en el formato adecuado nuevamente
    features_fechas = [
        'Fecha formalización del contrato',
        'Fecha del acuerdo licitación/lote'
        ]
    for feature in features_fechas:
        dataf[feature] = pd.to_datetime(dataf[feature], errors='coerce') 
    #   Aplicamos la mediana a los faltantes. Tiene más sentido para variables numéricas si utilizamos árboles (modelo de regresión)
    mediana_fecha_lic = dataf['Fecha del acuerdo licitación/lote'].median()
    dataf['Fecha del acuerdo licitación/lote'] = dataf['Fecha del acuerdo licitación/lote'].fillna(mediana_fecha_lic)

    # 1. Filtrar registros válidos con ambas fechas
    df_valid = dataf.dropna(subset=['Fecha formalización del contrato', 'Fecha del acuerdo licitación/lote'])

    # 2. Calcular diferencia en días
    df_valid['diferencia_dias'] = (
        df_valid['Fecha formalización del contrato'] - df_valid['Fecha del acuerdo licitación/lote']
    ).dt.days

    # 3. Mediana global y por 'Órgano de contratación'
    mediana_global = df_valid['diferencia_dias'].median()
    mediana_formalizacion = (
        df_valid.groupby('Órgano de contratación')['diferencia_dias']
        .median()
    )

    # 4. Imputar fechas faltantes
    mask = dataf['Fecha formalización del contrato'].isna() & dataf['Fecha del acuerdo licitación/lote'].notna()
    dias_a_sumar = dataf.loc[mask, 'Órgano de contratación'].map(mediana_formalizacion).fillna(mediana_global)
    dataf.loc[mask, 'Fecha formalización del contrato'] = (
        dataf.loc[mask, 'Fecha del acuerdo licitación/lote'] + pd.to_timedelta(dias_a_sumar, unit='D')
    )

    # 5. Anualizar fechas
    for feature in features_fechas:
        dataf[feature] = dataf[feature].dt.year

    return dataf