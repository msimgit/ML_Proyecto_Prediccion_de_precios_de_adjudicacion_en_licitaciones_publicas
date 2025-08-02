from itertools import combinations
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd
import seaborn as sns
from numpy import log1p
from sklearn.model_selection import KFold
from scipy.stats import boxcox, shapiro
from scipy import stats
from sklearn.preprocessing import StandardScaler


    
    


#----------------------------------------------------------------------------------------------------------------
# Función analizar_transformaciones(dataf, vars_continuas, sample_size=10000)
#----------------------------------------------------------------------------------------------------------------

def analizar_transformaciones(dataf, vars_continuas, sample_size=10000):
    """
    Evalúa distribución original, logarítmica y Box-Cox de variables numéricas continuas.
    Muestra histogramas y p-values del test de Shapiro para cada transformación.
    """


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
# Función transformar_boxcox_escalar(dataf)
#----------------------------------------------------------------------------------------------------------------

def transformar_boxcox_escalar(dataf):
    """
    Aplica Box-Cox y StandardScaler a columnas numéricas positivas.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original
    columnas : list
        Lista de nombres de columnas a transformar

    Retorna:
    --------
    df_transformado : pd.DataFrame
        DataFrame con columnas transformadas (Box-Cox + escalado)
    """
    df_transformado = dataf.copy()
    scaler = StandardScaler()

    num_cols = [
    "Presupuesto base sin impuestos",
    "Importe adjudicación sin impuestos",
    "Descuento_adjudicatario"
    ]

    for col in num_cols:
        serie = df_transformado[col].dropna()

        if (serie <= 0).any():
            print(f"Se omite '{col}' porque contiene valores <= 0 (no apto para Box-Cox).")
            continue

        # Aplicamos Box-Cox
        transformada, _ = boxcox(serie)
        df_transformado.loc[serie.index, col] = transformada

    # Escalamos columnas transformadas
    df_transformado[num_cols] = scaler.fit_transform(df_transformado[num_cols])

    return df_transformado

