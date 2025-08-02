import pandas as pd
import numpy as np
from mytb import controlar_tipos, imputar_y_transformar_fechas

def pipeline_transformaciones_manual(dataf, target='Importe adjudicación sin impuestos'):
    # 1. Filtros básicos
    dataf = dataf.dropna(subset=['Adjudicatario'])
    dataf = dataf.dropna(subset=[target])
    dataf = dataf[dataf['Presupuesto base sin impuestos'].notna() & (dataf['Presupuesto base sin impuestos'] != 0)]

    # 2. Tipos
    dataf = controlar_tipos(dataf)

    # 3. Eliminación de features irrelevantes
    features_a_eliminar = [
        "Tramitación", "Subasta electrónica", "Contrato mixto", "Vigente/Anulada/Archivada",
        "Link licitación", "Presupuesto base con impuestos", "Presupuesto base sin impuestos/lote",
        "Valor estimado licitación/lote", "Valor estimado del contrato", "Presupuesto base con impuestos/lote",
        "Fecha de actualización", "Objeto del Contrato", "Resultado licitación/lote", "NIF OC",
        "ID OC en PLACSP", "Enlace al Perfil de Contratante del OC", "Objeto de licitación/lote",
        "CPV licitación/lote", "Número de contrato", "Importe adjudicación con impuestos",
        "Tipo de identificador de adjudicatario", "Identificador adjudicatario",
        "Precio de la oferta más baja", "Precio de la oferta más alta", "Se han excluido ofertas anormalmente bajas"
    ]
    dataf.drop(columns=[col for col in features_a_eliminar if col in dataf.columns], inplace=True)

    # 4. Correlación con el target
    numeric_df = dataf.select_dtypes(include='number')
    corr = np.abs(numeric_df.corr()[target]).sort_values(ascending=True)

    bad_corr_feat = corr[corr < 0.2].index.difference(['CPV', 'Presupuesto base sin impuestos']).tolist()
    dataf.drop(columns=bad_corr_feat, inplace=True)

    # 5. Duplicados
    dataf = dataf.drop_duplicates(subset='Número de expediente', keep='last')
    dataf.drop(columns=['Número de expediente'], inplace=True, errors='ignore')

    # 6. Fechas
    dataf = imputar_y_transformar_fechas(dataf)

    # 7. Drop columnas con >50% missing
    percent_missing = dataf.isnull().mean() * 100
    cols_to_drop = percent_missing[percent_missing > 50].index
    dataf.drop(columns=cols_to_drop, inplace=True)

    # 8. Outliers extremos y nulos
    umbral_superior = dataf[target].quantile(0.999)
    dataf = dataf[dataf[target] <= umbral_superior]
    dataf = dataf[dataf[target] > 0]

    return dataf.copy()
