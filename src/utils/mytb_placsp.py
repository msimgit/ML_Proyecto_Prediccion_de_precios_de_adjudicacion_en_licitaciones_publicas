import csv
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile


#--------------------------------------------------------------------------------------------------------------------------
# Funcion descomprimir_zips(source_dir, destinataion_dir)
#--------------------------------------------------------------------------------------------------------------------------

def descomprimir_zips(source_dir, destinataion_dir):
    """
    Esta función busca y descomprime todos los archivos .zip ubicados en una carpeta
    de origen (source_dir) dentro de una carpeta de destino (destination_dir).
    
    Parámetros:
       - source_dir (str): Ruta del directorio que contiene los archivos .zip.
       - destination_dir (str): Ruta del directorio donde se extraerán los contenidos.
    
    Comportamiento:
       - Crea la carpeta de destino si no existe.
       - Itera sobre todos los archivos en la carpeta de origen.
       - Extrae el contenido de cada archivo .zip en la carpeta de destino.
    
    Resultado:
       - Los contenidos de todos los .zip quedan descomprimidos y accesibles en destination_dir.
    """
    
    # Crear la carpeta de destino si no existe
    os.makedirs(destination_dir, exist_ok=True)

    # Iterar sobre todos los archivos en la carpeta fuente
    for filename in os.listdir(source_dir):
        if filename.endswith('.zip'):
            zip_path = os.path.join(source_dir, filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination_dir)

    print("Todos los archivos zip han sido descomprimidos en la carpeta de destino.")



#--------------------------------------------------------------------------------------------------------------------------
# Funcion procesar_atom_folder(df)
#--------------------------------------------------------------------------------------------------------------------------

def procesar_atom_folder(carpeta):

    """
    Este módulo permite procesar archivos .atom provenientes de la Plataforma de Contratación del Sector Público (PLACSP),
    siguiendo la estructura definida por la herramienta OpenPLACSP (versión 2.2). Extrae información detallada de cada
    entrada de licitación y adjudicación, generando tres archivos CSV:

    1. `licitaciones_consolidado.csv`: contiene los datos generales de cada licitación (capítulo 5.1 del manual).
    2. `resultados_consolidado.csv`: contiene los datos de adjudicación y resolución (capítulo 5.2 del manual).
    3. `licitaciones_resultados_consolidado.csv`: combinación de ambos conjuntos de datos en una sola tabla.

    Funciones principales:
    - `t(elem, path)`: Acceso seguro al texto de un nodo XML con namespaces.
    - `id_by_scheme(party, scheme)`: Extrae identificadores específicos (NIF, ID_PLATAFORMA, etc.) de una entidad.
    - `parse_entry(entry)`: Extrae todos los campos relevantes de una entrada Atom, devolviendo tres diccionarios:
    licitación, resultado y combinado.
    - `procesar_atom_folder(carpeta)`: Recorre todos los archivos .atom/.txt de una carpeta, aplica `parse_entry` y
    guarda los resultados en CSV.

    Namespaces utilizados:
    - `atom`, `cbc`, `cac`, `cbc-place-ext`, `cac-place-ext`, `at`: definidos según los esquemas XML de PLACSP y CODICE.

    Este script está diseñado para facilitar la reutilización de datos abiertos de contratación pública, permitiendo su
    análisis posterior en herramientas como pandas, Excel o Power BI.
    """

    dir = "./src/data"
    
    # ▸ Namespaces reales usados en los .atom de PLACSP
    NS = {
        'atom': 'http://www.w3.org/2005/Atom',
        'cbc':  'urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2',
        'cac':  'urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2',
        # Extensiones “place‑ext” (¡faltaban!)
        'cbc-place-ext': 'urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2',
        'cac-place-ext': 'urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2',
        'at':   'http://purl.org/atompub/tombstones/1.0'
    }

    # ───────────────────────────────────────────────────────── helper
    def t(elem, path):
        """Shortcut para extraer texto sin explotar en caso de None."""
        node = elem.find(path, NS)
        return node.text.strip() if node is not None and node.text else ""

    def id_by_scheme(party, scheme):
        """Devuelve el ID cuyo atributo schemeName == scheme."""
        for ident in party.findall('cac:PartyIdentification', NS):
            id_node = ident.find('cbc:ID', NS)
            if id_node is not None and id_node.attrib.get('schemeName') == scheme:
                return id_node.text.strip()
        return ""

    # ───────────────────────────────────────────────────────── core
    def parse_entry(entry):
        lic, res = {}, {}

        # ------------- cabecera ATOM
        lic['Número de expediente'] = t(entry, 'cac-place-ext:ContractFolderStatus/cbc:ContractFolderID')
        lic['Link licitación']      = t(entry, 'atom:id')
        lic['Fecha de actualización'] = t(entry, 'atom:updated')
        lic['Objeto del Contrato']  = t(entry, 'atom:title')
        lic['Vigente/Anulada/Archivada'] = 'Anulada' if 'ANULADA' in t(entry, 'atom:summary') else 'Vigente'

        # ------------- Proyecto / presupuesto
        pp = entry.find('cac-place-ext:ContractFolderStatus/cac:ProcurementProject', NS)
        if pp is not None:
            lic['Contrato mixto'] = t(pp, 'cbc:MixContractIndicator')
            lic['CPV']            = t(pp, 'cac:RequiredCommodityClassification/cbc:ItemClassificationCode')
            lic['Tipo de contrato'] = t(pp, 'cbc:TypeCode')
            lic['Lugar de ejecución'] = t(pp, 'cac:RealizedLocation/cbc:CountrySubentityCode')
            ba = pp.find('cac:BudgetAmount', NS)
            if ba is not None:
                lic['Presupuesto base con impuestos']  = t(ba, 'cbc:TotalAmount')
                lic['Presupuesto base sin impuestos']  = t(ba, 'cbc:TaxExclusiveAmount')
                lic['Valor estimado del contrato']     = t(ba, 'cbc:EstimatedOverallContractAmount')

        # ------------- Órgano de contratación
        loc_party = entry.find('cac-place-ext:ContractFolderStatus/cac-place-ext:LocatedContractingParty', NS)
        if loc_party is not None:
            party = loc_party.find('cac:Party', NS)
            if party is not None:
                lic['Órgano de contratación'] = t(party, 'cac:PartyName/cbc:Name')
                lic['NIF OC']  = id_by_scheme(party, 'NIF')
                lic['ID OC en PLACSP'] = id_by_scheme(party, 'ID_PLATAFORMA')
            lic['Enlace al Perfil de Contratante del OC'] = t(loc_party, 'cbc:BuyerProfileURIID')

        # ------------- Proceso de licitación
        tp = entry.find('cac-place-ext:ContractFolderStatus/cac:TenderingProcess', NS)
        if tp is not None:
            lic['Tipo de Procedimiento']           = t(tp, 'cbc:ProcedureCode')
            lic['Sistema de contratación']         = t(tp, 'cbc:ContractingSystemCode')
            lic['Tramitación']                     = t(tp, 'cbc:UrgencyCode')
            lic['Forma de presentación de la oferta'] = t(tp, 'cbc:SubmissionMethodCode')
            lic['Subasta electrónica']             = t(tp, 'cac:AuctionTerms/cbc:AuctionConstraintIndicator')

        # ---------- RESULTADOS (puede no haber)
        res['Número de expediente']   = lic['Número de expediente']
        res['Link licitación']        = lic['Link licitación']
        res['Fecha de actualización'] = lic['Fecha de actualización']
        res['Objeto de licitación/lote'] = lic['Objeto del Contrato']
        res['CPV licitación/lote']    = lic['CPV']
        res['Presupuesto base con impuestos/lote'] = lic.get('Presupuesto base con impuestos', '')
        res['Presupuesto base sin impuestos/lote'] = lic.get('Presupuesto base sin impuestos', '')
        res['Valor estimado licitación/lote']      = lic.get('Valor estimado del contrato', '')
        res['Lugar de ejecución licitación/lote']  = lic['Lugar de ejecución']

        tr = entry.find('cac-place-ext:ContractFolderStatus/cac:TenderResult', NS)
        if tr is not None:
            res['Resultado licitación/lote']     = t(tr, 'cbc:Description')
            res['Fecha del acuerdo licitación/lote'] = t(tr, 'cbc:AwardDate')
            res['Número de ofertas recibidas por licitación/lote'] = t(tr, 'cbc:ReceivedTenderQuantity')
            res['Precio de la oferta más baja']  = t(tr, 'cbc:LowerTenderAmount')
            res['Precio de la oferta más alta']  = t(tr, 'cbc:HigherTenderAmount')
            res['Se han excluido ofertas anormalmente bajas'] = t(tr, 'cbc:AbnormallyLowTendersIndicator')

            contract = tr.find('cac:Contract', NS)
            if contract is not None:
                res['Número de contrato']           = t(contract, 'cbc:ID')
                res['Fecha formalización del contrato'] = t(contract, 'cbc:IssueDate')

            wp = tr.find('cac:WinningParty', NS)
            if wp is not None:
                res['Adjudicatario']                = t(wp, 'cac:PartyName/cbc:Name')
                res['Identificador adjudicatario']  = t(wp, 'cac:PartyIdentification/cbc:ID')
                id_node = wp.find('cac:PartyIdentification/cbc:ID', NS)
                res['Tipo de identificador de adjudicatario'] = id_node.attrib.get('schemeName','') if id_node is not None else ''
            monetario = tr.find('cac:AwardedTenderedProject/cac:LegalMonetaryTotal', NS)
            if monetario is not None:
                res['Importe adjudicación sin impuestos'] = t(monetario, 'cbc:TaxExclusiveAmount')
                res['Importe adjudicación con impuestos'] = t(monetario, 'cbc:PayableAmount')

        # --------- fusionar (lic + res → todo en una fila)
        combinado = lic.copy()
        for k, v in res.items():
            if k not in combinado:          # evita sobrescribir claves comunes
                combinado[k] = v

        return lic, res, combinado

    # ───────────────────────────────────────────────────────── main

    lic_rows, res_rows, all_rows = [], [], []

    for fname in os.listdir(carpeta):
        if fname.endswith(('.atom', '.txt')):
            path = os.path.join(carpeta, fname)
            root = ET.parse(path).getroot()
            for entry in root.findall('atom:entry', NS):
                lic, res, comb = parse_entry(entry)
                lic_rows.append(lic)
                res_rows.append(res)
                all_rows.append(comb)

    # DataFrames
    df_lic = pd.DataFrame(lic_rows).drop_duplicates()
    df_res = pd.DataFrame(res_rows).drop_duplicates()
    df_all = pd.DataFrame(all_rows).drop_duplicates()

    # Guardar CSV
    df_lic.to_csv(os.path.join(carpeta, 'licitaciones_consolidado.csv'), index=False, encoding='utf-8')
    df_res.to_csv(os.path.join(carpeta, 'resultados_consolidado.csv'), index=False, encoding='utf-8')
    df_all.to_csv(os.path.join(carpeta, 'licitaciones_resultados_consolidado.csv'), index=False, encoding='utf-8')

    print("3 archivos generados en:", carpeta)