# Digital Edition Draft

This repository hosts the pipeline that transforms a small corpus of Latin poems into progressively enriched TEI files (entities, topics, validation, visualizations). Each stage is a script you can re-run independently.

---

## English

### Overview
- Plain-text sources under `data_root/` go through numbered scripts (`0_*.py` … `8_*.py`).
- Intermediate TEI stages live in `data_root/tei_pipeline/<step>/`, ending with Relax NG–validated TEI in `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
- Supporting CSVs (entities, VIAF, Pleiades, Wikidata, topics) live under `data_root/entities/` and `data_root/outputs/`.

### Requirements
- Python 3.10+ (`py -3.11` used on Windows).
- Install deps: `pip install -r requirements.txt`.
- LatinCy model is not tracked; install it with:
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`

### Running the pipeline
1) Ensure `data_root/` has the poems and optional PDFs (Index Nominum, Pleiades CSVs). Sample data is included.
2) Run scripts in order; each can be rerun alone:
   - `0_tei_header_and_verses.py` → TEI skeletons + verse counting (02/03).
   - `1_entities_pipeline.py` → NER tagging into TEI (04) + entity CSVs/index checks.
   - `2_viaf_linking.py`, `3_pleiades_linking.py`, `4_wikidata_linking.py` → enrich entity CSV. (Wikidata may hit HTTP 403 if rate-limited.)
   - `5_topic_matching.py` → Soldevila matches to CSV.
   - `6_tei_topic_annotation.py` → apply topics (05 standoff, 06 flatten, 07 combined).
   - `7_tei_relaxng_header_and_validation.py` → add xml-model + validate into `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
3) Final TEI are in `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

---

## Español

### Descripción general
- Los textos en `data_root/` pasan por scripts numerados (`0_` a `8_`) hasta producir TEI enriquecidos y validados.
- Cada etapa guarda su salida en `data_root/tei_pipeline/<etapa>/`; los TEI finales validados quedan en `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
- Los CSV de entidades y enlaces externos están en `data_root/entities/` y los de tópicos en `data_root/outputs/`.

### Requisitos
- Python 3.10+ (`py -3.11` en Windows).
- Instala dependencias con `pip install -r requirements.txt`.
- El modelo `la_core_web_lg` no se incluye: instálalo con
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`.

### Ejecución
1) Verifica que `data_root/` tenga textos y (opcional) PDFs de índices/Pleiades.
2) Corre los scripts en orden:
   `0`→TEI base, `1`→NER/CSV, `2` VIAF, `3` Pleiades, `4` Wikidata (puede dar 403),
   `5` tópicos (CSV), `6` tópicos en TEI (05/06/07), `7` cabecera+validación (08).
3) Revisa los TEI finales en `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

---

## Português

### Visão geral
- Os textos em `data_root/` passam pelos scripts `0_` a `8_`, gerando TEI enriquecidos e validados.
- Cada fase grava em `data_root/tei_pipeline/<etapa>/`; os TEI finais validados ficam em `08_tei_header_with_relaxng/{standoff,flatten,combined}`.
- CSVs de entidades e enlaces ficam em `data_root/entities/`; matches de tópicos em `data_root/outputs/`.

### Requisitos
- Python 3.10+.
- `pip install -r requirements.txt`.
- Baixe o modelo spaCy `la_core_web_lg` com
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`.

### Como executar
1) Certifique-se de que `data_root/` contenha os textos e (opcional) PDFs.
2) Rode os scripts em ordem (`0`→TEI base, `1`→NER/CSV, `2`/`3`/`4`→enriquecimento, `5`→tópicos CSV, `6`→tópicos em TEI, `7`→cabeçalho+validação).
3) TEI finais em `data_root/tei_pipeline/08_tei_header_with_relaxng/`.

---

## data_root_export
- `texts/`: muestra de poemas y diccionario de ejemplo (puedes borrar lo que no quieras compartir).
- `pleiades/`: CSVs completos de Pleiades para reproducir el enriquecimiento.
- (Opcional) puedes añadir TEI de muestra aquí si deseas compartir ejemplos, pero el pipeline los regenera.

## Modelos
- Instalar LatinCy grande (no incluido por peso):
  `py -3 -m pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl`
