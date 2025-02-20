# Project Title

This repository uses SUBLIME with custom datasets for analyzing tweets with annotations on extremism and narrative features.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Quick Start](#quick-start)

## Overview

This project leverages the **SUBLIME** framework with custom datasets to perform tweet analysis focused on extremism and narrative features. The primary dataset consists of annotated tweets from 2024.

## Datasets

1. **ARENAS**
   - **Description:** Contains 303 tweets annotated for extremism and narrative features.
   - **Filename:** `2024_08_SCHEMA_A1.xlsx`

## Experiments

The repository includes set-ups for three experiments using Schema A1:

1. **Experiment 1** First try of SUBLIME
   - **Description:** Uses Schema A1 with the `french-document-embedding` model.
   - **Configuration:** Modify `arenas_si.sh`, set `exp_nb` argument to `1` and `epoch` to `1000`.
   - **Execution:** Run `arenas_si.sh` as is.
2. **Experiment 2** Filtering out tweets that are not extremist narratives

   - **Description:** Uses Schema A1 with the `french-document-embedding` model, with filtering out nodes without both IN GROUP and OUT GROUP.
   - **Configuration:** Modify `arenas_si.sh`, set `exp_nb` argument to `2` and `epoch` to `1000`.
   - **Execution:** Run `arenas_si.sh` after making modifications.

3. **Experiment 3** First test of increasing the nb of epochs
   - **Description:** Uses Schema A1 with the `french-document-embedding` model, applies filtering as in Experiment 2, and increases the iterations of SUBLIME to `4000`.
   - **Configuration:** Modify `arenas_si.sh`, set `exp_nb` argument to `3` and `epoch` to `4000`.
   - **Execution:** Run `arenas_si.sh` after making modifications.

## Quick Start

1. **Clone the Repository:**

   ```
   git clone https://github.com/T2WIN/projet_recherche_arenas.git
   cd projet_recherche_arenas
   ```

2. **Installation**
   Set Up Virtual Environment:

   ```
   python3 -m venv .venv
   source .venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

   Install Dependencies:

   ```
   pip install -r requirements.txt
   ```

3. **Run the Experiment Script**
   Update the configuration in arenas_si.sh according to the desired experiment.

   ```
   source arenas_si.sh
   ```

4. **Postprocessing and Analysis**:

   Currently, analysis of results is based on changing the file name in graph_eval.py at the beginning of the init method.
   To run the analysis :

   ```
   python graph_eval.py
   ```
