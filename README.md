# Project Title

This repository uses SUBLIME with custom datasets for analyzing tweets with annotations on extremism and narrative features.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Quick Start](#quick-start)

## Overview

This project leverages the **SUBLIME** framework with custom datasets to perform tweet analysis focused on extremism and narrative features.

## Datasets

1. **ARENAS**
   - **Description:** Contains 298 tweets annotated for extremism and narrative features.
   - **Filename:** `2024_08_SCHEMA_A1.xlsx`
2. **Full_french_tweet_data** -**Description** Contains around 40000 tweets with various informations about the tweet and some about the user who posted it -**Filename** `All_french_tweet_data.csv`

## Experiments

The repository includes set-ups the currently implemented experiments :
All modifications of parameters are done in `experiment_params.csv`.
Default config is exp_nb 0, so start from there when modifying.

1. **Experiment 1** First try of SUBLIME
   - **Description:** Uses Schema A1 with the `french-document-embedding` model.
   - **Configuration:** Set `exp_nb` argument to `1` and `epoch` to `1000`.
   - **Execution:** Run
   ```
   python main.py -exp_nb 1
   ```
2. **Experiment 2** Filtering out tweets that are not extremist narratives

   - **Description:** Uses Schema A1 with the `french-document-embedding` model, with filtering out nodes without both IN GROUP and OUT GROUP.
   - **Configuration:** Set `exp_nb` argument to `2` and `epoch` to `1000`.
   - **Execution:** Run

   ```
   python main.py -exp_nb 2
   ```

3. **Experiment 3** First test of increasing the nb of epochs
   - **Description:** Uses Schema A1 with the `french-document-embedding` model, applies filtering as in Experiment 2, and increases the iterations of SUBLIME to `4000`.
   - **Configuration:** Set `exp_nb` argument to `3` and `epoch` to `4000`.
   - **Execution:** Run
   ```
   python main.py -exp_nb 3
   ```
4. **Experiment 4** Run SUBLIME on bigger dataset Full_french_tweet.csv
   - **Description:** Uses Full_french_tweet.csv with the `french-document-embedding` model, applies filtering as in Experiment 2, and increases the iterations of SUBLIME to `4000`.
   - **Configuration:** Set `exp_nb` argument to `4` and `epoch` to `4000`.
   - **Execution:** Run
   ```
   python main.py -exp_nb 4
   ```

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
   source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
   ```

   Install Dependencies:

   ```
   pip install -r requirements.txt
   ```

   Please note that the dgl (Deep Graph Library) install is a CPU install not GPU so you may encounter errors

3. **Run the Experiment Script**
   Update the configuration in `experiment_params.csv` and run

   ```
   python main.py -exp_nb 1
   ```

   Replace 1 with the correct experiment number.
   After running this, you should have the tweet embeddings in `embeddings`, the resulting matrices in `adjacency_matrices` and a plot of the loss in `plots`.

4. **Postprocessing and Analysis**:

   Currently, choosing the adjacency matrix to study is based on changing the file name in graph_eval.py at the beginning of the init method.
   To run the analysis :

   ```
   python graph_eval.py
   ```
