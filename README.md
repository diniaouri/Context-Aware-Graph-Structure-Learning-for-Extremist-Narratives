# Project Title

This repository uses SUBLIME with custom datasets for analyzing tweets with annotations on extremism and narrative features.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview

This project leverages the **SUBLIME** framework with custom datasets to perform tweet analysis focused on extremism and narrative features. The primary dataset consists of annotated tweets from 2024.

_Additional background information about the project:_

- _[Insert detailed project background and motivation here]_

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

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Ensure the Required Datasets and Scripts are in Place**

   Place 2024_08_SCHEMA_A1.xlsx into the designated datasets folder.
   Check that arenas_si.sh is configured as described in the experiments section.

3. **Run the Experiment Script**

```bash
chmod +x arenas_si.sh
./arenas_si.sh
```

4. **Installation**

   Prerequisites:
   Python (version 3.8 or higher)
   SUBLIME framework or repository
   Necessary Python packages: pandas, numpy, etc. (see requirements.txt for complete list)

   Set Up Virtual Environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

Install Dependencies:
bash

    pip install -r requirements.txt

Usage

    Preprocessing:
        Run preprocessing scripts to prepare the data.
        Example command:
        bash

    python preprocess.py --input datasets/2024_08_SCHEMA_A1.xlsx --output processed_data.csv

Running Experiments:

    Update the configuration in arenas_si.sh according to the desired experiment.
    Execute the script:
    bash

    ./arenas_si.sh

Postprocessing and Analysis:

    Use the provided analysis scripts to evaluate experiment results.
    Example command:
    bash

        python analyze_results.py --input results/experiment1_output.csv

Contributing

Contributions are welcome! To contribute to this project, please follow these steps:

    Fork the repository.
    Create a new branch for your feature or bug fix:
    bash

git checkout -b feature/your-feature-name

Commit your changes:
bash

git commit -am 'Add your feature'

Push to your branch:
bash

    git push origin feature/your-feature-name

    Open a pull request on GitHub.

For major changes, please open an issue first to discuss what you would like to change. Make sure to update tests as appropriate.
License

This project is licensed under the MIT License.
(Alternatively, update this section with your project's chosen license information.)
Contact

For any inquiries or feedback, please contact:

    Name: Your Name
    Email: your.email@example.com
    LinkedIn/GitHub: Your Profile

Acknowledgements

    Thanks to the contributors of the SUBLIME framework.
    Acknowledgement to researchers and institutions that supported the project.
    Any other people or projects that helped or inspired you.

```

```

```

```
