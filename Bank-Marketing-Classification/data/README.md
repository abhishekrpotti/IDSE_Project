# Dataset acquisition

1. Visit the [UCI Bank Marketing dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2. Download the official Bank Marketing ZIP archive.
3. Extract the outer ZIP archive. It contains separate dataset ZIP files.
4. Locate and extract the nested legacy `bank.zip` archive. Do not use `bank-additional.zip`; it contains a different dataset version.
5. In the extracted legacy archive, locate `bank-full.csv` and `bank-names.txt`.
6. Create the `data/raw/` directory in the repository if it does not already exist.
7. Place both `bank-full.csv` and `bank-names.txt` inside `data/raw/`.
8. `bank-full.csv` is semicolon-separated. Preserve the source file without editing it.
9. The project scripts expect the exact filename `bank-full.csv` at `data/raw/bank-full.csv` and the documentation file `bank-names.txt` at `data/raw/bank-names.txt`.

The source is the UCI Machine Learning Repository. Cite: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306). The dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license; retain appropriate attribution when sharing or adapting it. The raw data files are not included in this repository.
