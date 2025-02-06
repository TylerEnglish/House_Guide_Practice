# Raw Data

This directory contains raw data files that serve as the input for data processing and analysis pipelines.

## Guidelines

1. **Data Format**: Ensure that all raw data files are in approated formats and adhere to the expected schema and structure.
2. **File Naming Conventions**: Use clear, descriptive names for the data files to indicate their content and date.
3. **Data Integrity**: Ensure that the raw data is clean and free of inconsistencies. Perform initial validation checks before placing data in this directory.
4. **Updating Data**: If new data files are added or existing files are updated, document these changes and maintain version control.
5. **Data Source**: Mention the source of the data (e.g., API, database export, manual entry).

## Usage

- To process the raw data, follow the instructions in the pipeline scripts located in the `scripts/` directory.
- The raw data is typically read and processed by scripts such as `logic.py`.

## Additional Notes

- Make sure to keep a backup of original raw data files before any processing or transformation steps.
- This directory is read-only for processing scripts; do not modify the raw data files during processing.
