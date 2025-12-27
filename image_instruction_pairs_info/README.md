# Test Set Metadata

This directory contains the full metadata for the 226 instruction–target pairs used in our evaluation, as described in the paper.

## Files
- `metadata.csv`: Standard CSV format (recommended for quick viewing on GitHub).
- `metadata.xlsx`: Excel format for easier local filtering and sorting.

## Column Descriptions
* **`image_name`**: The filename of the original source image.
* **`instruct`**: The specific editing instruction provided to the model.
* **`target_description`**: A textual description of the expected output.
* **`image_address`**: Direct URL to the source image (Unsplash, Pixabay, or Flickr) for reproducibility.
* **`result_image_name`**: The filename of the generated output image.
* **`Type`**: The category of the instruction (e.g., Absolute Direction).

## Reproducibility
For transparency, we have included the direct source links for all images. These images are predominantly sourced from Unsplash, with additional samples from Pixabay and Flickr, and are all publicly available for download.
