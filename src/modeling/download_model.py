import gdown
from pathlib import Path
import zipfile


def download_model(zip_path, extract_to):
    """
    Downloads model and unzips its.

    Parameters:
    - zip_path (str or Path): Path to the .zip file
    - extract_to (str or Path): Directory where files should be extracted

    Returns:
    - Path: The path to the extraction directory
    """

    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    url = "https://drive.google.com/uc?id=1dMFYAkmqTd0fYiwuXXodSX8Kb3rDdaXH"
    output = str(zip_path.resolve())
    gdown.download(url, output)

    # Make sure the destination directory exists
    extract_to.mkdir(parents=True, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extracted {zip_path.name} to {extract_to.resolve()}")
    return extract_to
