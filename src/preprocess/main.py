import random
from pathlib import Path

import requests
from loguru import logger

from preprocess.utils import open_and_read_pdf


def main():
    pdf_path = Path("human-nutrition-text.pdf")

    if not pdf_path.exists():
        logger.info("File doesn't exist. Downloading...")

        # Enter the URL of the PDF
        url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the file and save it
            with open(pdf_path, "wb") as file:
                file.write(response.content)
            logger.info(f"File downloaded and saved as {pdf_path}")
        else:
            logger.error(
                f"Failed to download the file. Status code: {response.status_code}"
            )

    else:
        logger.info("File already exists. Skipping download.")

    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
    logger.debug(random.sample(pages_and_texts, 3))


if __name__ == "__main__":
    main()
