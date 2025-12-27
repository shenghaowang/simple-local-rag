import random
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from tqdm import tqdm

from preprocess.utils import open_and_read_pdf, split_list


def main():
    pdf_path = Path("data/human-nutrition-text.pdf")
    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

    # Split pages into sentences
    nlp = English()
    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings (the default type is a spaCy data type)
        item["sentences"] = [str(sent) for sent in item["sentences"]]

        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])

    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"])
        item["num_chunks"] = len(item["sentence_chunks"])

    # Split each chunk into its own item
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_num"] = item["page_num"]

            # Join the sentences back into a pagraph-link structure
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(
                r"\.([A-Z])", r". \1", joined_sentence_chunk
            )  # ".A" => ". a"
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Add stats
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
            chunk_dict["chunk_token_count"] = (
                len(joined_sentence_chunk) / 4
            )  # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)

    logger.debug(f"Number of chunks: {len(pages_and_chunks)}")
    logger.debug(random.sample(pages_and_chunks, k=1))

    df = pd.DataFrame(pages_and_chunks)
    logger.info(f"Simple stats:\n{df.describe().round(2)}")

    # Filter short chunks
    min_chunk_length = 30
    pages_and_chunks_over_min_token_len = df[
        df["chunk_token_count"] >= min_chunk_length
    ].to_dict(orient="records")
    logger.debug(random.sample(pages_and_chunks_over_min_token_len, k=1))

    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2", device="cpu"
    )
    embedding_model.to("mps")
    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(
            item["sentence_chunk"], show_progress_bar=False
        )

    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    embeddings_df_save_path = Path("data/text_chunks_and_embeddings.csv")
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)


if __name__ == "__main__":
    main()
