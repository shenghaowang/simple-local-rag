import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from rag.retriever import print_top_results_and_scores


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    text_chunks_and_embeddings_df = pd.read_csv("data/text_chunks_and_embeddings.csv")

    # Convert embedding column back to np.array
    text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df[
        "embedding"
    ].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")
    embeddings = torch.tensor(
        np.stack(text_chunks_and_embeddings_df["embedding"].tolist(), axis=0),
        dtype=torch.float32,
    ).to(device)
    logger.debug(f"Embeddings tensor shape: {embeddings.shape}")

    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2", device=device
    )

    query = "macronutrients functions"
    # query = "breastfeeding infant timeline"
    # query = "foods high in fiber"

    print_top_results_and_scores(
        query=query,
        embeddings=embeddings,
        embedding_model=embedding_model,
        pages_and_chunks=pages_and_chunks,
        n_resources_to_return=5,
    )


if __name__ == "__main__":
    main()
