import textwrap
from time import perf_counter as timer
from typing import Dict, List

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer, util


def retrieve_relevant_resources(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer,
    n_resources_to_return: int = 5,
    print_time: bool = True,
):
    """
    Embeds a query with model and returns top k scores
    and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    end_time = timer()

    if print_time:
        logger.info(
            f"Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds."
        )

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices


def print_top_results_and_scores(
    query: str,
    embeddings: torch.Tensor,
    embedding_model: SentenceTransformer,
    pages_and_chunks: List[Dict],
    n_resources_to_return: int = 5,
):
    """
    Finds relevant passages given a query and prints them out
    along with their scores.
    """
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
        n_resources_to_return=n_resources_to_return,
    )

    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        logger.info(f"Score: {score:.4f}")
        logger.info(f"Text:\n{wrap_text(pages_and_chunks[idx]['sentence_chunk'])}")
        logger.info(f"Page number: {pages_and_chunks[idx]['page_num']}")
        logger.info("\n")


def wrap_text(text: str, wrap_length: int = 80) -> str:
    """Wrap text for better readability in console."""

    return textwrap.fill(text, width=wrap_length)
