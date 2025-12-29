import random

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.generator import ask
from rag.retriever import print_top_results_and_scores, retrieve_relevant_resources


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

    # Nutrition-style questions generated with GPT4
    gpt4_questions = [
        "What are the macronutrients, and what roles do they play in the human body?",
        "How do vitamins and minerals differ in their roles and importance for health?",
        "Describe the process of digestion and absorption of nutrients in the human body.",
        "What role does fibre play in digestion? Name five fibre containing foods.",
        "Explain the concept of energy balance and its importance in weight management.",
    ]

    # Manually created question list
    manual_questions = [
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the RDI for protein per day?",
        "water soluble vitamins",
    ]

    query_list = gpt4_questions + manual_questions
    query = random.choice(query_list)

    print_top_results_and_scores(
        query=query,
        embeddings=embeddings,
        embedding_model=embedding_model,
        pages_and_chunks=pages_and_chunks,
        n_resources_to_return=5,
    )

    # Load an LLM model locally
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_type=torch.float16,
    # )
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        low_cpu_mem_usage=False,  # use as much memory as we can
        # attn_implementation="sdpa"
        attn_implementation="eager",
    ).to(device)
    # logger.debug(f"Loaded tokenizer: \n{tokenizer}")
    logger.debug(f"Loaded LLM model: \n{llm_model}")

    # Create prompt template
    dialogue_template = [
        {
            "role": "user",
            "content": query,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the input text
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    logger.debug(f"Input IDs: {input_ids}")

    # Generate outputs from local LLM
    outputs = llm_model.generate(**input_ids, max_new_tokens=256)
    logger.debug(f"Model output (tokens): \n{outputs[0]}\n")

    # Decode the output tokens to text
    decoded_output = tokenizer.decode(outputs[0])
    logger.info(f"Model output (decoded): \n{decoded_output}\n")

    # Get relevant resources
    _, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
    )

    # Create a list of context items
    context_items = [pages_and_chunks[idx] for idx in indices]

    # Generate an answer using RAG
    output_text = ask(
        query=query,
        context_items=context_items,
        tokenizer=tokenizer,
        llm_model=llm_model,
        device=device,
        format_answer_text=True,
    )

    logger.info(f"Query: {query}")
    logger.info(f"RAG answer:\n{output_text}")


if __name__ == "__main__":
    main()
