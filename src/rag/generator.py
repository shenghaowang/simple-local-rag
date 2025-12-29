from typing import List, Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor


class NanInfClampLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        return torch.clamp(scores, -1e4, 1e4)


def prompt_formatter(
    query: str, context_items: List[str], tokenizer: AutoTokenizer
) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    # base_prompt = """Based on the following context items, please answer the query.
    # Context items:
    # {context}
    # Query: {query}
    # Answer:
    # """
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: What are the fat-soluble vitamins?
    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are
    absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use.
    Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in
    calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage.
    Vitamin K is essential for blood clotting and bone metabolism.
    \nExample 2:
    Query: What are the causes of type 2 diabetes?
    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories
    leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin
    resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas
    cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally,
    excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain
    and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
    \nExample 3:
    Query: What is the importance of hydration for physical performance?
    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood
    volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate
    hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased
    performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient
    water before, during, and after exercise helps ensure peak physical performance and recovery.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction tuned model
    dialogue_template = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": base_prompt},
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def ask(
    query: str,
    context_items: List[str],
    tokenizer: AutoTokenizer,
    llm_model: torch.nn.Module,
    device: torch.device,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    format_answer_text: bool = True,
) -> Tuple[str, List[str]]:
    """
    Takes a query and generates an answer to the query
    based on the relevant resources.
    """
    prompt = prompt_formatter(
        query=query,
        context_items=context_items,
        tokenizer=tokenizer,
    )
    logger.debug(f"RAG prompt:\n{prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate an output of tokens
    llm_model.eval()
    with torch.inference_mode():
        outputs = llm_model.generate(
            **input_ids,
            temperature=temperature,  # controls randomness of generation
            do_sample=True,  # whether or not to use sampling
            top_p=0.9,
            top_k=50,
            max_new_tokens=max_new_tokens,
            logits_processor=[NanInfClampLogitsProcessor()],
        )

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    # Format the answer
    if format_answer_text:
        # Replace prompt and special tokens
        output_text = (
            output_text.replace(prompt, "")
            .replace("<|begin_of_text|>", "")
            .replace("<|eot_id|>", "")
        )

    return output_text
