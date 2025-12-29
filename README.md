# NutriChat

Create and run a local RAG pipeline from scratch to chat with a nutrition textbook. Adapted from Daniel Bourke's [RAG tutorial](https://github.com/mrdbourke/simple-local-rag).

## What is RAG?

RAG stands for Retrieval Augmented Generation.

The goal of RAG is to take information and pass it to an LLM so it can generate outputs based on that information.

* Retrieval - Find relevant information given a query, e.g. "what are the macronutrients and what do they do?" -> retrieves passages of text related to the macronutrients from a nutrition textbook.
* Augmented - We want to take the relevant information and augment our input (prompt) to an LLM with that relevant information.
* Generation - Take the first two steps and pass them to an LLM for generative outputs.

If you want to read where RAG came from, see the [paper](https://arxiv.org/pdf/2005.11401) from Facebook AI:

> This work offers several positive societal benefits over previous work: the fact that it is more strongly grounded in real factual knowledge (in this case Wikipedia) makes it “hallucinate” less with generations that are more factual, and offers more control and interpretability. RAG could be employed in a wide variety of scenarios with direct benefit to society, for example by endowing it with a medical index and asking it open-domain questions on that topic, or by helping people be more effective at their jobs.

## Why RAG?

The main goal of RAG is to improve the generation outputs of LLMs.

1. Prevent hallucinations - LLMs are incredibly good at generating good looking text, however, this text doesn't mean it's factual. RAG can help LLMs generate information based on relevant passages that are factual.
2. Work with custom data - Many base LLMs are trained with internet-scale data. This means they have a fairly good understanding of language in general. However, it also means a lot of their responses can be generic in nature. RAG helps to create specific responses based on specific documents (e.g. your own companies's customer support documents).

## What can RAG be used for?

* Customer support Q&A chat - Treat your existing customer support documents as a resource and when a customer asks a question, you could have a retrieval system, retrieval relevant documentation snippets and then have an LLM craft those snippets into an answer. Think of this as a "chatbot for your documentation".
* Email chain analysis - Let's say you're a large insurance company and you have chains and chains of emails of customer claims. You could use a RAG pipeline to find relevant information from those emails and then use an LLM to process that information into structured data.
* Company internal documentation chat
* Textbook Q&A - Let's say you're a nutrition student and you've got a 1200 page textbook to read. You could build a RAG pipeline to go through the textbook and find relevant passages to the questions you have.

Common theme here: take your relevant documents to a query and process them with an LLM.

From this angle, you can consider LLM as a calculator for words.

## Why Local?

Fun.

Privacy, speed, cost.

* Privacy - If you have private documentation, maybe you don't want to send that to an API. You want to setup an LLM and run it on your own hardware.
* Speed - Whenever you use an API, you have to send some kind of data across the Internet. This takes time. Running locally means we don't have to wait for transfers of data.
* Cost - If you own your hardware, the cost is paid. It may have a large cost to begin with. But overtime, you don't have to keep paying API fees.
* No vendor lockin - If you run your own software / hardware. If you OpenAI / another large internet company shut down tomorrow, you can still run your business.

## What we're going to build

* https://github.com/mrdbourke/simple-local-rag
* [RAG 101: Retrieval-Augmented Generation Questions Answered](https://developer.nvidia.com/blog/rag-101-retrieval-augmented-generation-questions-answered/)

We're going to build NutriChat to "chat with a nutrition textbook".

Specifically:

1. Open a PDF document (you could use almost any PDF here or even a collection of PDFs).
2. Format the text of the PDF textbook ready for an embedding model.
3. Embed all of the chunks of text in the textbook and turn them into numerical representations (embedding) which can store for later.
4. Build a retrieval system that uses vector search to find relevant chunk of text based on a query.
5. Create a prompt that incorporates the retrieved pieces of text.
6. Generate an answer to a query based on the passages of the textbook with an LLM.

All locally!

1. Step 1-3: Document preprocessing and embedding creation.
2. Step 4-6: Search and answer.

---

## 📦 Installation

To set up the project using a Python virtual environment, follow the steps below.

1. **Clone the repository**
```bash
git clone https://github.com/shenghaowang/simple-local-rag.git
cd simple-local-rag
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the `pre-commit` hooks**
```bash
pre-commit install
pre-commit run --all-files
```

## 🔥 Usage

### Download PDF doc

```bash
python src/data/main.py
```

### Preprocess text and export embeddings to file

```bash
export PYTHONPATH=src
python src/preprocess/main.py
```

### Retrieve relevant passages and generate output given specific query

```bash
export PYTHONPATH=src
python src/rag/main.py
```
