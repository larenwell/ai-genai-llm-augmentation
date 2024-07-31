import os
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
import evaluate
import logging

# Define constants
URL = "http://localhost:6333"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
QDRANT_COLLECTION_NAME = "gutenberg_book_embeddings_cleaned"
token = os.getenv("HUGGING_FACE_API_TOKEN")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the sentence transformer model for generating embeddings
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize the Qdrant client
client = QdrantClient(
    url=URL, 
    prefer_grpc=False
)

# Initialize the Qdrant database with the client and embedding model
db = Qdrant(client=client, embeddings=embedding_model, collection_name=QDRANT_COLLECTION_NAME)

def docs_retriever_response(question: str, top_k: int = 4) -> list:
    """
    Retrieve top-k relevant documents from Qdrant based on a question.

    Args:
        question (str): The question to retrieve documents for.
        top_k (int): The number of top documents to retrieve.

    Returns:
        list: A list of the top-k relevant documents.
    """
    query_vector = embedding_model.encode(question)
    results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    documents = []
    for result in results:
        payload = result.payload
        documents.append(payload.get("text", ""))
    return documents

async def llm_response(question: str, context: str) -> str:
    """
    Generate a response from a language model based on a question and context.

    Args:
        question (str): The question to generate a response for.
        context (str): The context to use for generating the response.

    Returns:
        str: The generated response.
    """
    try:
        # Read the models from Hugging Face Hub
        best_model_finetuned_hf = "larenwell/flan_t5_small_finetuned_kmfodabooksum_75books"
        tokenizer = T5Tokenizer.from_pretrained(best_model_finetuned_hf)
        model = T5ForConditionalGeneration.from_pretrained(best_model_finetuned_hf)

        # Optional: Uncomment and replace the model name to use a different Llama model
        # model_hf = "meta-llama/Llama-2-7b-hf"
        # tokenizer = LlamaTokenizer.from_pretrained(model_hf, use_auth_token=token)
        # model = LlamaForCausalLM.from_pretrained(model_hf, use_auth_token=token)

        formatted_prompt = f"Provide a detailed summary of the following content based on the question '{question}': {context}"

        inputs = tokenizer.encode_plus(
            formatted_prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return ""

def combine_docs(docs: list) -> str:
    """
    Combine a list of documents into a single string.

    Args:
        docs (list): A list of documents to combine.

    Returns:
        str: The combined documents as a single string.
    """
    return "\n\n".join(docs)

def enhance_summary_with_context(summary: str, context: str) -> str:
    """
    Enhance a summary by adding context.

    Args:
        summary (str): The summary to enhance.
        context (str): The context to add to the summary.

    Returns:
        str: The enhanced summary with added context.
    """
    return f"{summary}\n\nContext: {context}"

def evaluate_summary(generated_summary: str, reference_summary: str) -> dict:
    """
    Evaluate the generated summary using ROUGE and BLEU metrics.

    Args:
        generated_summary (str): The generated summary to evaluate.
        reference_summary (str): The reference summary to compare against.

    Returns:
        dict: A dictionary containing ROUGE and BLEU scores.
    """
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')

    rouge_result = rouge.compute(predictions=[generated_summary], references=[reference_summary])
    bleu_result = bleu.compute(predictions=[generated_summary], references=[reference_summary])
    
    return {
        "rouge": rouge_result,
        "bleu": bleu_result
    }

async def docs_retriever(question: str) -> str:
    """
    Retrieve documents based on a question and generate an enhanced summary.

    Args:
        question (str): The question to retrieve documents for.

    Returns:
        str: The enhanced summary with context.
    """
    retrieved_docs = docs_retriever_response(question)
    formatted_context = combine_docs(retrieved_docs)

    print("Formatted Context:", formatted_context)
    logger.info("Formatted Context: %s", formatted_context)

    initial_summary = await llm_response(question, formatted_context)
    enhanced_summary = enhance_summary_with_context(initial_summary, formatted_context)
    
    return enhanced_summary

