import ollama
import os
from openai import OpenAI
import numpy as np


def read_document(filepath):
    """Read text from a file."""
    with open(filepath, 'r') as file:
        return file.read()


def truncate_text(text, max_length=2000):
    """Truncate text to max length."""
    if len(text) > max_length:
        return text[:max_length]
    return text


def get_embedding(text):
    """Get embedding for text using Ollama."""
    return ollama.embed(
        model='nomic-embed-text',
        input=text,
    ).embeddings


def edit_document(text):
    """Edit document using DeepSeek API."""
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    return client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "replace all of the nouns in this text with the word rock but keep EVERYTHING else the same" + text},
        ],
    ).choices[0].message.content


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1[0])
    vec2 = np.array(vec2[0])
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    vec1 = np.array(vec1[0])
    vec2 = np.array(vec2[0])
    return np.linalg.norm(vec1 - vec2)


def magnitude_difference(vec1, vec2):
    """Calculate difference in vector magnitudes."""
    mag1 = np.linalg.norm(np.array(vec1[0]))
    mag2 = np.linalg.norm(np.array(vec2[0]))
    return abs(mag1 - mag2)


def max_element_change(vec1, vec2):
    """Find maximum element-wise change."""
    vec1 = np.array(vec1[0])
    vec2 = np.array(vec2[0])
    changes = np.abs(vec1 - vec2)
    return np.max(changes)


def main():
    """Main function to process document."""
    textFromDoc = read_document('source_docs/DOC_1.txt')
    textFromDoc = truncate_text(textFromDoc)
    
    print("Original text:", textFromDoc[:100] + "...\n")
    
    docEmbedding = get_embedding(textFromDoc)
    editedDoc = edit_document(textFromDoc)
    editedDocEmbedding = get_embedding(editedDoc)
    
    print("Edited text:", editedDoc[:100] + "...\n")
    
    # Analysis tests
    print("=== Embedding Analysis ===")
    print(f"Cosine Similarity: {cosine_similarity(docEmbedding, editedDocEmbedding):.4f}")
    print(f"Euclidean Distance: {euclidean_distance(docEmbedding, editedDocEmbedding):.4f}")
    print(f"Magnitude Difference: {magnitude_difference(docEmbedding, editedDocEmbedding):.4f}")
    print(f"Max Element Change: {max_element_change(docEmbedding, editedDocEmbedding):.4f}")


if __name__ == "__main__":
    main()




