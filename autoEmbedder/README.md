# Text Autoencoder Learning Project

Learn how text autoencoders work through hands-on examples!

## What is a Text Autoencoder?

An autoencoder is a neural network that compresses data into a lower-dimensional representation and reconstructs it back. For text:

```
Original Text (Variable Length)
         ↓
    [ENCODER]
         ↓
  Embedding (384 dims) ← Compressed representation
         ↓
    [DECODER]
         ↓
Reconstructed Text
```

The **encoder** converts text → embeddings (compression)
The **decoder** converts embeddings → reconstructed text (decompression)

## Project Structure

### 1. `1_download_model.py` - Setup
- Downloads a pre-trained text autoencoder model
- Installs required dependencies
- **Run first!**

```bash
python 1_download_model.py
```

### 2. `2_embed_text.py` - Encoding Tutorial
Learn how text becomes numerical embeddings:
- Convert text to embeddings
- Understand embedding dimensions
- Explore semantic similarity between texts

```bash
python 2_embed_text.py
```

**Key concepts:**
- Embeddings capture semantic meaning
- Similar texts have similar embeddings
- 384-dimensional vector for all input

### 3. `3_unembed_text.py` - Decoding Tutorial
Learn about reconstruction:
- Approximate decoding through similarity search
- Find the most similar text to an embedding
- Understand the "bottleneck" concept

```bash
python 3_unembed_text.py
```

**Key concepts:**
- Decoder reverses the encoder
- We approximate with similarity search
- Similarity score shows reconstruction quality

### 4. `4_autoencoder_test.py` - Complete Workflow
Full autoencoder test:
- Encode test texts
- Reconstruct from embeddings
- Perform vector arithmetic
- Analyze clustering behavior
- Save results to JSON

```bash
python 4_autoencoder_test.py
```

**Key concepts:**
- Complete embedding → reconstruction cycle
- Vector arithmetic in embedding space
- Clustering via centroid analysis
- Generates `test_results.json`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run scripts in order
python 1_download_model.py
python 2_embed_text.py
python 3_unembed_text.py
python 4_autoencoder_test.py
```

## Model Used

**All-MiniLM-L6-v2**
- Lightweight & fast (great for learning)
- 384-dimensional embeddings
- Trained on semantic similarity
- ~22MB size
- Automatically cached after first download

## What You'll Learn

1. **Encoding**: How text becomes vectors
   - Text → Embedding (384 dimensions)
   - Semantic meaning in vector space
   
2. **Decoding**: How to reconstruct from vectors
   - Embedding → Similar text
   - Reconstruction quality vs compression
   
3. **Vector Operations**: Math in embedding space
   - Vector arithmetic preserves semantics
   - Similarity calculations
   - Centroid analysis for clustering

4. **Applications**:
   - Semantic search
   - Text clustering
   - Anomaly detection
   - Dimensionality reduction

## Understanding Embeddings

An embedding is a way to represent text as numbers:

```
Original:  "The dog is happy"
           ↓ (ENCODER)
Embedding: [0.234, -0.891, 0.123, ..., 0.456]  (384 values)
           ↓ (DECODER)
Approx:    "Dogs are happy animals"
```

The 384-dimensional vector captures:
- Semantic meaning
- Relationships between words
- Context and usage patterns

## Similarity in Embedding Space

Texts with similar meanings have similar embeddings:

```
"The quick brown fox" 
    ↓
[0.1, -0.2, 0.5, ...]  ← Similar to other fox sentences
    ↓
cos_similarity = 0.92 with "A speedy orange fox"
cos_similarity = 0.45 with "Machine learning is great"
```

## Next Steps

After learning these basics, you can:

1. **Build a Semantic Search Engine**
   - Embed documents
   - Query embeddings
   - Return most similar documents

2. **Fine-tune on Custom Data**
   - Retrain the model on your domain
   - Create specialized embeddings

3. **Use in RAG Systems**
   - Retrieve relevant documents
   - Provide context to LLMs

4. **Explore Advanced Architectures**
   - VAEs (Variational Autoencoders)
   - Transformer-based autoencoders
   - Denoising autoencoders

## Troubleshooting

**ImportError for sentence_transformers?**
```bash
pip install sentence-transformers
```

**Model download slow?**
- First download is slower (caches after)
- Check internet connection
- Models cached in `~/.cache/huggingface/`

**Out of memory?**
- Use smaller batch sizes
- Close other applications
- Try running on GPU if available

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Autoencoders Explained](https://en.wikipedia.org/wiki/Autoencoder)
- [Semantic Similarity](https://en.wikipedia.org/wiki/Semantic_similarity)

---

Happy learning! 🚀
