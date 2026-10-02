import os
import openai
import ollama
import numpy as np
from pathlib import Path
import re

# Configure Poe API
poe_client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY", ""),
    base_url="https://api.poe.com/v1",
)


class SemanticShifter:
    """Generates semantic shifts of text for differential fingerprinting."""
    
    SHIFT_PROMPTS = [
        # Shift 1: Extract knowledge graph triples
        """Extract the key relationships from this text as subject-predicate-object triples. 
        Format: "X does Y to Z" or "A has property B" etc.
        
        Text: {text}
        
        Triples:""",
        
        # Shift 2: Abstract to higher conceptual level
        """Rewrite this text at a higher level of abstraction, replacing specific details with general concepts.
        
        Text: {text}
        
        Abstract version:""",
        
        # Shift 3: Remove qualifiers and hedge words
        """Rewrite this text removing all qualifiers, hedge words, and uncertainty. State everything as definite assertions.
        
        Text: {text}
        
        Definite version:""",
        
        # Shift 4: Convert to timeless logical statements
        """Convert this text into timeless logical statements, removing temporal references and making it universal.
        
        Text: {text}
        
        Logical statements:""",
        
        # Shift 5: Extract core technical terminology
        """List the core technical terms and concepts from this text, along with brief definitions.
        
        Text: {text}
        
        Technical terms:""",
        
        # Shift 6: Reverse all negations
        """Rewrite this text by reversing all negations (NOT becomes affirmative, affirmative becomes NOT).
        
        Text: {text}
        
        Negation-reversed version:""",
        
        # Shift 7: Compress to essential information
        """Compress this text to only the most essential information, removing all redundancy.
        
        Text: {text}
        
        Compressed version:""",
        
        # Shift 8: Extract implicit assumptions
        """What assumptions or preconditions are implicit in this text? List them.
        
        Text: {text}
        
        Implicit assumptions:"""
    ]
    
    def generate_all_shifts(self, text: str) -> list[str]:
        """Generate all 8 semantic shifts of the text."""
        shifts = []
        for i, prompt_template in enumerate(self.SHIFT_PROMPTS, 1):
            print(f"  Generating shift {i}/8...")
            prompt = prompt_template.format(text=text)
            
            response = poe_client.chat.completions.create(
                model="llama-3.1-8b-cs",
                messages=[{"role": "user", "content": prompt}]
            )
            
            shifts.append(response.choices[0].message.content)
        
        return shifts


class HypotheticalDocumentGenerator:
    """Generates a hypothetical document that would answer the query using HyDE."""
    
    HYDE_PROMPT = """You are an expert assistant. A user has asked the following question:

"{query}"

Write a comprehensive, detailed hypothetical answer/document that would address this question well. 
Write as if you are explaining the answer directly. Include specific details, examples, and step-by-step explanations.
This should be a well-written document that would rank highly if it were indexed in a search engine for this query.

Hypothetical Answer Document:"""
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = self.HYDE_PROMPT.format(query=query)
        
        print("Generating hypothetical document that would answer the query...")
        
        response = poe_client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content


class DocumentEmbeddings:
    """Loads cached embeddings for a document."""
    
    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        self.original_embedding = None
        self.shift_embeddings = []
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings from cached files."""
        # Load original document
        original_file = self.doc_path / f"{self.doc_path.name}.txt"
        self.original_text = original_file.read_text()
        self.original_embedding = self._extract_embedding_from_file(original_file)
        
        # Load shift embeddings
        shifts_file = self.doc_path / f"{self.doc_path.name}_shifts.txt"
        if shifts_file.exists():
            self.shift_embeddings = self._extract_shift_embeddings(shifts_file)
    
    def _extract_embedding_from_file(self, file_path: Path) -> np.ndarray:
        """Extract embedding vector from file header."""
        content = file_path.read_text()
        first_line = content.split('\n')[0]
        vector_str = first_line.strip('[]')
        vector = np.array([float(x) for x in vector_str.split(', ')])
        return vector
    
    def _extract_shift_embeddings(self, shifts_file: Path) -> list[np.ndarray]:
        """Extract 8 shift embeddings from shifts file."""
        content = shifts_file.read_text()
        shifts = re.split(r'={80}\nSHIFT \d+:', content)[1:]
        
        embeddings = []
        for shift_content in shifts:
            lines = shift_content.strip().split('\n')
            for line in lines:
                if line.strip().startswith('['):
                    vector_str = line.strip('[]')
                    vector = np.array([float(x) for x in vector_str.split(', ')])
                    embeddings.append(vector)
                    break
        
        return embeddings


class QueryEmbeddings:
    """Generates and embeds a query using HyDE + semantic shifts."""
    
    def __init__(self, query_text: str, hyde_generator: HypotheticalDocumentGenerator, shifter: SemanticShifter):
        self.query_text = query_text
        self.hypothetical_text = None
        self.hypothetical_embedding = None
        self.shift_embeddings = []
        self._generate_and_embed(hyde_generator, shifter)
    
    def _generate_and_embed(self, hyde_generator: HypotheticalDocumentGenerator, shifter: SemanticShifter):
        """Generate hypothetical document, shift it, and embed everything."""
        # Generate hypothetical document that would answer the query
        self.hypothetical_text = hyde_generator.generate_hypothetical_document(self.query_text)
        
        # Embed the hypothetical document
        print("Embedding hypothetical document...")
        self.hypothetical_embedding = self._embed_text(self.hypothetical_text)
        
        # Generate semantic shifts of the hypothetical document
        print("Generating semantic shifts of hypothetical document...")
        shifts = shifter.generate_all_shifts(self.hypothetical_text)
        
        # Embed each shift
        print("Embedding shifts...")
        for i, shift in enumerate(shifts, 1):
            print(f"  Embedding shift {i}/8...")
            self.shift_embeddings.append(self._embed_text(shift))
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using local Nomic."""
        response = ollama.embed(
            model='nomic-embed-text',
            input=text
        )
        return np.array(response['embeddings'][0])


class HyDEDifferentialRAG:
    """RAG system using HyDE + Directional Differential Fingerprinting."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.documents = []
        self.hyde_generator = HypotheticalDocumentGenerator()
        self.shifter = SemanticShifter()
        self._load_documents()
    
    def _load_documents(self):
        """Load all document embeddings."""
        print("Loading document embeddings...")
        doc_folders = sorted([d for d in self.docs_dir.iterdir() if d.is_dir()])
        
        for doc_folder in doc_folders:
            print(f"Loading {doc_folder.name}...")
            doc_embeddings = DocumentEmbeddings(doc_folder)
            self.documents.append(doc_embeddings)
        
        print(f"Loaded {len(self.documents)} documents\n")
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def _directional_differential_score(self, query_emb: QueryEmbeddings, doc_emb: DocumentEmbeddings) -> dict:
        """
        Compute directional differential fingerprint score.
        
        Returns a dict with:
        - 'position_score': How similar the hypothetical doc is to the real doc (baseline HyDE)
        - 'direction_score': How similar the shift DIRECTIONS are
        - 'magnitude_score': How similar the shift MAGNITUDES are
        - 'combined_score': Weighted combination
        """
        # 1. Position score (baseline HyDE similarity)
        position_score = self._cosine_similarity(
            query_emb.hypothetical_embedding,
            doc_emb.original_embedding
        )
        
        # 2. Compute deltas (shift directions)
        query_deltas = [
            shift - query_emb.hypothetical_embedding
            for shift in query_emb.shift_embeddings
        ]
        
        doc_deltas = [
            shift - doc_emb.original_embedding
            for shift in doc_emb.shift_embeddings
        ]
        
        # 3. Direction alignment scores
        direction_scores = []
        magnitude_scores = []
        
        for q_delta, d_delta in zip(query_deltas, doc_deltas):
            q_mag = np.linalg.norm(q_delta)
            d_mag = np.linalg.norm(d_delta)
            
            if q_mag < 1e-6 or d_mag < 1e-6:
                continue
            
            # Direction: normalized delta comparison
            q_dir = q_delta / q_mag
            d_dir = d_delta / d_mag
            dir_sim = self._cosine_similarity(q_dir, d_dir)
            direction_scores.append(dir_sim)
            
            # Magnitude: how similar are the movement magnitudes?
            mag_ratio = min(q_mag, d_mag) / max(q_mag, d_mag)
            magnitude_scores.append(mag_ratio)
        
        # 4. Aggregate scores
        direction_score = np.mean(direction_scores) if direction_scores else 0.0
        magnitude_score = np.mean(magnitude_scores) if magnitude_scores else 0.0
        
        # 5. Combined score (weighted)
        # Position: how close they are (HyDE baseline)
        # Direction: do they transform the same way? (structural similarity)
        # Magnitude: do they shift by similar amounts? (consistency check)
        combined_score = (
            0.30 * position_score +    # 30% HyDE baseline
            0.50 * direction_score +   # 50% directional fingerprint (KEY!)
            0.20 * magnitude_score     # 20% magnitude consistency
        )
        
        return {
            'position': position_score,
            'direction': direction_score,
            'magnitude': magnitude_score,
            'combined': combined_score
        }
    
    def search(self, query: str, top_k: int = 5, method: str = 'combined') -> list[tuple[str, float, dict]]:
        """
        Search for documents using HyDE + Directional Differential Fingerprinting.
        
        Args:
            query: The search query text
            top_k: Number of top results to return
            method: Scoring method - 'position', 'direction', 'magnitude', or 'combined'
        
        Returns:
            List of (doc_name, score, score_breakdown) tuples
        """
        # Generate query embeddings using HyDE + shifts
        print("Processing query using HyDE + Differential Fingerprinting...")
        query_emb = QueryEmbeddings(query, self.hyde_generator, self.shifter)
        
        # Compute differential scores for each document
        print(f"\nComputing differential scores...")
        results = []
        for doc_emb in self.documents:
            scores = self._directional_differential_score(query_emb, doc_emb)
            
            # Select which score to use for ranking
            final_score = scores[method]
            
            results.append((doc_emb.doc_path.name, final_score, scores))
            
            # Print detailed breakdown
            print(f"{doc_emb.doc_path.name}: "
                  f"pos={scores['position']:.4f} "
                  f"dir={scores['direction']:.4f} "
                  f"mag={scores['magnitude']:.4f} "
                  f"→ {method}={final_score:.4f}")
        
        # Sort by selected scoring method
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize the RAG system using HyDE + Differential Fingerprinting
    rag = HyDEDifferentialRAG(docs_dir="docs")
    
    query = "How do I implement secure token validation in an API?"
    
    # Run the search with different methods
    for method in ['position', 'direction', 'magnitude', 'combined']:
        print("\n" + "=" * 80)
        print(f"HyDE + DIFFERENTIAL FINGERPRINTING ({method.upper()} SCORING)")
        print("=" * 80)
        
        results = rag.search(query, top_k=5, method=method)
        
        print("\n" + "=" * 80)
        print(f"TOP 5 RESULTS (ranked by {method}):")
        print("=" * 80)
        for rank, (doc_name, score, breakdown) in enumerate(results, 1):
            print(f"{rank}. {doc_name}: {score:.4f}")
            print(f"   └─ position={breakdown['position']:.4f}, "
                  f"direction={breakdown['direction']:.4f}, "
                  f"magnitude={breakdown['magnitude']:.4f}")
        
        print()