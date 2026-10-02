import os
# ============================================================================
# CONFIGURATION & TUNABLE PARAMETERS
# ============================================================================

# API and Directories
POE_API_KEY = os.getenv("POE_API_KEY", "")
SOURCE_DOCS_DIR = "source_docs"
RESULTS_DIR = "shiftdim3_results"
MAX_DOCUMENTS = None  # Set to a number (e.g., 5) to limit documents, or None for all

# LLM Configuration
LLM_MODEL = "llama-3.1-8b-cs"

# Embedding Configuration
EMBEDDING_MODEL = "nomic-embed-text"

# Shift Configuration - Replacement Words (empty strings remove the word entirely)
ADJECTIVE_REPLACEMENT_WORD = "adjective"         # Empty string removes adjectives

# Shift Prompts
PREPOSITION_EXTRACTION_PROMPT = """Extract ONLY the prepositions (words showing spatial, temporal, or logical relationships) from this text. Include each preposition exactly once, in lowercase. Do not include other parts of speech.

Common prepositions: in, on, at, by, from, to, with, for, of, about, before, after, between, through, during, under, over, above, below, near, inside, outside, within, without, among, across, along, around, beside, beyond, down, up, out, against, toward, until, since, except, throughout

Example input: "The database runs inside the datacenter and performs queries across multiple servers throughout the day"
Example output: inside, across, throughout

Text to extract from:
{text}"""

ADVERB_EXTRACTION_PROMPT = """Extract ONLY the adverbs (words modifying verbs, adjectives, or other adverbs, typically ending in -ly or indicating manner, time, frequency, or degree) from this text. Include each adverb exactly once, in lowercase. Do not include adjectives.

Common adverbs: quickly, slowly, carefully, efficiently, effectively, reliably, significantly, constantly, continuously, frequently, rarely, always, never, sometimes, usually, particularly, especially, very, quite, extremely, fairly, relatively, completely, partly, fully, absolutely, clearly, obviously, certainly, probably, possibly, perhaps, indeed, certainly, carefully, thoroughly, broadly, strictly

Example input: "The system responds quickly and efficiently handles requests very reliably"
Example output: quickly, efficiently, very, reliably

Text to extract from:
{text}"""

ADJECTIVE_EXTRACTION_PROMPT = """Extract ONLY the adjectives (words describing or modifying nouns, indicating qualities or characteristics) from this text. Include each adjective exactly once, in lowercase. Do not include adverbs ending in -ly or articles.

Common adjectives: fast, slow, complex, simple, large, small, high, low, strong, weak, efficient, reliable, stable, distributed, scalable, concurrent, sequential, synchronous, asynchronous, dynamic, static, fixed, variable, optimal, minimal, maximum, consistent, inconsistent, available, unavailable, predictable, unpredictable, secure, insecure, open, closed

Example input: "The fast and reliable server handles complex requests on distributed systems"
Example output: fast, reliable, complex, distributed

Text to extract from:
{text}"""

# Semantic shifts - focused on adjectives only
SHIFTS = {
    "remove_adjectives": "extract_adjectives"
}

# Query to expected document mapping
QUERY_MAPPINGS = {
    "What's the difference between synchronous and asynchronous replication in databases?": "DOC_8",
    "How does the token bucket algorithm work for rate limiting?": "DOC_9",
    "What are the main differences between Kafka and RabbitMQ?": "DOC_11",
    "Why should I use PKCE when implementing OAuth2?": "DOC_1",
    "What's the deal with composite indexes - does column order actually matter?": "DOC_2",
    "How does a circuit breaker pattern prevent cascading failures?": "DOC_10",
    "What's the difference between write-through and write-behind caching?": "DOC_4",
    "Can you explain what a saga pattern is and when I'd use it instead of two-phase commit?": "DOC_21",
    "What are the three states in a circuit breaker and how do they work?": "DOC_10",
    "How does Raft handle leader election when a leader fails?": "DOC_13",
    "What's the difference between Layer 4 and Layer 7 load balancing?": "DOC_7",
    "Why do I need correlation IDs in a microservices architecture?": "DOC_5",
    "What are the different OAuth2 grant types and when would I use each one?": "DOC_1",
    "How does consistent hashing help when adding new cache servers?": "DOC_24",
    "What's the difference between event sourcing and traditional database storage?": "DOC_17",
    "How do I handle cache invalidation - should I use TTL or event-based invalidation?": "DOC_4",
    "What does CQRS stand for and why would I separate my read and write models?": "DOC_17",
    "How does Kubernetes handle service discovery between pods?": "DOC_12",
    "What's the difference between optimistic and pessimistic concurrency control?": "DOC_21",
    "Why is exponential backoff with jitter better than just exponential backoff for retries?": "DOC_10"
}
