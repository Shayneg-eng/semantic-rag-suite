import os
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Configuration
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'query_document_pairs.csv')
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'vector_db.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'query_embedding_mapper.pt')

EMBEDDING_DIM = 768  # nomic-embed-text dimension
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005  # Reduced learning rate
NUM_NEGATIVES = 32  # Reduced from 100 to avoid overfitting
MARGIN = 0.1  # Margin for contrastive loss
EARLY_STOPPING_PATIENCE = 8  # Stop if no improvement for 8 epochs
WEIGHT_DECAY = 1e-4  # L2 regularization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ContrastiveEmbeddingDataset(Dataset):
    """Dataset for contrastive learning with positive and negative samples."""
    
    def __init__(self, queries, vector_db):
        self.query_embeddings = []
        self.doc_embeddings = []  # Positive (correct) documents
        self.all_doc_embeddings = []
        self.all_doc_names = []
        self.query_to_correct_doc = {}  # Map query idx to correct doc name
        
        # Store all document embeddings
        self.all_doc_names = list(vector_db.keys())
        self.all_doc_embeddings = [np.array(vector_db[name]['embedding'], dtype=np.float32) 
                                   for name in self.all_doc_names]
        
        # Build dataset
        for idx, row in enumerate(queries):
            try:
                query_embedding = eval(row['query_embedding'])
                correct_doc = row['correct_document']
                
                if correct_doc in vector_db:
                    doc_embedding = vector_db[correct_doc]['embedding']
                    
                    self.query_embeddings.append(np.array(query_embedding, dtype=np.float32))
                    self.doc_embeddings.append(np.array(doc_embedding, dtype=np.float32))
                    self.query_to_correct_doc[len(self.query_embeddings) - 1] = correct_doc
            except:
                pass
        
        self.query_embeddings = np.array(self.query_embeddings)
        self.doc_embeddings = np.array(self.doc_embeddings)
    
    def __len__(self):
        return len(self.query_embeddings)
    
    def __getitem__(self, idx):
        query_emb = torch.tensor(self.query_embeddings[idx], dtype=torch.float32)
        positive_doc_emb = torch.tensor(self.doc_embeddings[idx], dtype=torch.float32)
        
        # Sample random negative documents
        negative_indices = random.sample(range(len(self.all_doc_embeddings)), NUM_NEGATIVES)
        negative_doc_embs = [torch.tensor(self.all_doc_embeddings[i], dtype=torch.float32) 
                             for i in negative_indices]
        
        return query_emb, positive_doc_emb, negative_doc_embs

class EmbeddingMapper(nn.Module):
    """Neural network that maps query embeddings to document embeddings."""
    
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super(EmbeddingMapper, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, query_embedding):
        return self.encoder(query_embedding)

def contrastive_loss(mapped_query, positive_doc, negative_docs, margin=MARGIN):
    """
    Contrastive loss: pull correct document close, push negatives away.
    Uses a combination of:
    1. Minimize distance to positive
    2. Maximize distance to negatives (with margin)
    """
    # Normalize vectors
    mapped_query_norm = torch.nn.functional.normalize(mapped_query, dim=1)
    positive_doc_norm = torch.nn.functional.normalize(positive_doc, dim=1)
    
    # Loss component 1: Pull positive closer (maximize similarity)
    positive_sim = torch.sum(mapped_query_norm * positive_doc_norm, dim=1)
    pull_loss = -positive_sim.mean()  # Negative because we want to maximize similarity
    
    # Loss component 2: Push negatives away
    push_loss = 0.0
    for negative_doc in negative_docs:
        negative_doc_norm = torch.nn.functional.normalize(negative_doc, dim=1)
        negative_sim = torch.sum(mapped_query_norm * negative_doc_norm, dim=1)
        
        # We want negative_sim to be less than positive_sim
        # Loss = max(0, margin + negative_sim - positive_sim)
        push_loss += torch.clamp(margin + negative_sim - positive_sim, min=0.0).mean()
    
    push_loss = push_loss / len(negative_docs)
    
    # Combine losses (weight them equally)
    total_loss = pull_loss + push_loss
    
    return total_loss

def train_epoch(model, train_loader, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for query_emb, positive_doc_emb, negative_doc_embs in train_loader:
        query_emb = query_emb.to(device)
        positive_doc_emb = positive_doc_emb.to(device)
        negative_doc_embs = [neg.to(device) for neg in negative_doc_embs]
        
        # Forward pass
        mapped_query_emb = model(query_emb)
        loss = contrastive_loss(mapped_query_emb, positive_doc_emb, negative_doc_embs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for query_emb, positive_doc_emb, negative_doc_embs in test_loader:
            query_emb = query_emb.to(device)
            positive_doc_emb = positive_doc_emb.to(device)
            negative_doc_embs = [neg.to(device) for neg in negative_doc_embs]
            
            mapped_query_emb = model(query_emb)
            loss = contrastive_loss(mapped_query_emb, positive_doc_emb, negative_doc_embs)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def main():
    """Train embedding mapper network with contrastive learning."""
    
    print("Training Query-Document Embedding Mapper (Contrastive Loss)")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    with open(VECTOR_DB_PATH, 'rb') as f:
        vector_db = pickle.load(f)
    
    queries = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    
    print(f"✅ Loaded {len(queries)} queries and {len(vector_db)} documents")
    
    # Create dataset
    print("Preparing dataset...")
    dataset = ContrastiveEmbeddingDataset(queries, vector_db)
    print(f"✅ Dataset size: {len(dataset)} query-document pairs")
    print(f"✅ Each query will have {NUM_NEGATIVES} negative samples")
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Train set: {len(train_dataset)}, Test set: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating model...")
    model = EmbeddingMapper(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    print(f"\nTraining for up to {EPOCHS} epochs (early stopping at {EARLY_STOPPING_PATIENCE})...")
    print("-" * 80)
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        test_loss = evaluate(model, test_loader)
        
        overfit_gap = train_loss - test_loss
        print(f"Epoch {epoch+1:2d}/{EPOCHS}: Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Gap: {overfit_gap:+.4f}")
        
        # Save best model and track early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"            ✅ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    print("-" * 80)
    print(f"\n✅ Training complete!")
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Best test loss: {best_test_loss:.4f}")
    
    # Test on a sample
    print(f"\nTesting on sample queries...")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        # Get a few random samples
        sample_indices = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))
        
        for i, idx in enumerate(sample_indices, 1):
            query_emb, positive_doc_emb, negative_doc_embs = test_dataset[idx]
            query_emb_device = query_emb.unsqueeze(0).to(device)
            
            # Predicted mapping
            mapped_query_emb = model(query_emb_device)
            
            # Compute similarities
            query_norm = torch.nn.functional.normalize(query_emb_device, dim=1)
            positive_norm = torch.nn.functional.normalize(positive_doc_emb.unsqueeze(0).to(device), dim=1)
            mapped_norm = torch.nn.functional.normalize(mapped_query_emb, dim=1)
            
            raw_sim = torch.sum(query_norm * positive_norm, dim=1).item()
            mapped_sim = torch.sum(mapped_norm * positive_norm, dim=1).item()
            
            print(f"\nSample {i}:")
            print(f"  Raw query-doc similarity:    {raw_sim:.4f}")
            print(f"  Mapped query-doc similarity: {mapped_sim:.4f}")
            print(f"  Improvement:                 {(mapped_sim - raw_sim):+.4f}")

if __name__ == '__main__':
    main()
