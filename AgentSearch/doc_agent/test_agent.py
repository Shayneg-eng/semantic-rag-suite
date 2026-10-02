"""
Test script for Document Analysis Agent
Demonstrates basic functionality and tool usage
"""

from agent import DocumentAnalysisAgent


def test_basic_functionality():
    """Test basic agent functionality"""
    print("="*60)
    print("Testing Document Analysis Agent")
    print("="*60)
    
    # Initialize agent
    print("\n1. Initializing agent...")
    agent = DocumentAnalysisAgent()
    agent.initialize()
    
    # Test queries
    test_queries = [
        "What types of documents are in the collection?",
        "Find documents mentioning confidentiality or non-disclosure",
        "What dates are mentioned in the documents?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}: {query}")
        print(f"{'='*60}")
        
        try:
            result = agent.run_query(query)
            
            if result.get("success"):
                print("\n✅ SUCCESS")
                print(f"Answer: {result.get('answer', 'N/A')[:300]}...")
                print(f"Sources: {result.get('sources', [])}")
                print(f"Confidence: {result.get('confidence', 'N/A')}")
            else:
                print(f"\n❌ FAILED: {result.get('error')}")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
        
        print()
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)


def test_individual_tools():
    """Test individual tools without full agent loop"""
    print("\n" + "="*60)
    print("Testing Individual Tools")
    print("="*60)
    
    from tools.agent_tools import execute_tool
    from utils.document_manager import get_document_manager
    
    # Initialize document manager
    doc_manager = get_document_manager()
    doc_manager.initialize()
    
    # Test 1: List documents
    print("\n1. Testing list_docs...")
    result = execute_tool("list_docs", {"scope": "all"})
    print(f"   Found {result.get('total_documents', 0)} documents")
    if result.get('documents'):
        print(f"   First document: {result['documents'][0]['doc_id']}")
    
    # Test 2: Simple search
    print("\n2. Testing search...")
    result = execute_tool("search", {
        "query": "confidential",
        "mode": "simple",
        "scope": "all"
    })
    print(f"   Total matches: {result.get('total_matches', 0)}")
    
    # Test 3: Get metadata
    print("\n3. Testing get_metadata...")
    result = execute_tool("get_metadata", {"scope": "all"})
    if result.get('aggregate'):
        agg = result['aggregate']
        print(f"   Total words: {agg.get('total_words', 0):,}")
        print(f"   Total documents: {result.get('total_documents', 0)}")
        print(f"   Average words per doc: {agg.get('avg_words_per_doc', 0):,}")
    
    # Test 4: Extract entities
    print("\n4. Testing extract_entities...")
    result = execute_tool("extract_entities", {
        "scope": "all",
        "entity_types": ["dates", "money"]
    })
    print(f"   Processed {result.get('total_documents', 0)} documents")
    if result.get('results') and len(result['results']) > 0:
        first = result['results'][0]
        if 'dates' in first:
            print(f"   Dates found in first doc: {first['dates'].get('count', 0)}")
        if 'money' in first:
            print(f"   Money amounts in first doc: {first['money'].get('count', 0)}")
    
    print("\n✅ Individual tool tests completed!")


if __name__ == "__main__":
    import sys
    
    print("\nDocument Analysis Agent - Test Suite")
    print("=" * 60)
    print("\nOptions:")
    print("1. Test individual tools (quick)")
    print("2. Test full agent queries (with LLM)")
    print("3. Run both tests")
    print()
    
    choice = input("Enter choice (1/2/3) or press Enter for option 1: ").strip()
    
    if not choice:
        choice = "1"
    
    if choice in ["1", "3"]:
        test_individual_tools()
    
    if choice in ["2", "3"]:
        test_basic_functionality()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nTo use the interactive agent, run:")
    print("  python agent.py")
    print()
