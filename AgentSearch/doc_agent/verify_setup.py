#!/usr/bin/env python3
"""
Setup verification script for Document Analysis Agent
Checks all prerequisites and helps diagnose issues
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    packages = {
        'openai': 'openai',
        'numpy': 'numpy',
        'ollama': 'ollama'
    }
    
    all_ok = True
    for package, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package} installed")
        except ImportError:
            print(f"  ❌ {package} NOT installed")
            all_ok = False
    
    if not all_ok:
        print("\n  To install missing packages:")
        print("  pip install -r requirements.txt")
    
    return all_ok

def check_ollama():
    """Check if Ollama is accessible"""
    print("\nChecking Ollama connection...")
    
    try:
        import ollama
        # Try to list models
        models = ollama.list()
        print("  ✅ Ollama is accessible")
        
        # Check for nomic-embed-text model
        model_names = [m['name'] for m in models.get('models', [])]
        if any('nomic-embed-text' in name for name in model_names):
            print("  ✅ nomic-embed-text model found")
            return True
        else:
            print("  ⚠️  nomic-embed-text model NOT found")
            print("  Run: ollama pull nomic-embed-text")
            return False
    
    except Exception as e:
        print(f"  ❌ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

def check_data_directory():
    """Check if data directory exists and has documents"""
    print("\nChecking data directory...")
    
    data_dir = "./data"
    if not os.path.exists(data_dir):
        # Try parent directory
        data_dir = "../data"
    
    if os.path.exists(data_dir):
        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if txt_files:
            print(f"  ✅ Found {len(txt_files)} .txt files in {data_dir}")
            return True
        else:
            print(f"  ⚠️  No .txt files found in {data_dir}")
            print("  Add some .txt documents to analyze")
            return False
    else:
        print(f"  ❌ Data directory not found: {data_dir}")
        print("  Create the directory and add .txt files")
        return False

def check_api_key():
    """Check if DeepSeek API key is configured"""
    print("\nChecking API configuration...")
    
    try:
        from settings import DEEPSEEK_API_KEY
        if DEEPSEEK_API_KEY and len(DEEPSEEK_API_KEY) > 10:
            print(f"  ✅ DeepSeek API key configured ({DEEPSEEK_API_KEY[:10]}...)")
            return True
        else:
            print("  ⚠️  DeepSeek API key may not be valid")
            return False
    except Exception as e:
        print(f"  ❌ Error loading config: {e}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("Document Analysis Agent - Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama),
        ("Data Directory", check_data_directory),
        ("API Key", check_api_key)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Error during {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print()
    
    if all_passed:
        print("🎉 All checks passed! You're ready to use the agent.")
        print("\nNext steps:")
        print("  1. Run tests: python test_agent.py")
        print("  2. Start agent: python agent.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Start Ollama: ollama serve")
        print("  • Pull model: ollama pull nomic-embed-text")
        print("  • Add documents to data/ directory")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
