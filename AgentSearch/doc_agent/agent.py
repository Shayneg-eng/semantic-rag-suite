"""
Agent Orchestrator: Main loop managing LLM interactions and tool execution
"""

import json
import os
import sys
from typing import Dict, Any, List
from openai import OpenAI

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_ITERATIONS,
    HISTORY_LIMIT,
    SYSTEM_PROMPT,
)
from utils.document_manager import get_document_manager
from tools.agent_tools import TOOL_DEFINITIONS, execute_tool


class DocumentAnalysisAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.doc_manager = get_document_manager()
        self.conversation_history = []
        
    def initialize(self):
        """Initialize the agent and document manager"""
        print("Initializing Document Analysis Agent...")
        self.doc_manager.initialize()
        print("Agent ready!")
        
    def run_query(self, user_query: str) -> Dict[str, Any]:
        """
        Run a user query through the agent loop
        """
        # Reset conversation history for new query
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add user query
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")
        print(f"{'='*60}\n")
        
        iteration = 0
        final_answer = None
        
        while iteration < MAX_ITERATIONS:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
            
            try:
                # Call LLM with function calling
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=self.conversation_history,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                
                # Check if LLM wants to call tools
                if message.tool_calls:
                    # Add assistant message to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })
                    
                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        print(f"🔧 Calling tool: {tool_name}")
                        print(f"   Arguments: {json.dumps(arguments, indent=2)}")
                        
                        # Execute tool
                        result = execute_tool(tool_name, arguments)
                        
                        print(f"   Result: {json.dumps(result, indent=2)[:200]}...")
                        
                        # Add tool result to history
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })
                        
                        # Check if final answer submitted
                        if tool_name == "submit_answer" and result.get("success"):
                            final_answer = result
                            print("\n✅ Final answer submitted!")
                            break
                    
                    # If final answer found, exit loop
                    if final_answer:
                        break
                    
                else:
                    # LLM responded with text (no tool calls)
                    print(f"💬 LLM Response: {message.content}")
                    final_answer = {
                        "success": True,
                        "answer": message.content,
                        "sources": [],
                        "confidence": "unknown"
                    }
                    break
                
                # Trim conversation history to prevent context overflow
                self._trim_history()
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                return {
                    "success": False,
                    "error": f"Agent error: {str(e)}",
                    "iteration": iteration
                }
        
        # Check if max iterations reached
        if iteration >= MAX_ITERATIONS and not final_answer:
            print("\n⚠️ Maximum iterations reached without final answer")
            return {
                "success": False,
                "error": "Maximum iterations reached",
                "iteration": iteration
            }
        
        print(f"\n{'='*60}")
        print(f"COMPLETED IN {iteration} ITERATIONS")
        print(f"{'='*60}\n")
        
        return final_answer
    
    def _trim_history(self):
        """Trim conversation history to prevent context overflow"""
        # Keep system prompt and recent history
        if len(self.conversation_history) > HISTORY_LIMIT + 1:
            # Keep system prompt (index 0) and last HISTORY_LIMIT messages
            self.conversation_history = [
                self.conversation_history[0]
            ] + self.conversation_history[-(HISTORY_LIMIT):]
    
    def interactive_mode(self):
        """Run agent in interactive mode"""
        print("\n" + "="*60)
        print("Document Analysis Agent - Interactive Mode")
        print("="*60)
        print("\nType your questions, or 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("\n📝 Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Run query
                result = self.run_query(user_input)
                
                # Display result
                print("\n" + "="*60)
                print("FINAL ANSWER")
                print("="*60)
                
                if result.get("success"):
                    print(f"\n{result.get('answer', 'No answer provided')}\n")
                    
                    if result.get('sources'):
                        print(f"Sources: {', '.join(result['sources'])}")
                    
                    if result.get('confidence'):
                        print(f"Confidence: {result['confidence']}")
                    
                    if result.get('excerpts'):
                        print("\nSupporting excerpts:")
                        for i, excerpt in enumerate(result['excerpts'], 1):
                            print(f"{i}. {excerpt}")
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}\n")
                
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    # Create and initialize agent
    agent = DocumentAnalysisAgent()
    agent.initialize()
    
    # Run in interactive mode
    agent.interactive_mode()


if __name__ == "__main__":
    main()
