# app/conversation_agent.py
"""
Interim agent-like capabilities using existing Google Gemini setup
Implements conversation memory and function calling patterns
without requiring the full Google Agent SDK
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.rag_pipeline import build_vector_store


@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    timestamp: datetime
    user_message: str
    ai_response: str
    tools_used: List[str]
    sources: List[str]
    reasoning: Optional[str] = None


@dataclass
class AgentFunction:
    """Represents an available function/tool for the agent"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class ConversationMemory:
    """Simple conversation memory using SQLite"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize conversation database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    timestamp TIMESTAMP,
                    user_message TEXT,
                    ai_response TEXT,
                    tools_used TEXT,
                    sources TEXT,
                    reasoning TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
    
    def create_conversation(self, conversation_id: str):
        """Create a new conversation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, datetime.now(), datetime.now())
            )
    
    def add_turn(self, conversation_id: str, turn: ConversationTurn):
        """Add a conversation turn"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversation_turns 
                (conversation_id, timestamp, user_message, ai_response, tools_used, sources, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                turn.timestamp,
                turn.user_message,
                turn.ai_response,
                json.dumps(turn.tools_used),
                json.dumps(turn.sources),
                turn.reasoning
            ))
            
            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.now(), conversation_id)
            )
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[ConversationTurn]:
        """Get recent conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, user_message, ai_response, tools_used, sources, reasoning
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, limit))
            
            turns = []
            for row in cursor.fetchall():
                turns.append(ConversationTurn(
                    timestamp=datetime.fromisoformat(row[0]),
                    user_message=row[1],
                    ai_response=row[2],
                    tools_used=json.loads(row[3]),
                    sources=json.loads(row[4]),
                    reasoning=row[5]
                ))
            
            return list(reversed(turns))  # Return in chronological order


class SimpleConversationAgent:
    """
    Agent-like capabilities using existing Google Gemini
    Implements conversation memory and function calling patterns
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=settings.google_api_key,
            temperature=0.1
        )
        
        self.memory = ConversationMemory()
        self.vector_store = build_vector_store()
        self.functions = self._register_functions()
        
    def _register_functions(self) -> Dict[str, AgentFunction]:
        """Register available functions"""
        functions = {}
        
        # Codebase search function
        functions["search_codebase"] = AgentFunction(
            name="search_codebase",
            description="Search the codebase for relevant code, documentation, or patterns",
            parameters={
                "query": {"type": "string", "description": "Search query for the codebase"},
                "max_results": {"type": "integer", "description": "Maximum results to return", "default": 8}
            },
            function=self._search_codebase
        )
        
        # Project analysis function
        functions["analyze_project_structure"] = AgentFunction(
            name="analyze_project_structure", 
            description="Analyze the structure and architecture of a specific project",
            parameters={
                "project_name": {"type": "string", "description": "Name of the project to analyze"},
                "analysis_type": {"type": "string", "description": "Type of analysis: 'structure', 'dependencies', 'patterns'", "default": "structure"}
            },
            function=self._analyze_project_structure
        )
        
        # Code explanation function
        functions["explain_code_pattern"] = AgentFunction(
            name="explain_code_pattern",
            description="Explain specific code patterns, implementations, or architectural decisions",
            parameters={
                "pattern_query": {"type": "string", "description": "Description of the code pattern to explain"},
                "include_examples": {"type": "boolean", "description": "Include code examples", "default": True}
            },
            function=self._explain_code_pattern
        )
        
        return functions
    
    async def _search_codebase(self, query: str, max_results: int = 8) -> Dict[str, Any]:
        """Search codebase using existing RAG pipeline"""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": max_results})
        docs = await retriever.ainvoke(query)
        
        results = {
            "documents_found": len(docs),
            "sources": [doc.metadata.get('source', 'unknown') for doc in docs],
            "projects": list(set(doc.metadata.get('project', 'unknown') for doc in docs)),
            "content_summary": []
        }
        
        for doc in docs[:3]:  # Include content from top 3 results
            results["content_summary"].append({
                "file": doc.metadata.get('source', 'unknown'),
                "project": doc.metadata.get('project', 'unknown'),
                "preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        return results
    
    async def _analyze_project_structure(self, project_name: str, analysis_type: str = "structure") -> Dict[str, Any]:
        """Analyze project structure"""
        # Use your existing project discovery logic
        query = f"project:{project_name} {analysis_type}"
        search_results = await self._search_codebase(query, max_results=15)
        
        # Enhanced analysis based on file types and patterns
        analysis = {
            "project": project_name,
            "analysis_type": analysis_type,
            "files_found": search_results["documents_found"],
            "architecture_insights": [],
            "key_components": [],
            "recommendations": []
        }
        
        # Add structure-specific analysis
        if analysis_type == "structure":
            analysis["architecture_insights"] = [
                "Analyzed project file organization",
                "Identified key architectural patterns",
                "Mapped component relationships"
            ]
        
        return analysis
    
    async def _explain_code_pattern(self, pattern_query: str, include_examples: bool = True) -> Dict[str, Any]:
        """Explain code patterns with examples"""
        search_results = await self._search_codebase(f"pattern implementation {pattern_query}")
        
        explanation = {
            "pattern": pattern_query,
            "explanation": f"Analysis of {pattern_query} based on codebase",
            "examples_found": len(search_results.get("content_summary", [])),
            "implementation_files": search_results.get("sources", [])
        }
        
        if include_examples and search_results.get("content_summary"):
            explanation["code_examples"] = search_results["content_summary"]
        
        return explanation
    
    def _create_function_calling_prompt(self, user_query: str, conversation_history: List[ConversationTurn]) -> str:
        """Create a prompt that enables function calling"""
        
        # Build conversation context
        context = ""
        if conversation_history:
            context = "Previous conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                context += f"User: {turn.user_message}\n"
                context += f"Assistant: {turn.ai_response[:200]}...\n\n"
        
        # Build function descriptions
        function_descriptions = ""
        for func_name, func_def in self.functions.items():
            function_descriptions += f"- {func_name}: {func_def.description}\n"
            function_descriptions += f"  Parameters: {json.dumps(func_def.parameters, indent=2)}\n\n"
        
        prompt = f"""You are an intelligent code analysis assistant with access to function calling capabilities.

{context}
Available Functions:
{function_descriptions}

User Query: {user_query}

Instructions:
1. Analyze the user query to determine if you need to call any functions
2. If functions are needed, specify which functions to call with their parameters using this format:
   FUNCTION_CALL: function_name(param1="value1", param2="value2")
   
3. You can make multiple function calls if needed
4. After getting function results, provide a comprehensive answer

Think step by step:
1. What information do I need to answer this query?
2. Which functions can provide that information?
3. What parameters should I use?

Respond with either:
- FUNCTION_CALL: function_name(parameters) if you need to gather information
- A direct answer if you have sufficient information from conversation history

Your response:"""
        
        return prompt
    
    async def _execute_function_calls(self, response: str) -> tuple[str, List[str], Dict[str, Any]]:
        """Parse and execute function calls from LLM response"""
        tools_used = []
        function_results = {}
        
        # Simple function call parsing (could be enhanced with proper parsing)
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('FUNCTION_CALL:'):
                # Extract function call
                call_part = line.strip().replace('FUNCTION_CALL:', '').strip()
                
                # Parse function name and parameters (simplified)
                if '(' in call_part and call_part.endswith(')'):
                    func_name = call_part.split('(')[0].strip()
                    params_str = call_part[call_part.index('(')+1:-1]
                    
                    if func_name in self.functions:
                        try:
                            # Simple parameter parsing (could use ast.literal_eval for complex cases)
                            params = {}
                            if params_str:
                                for param in params_str.split(','):
                                    if '=' in param:
                                        key, value = param.split('=', 1)
                                        key = key.strip()
                                        value = value.strip().strip('"\'')
                                        params[key] = value
                            
                            # Execute function
                            result = await self.functions[func_name].function(**params)
                            function_results[func_name] = result
                            tools_used.append(func_name)
                            
                        except Exception as e:
                            function_results[func_name] = {"error": str(e)}
        
        return response, tools_used, function_results
    
    async def chat(
        self, 
        query: str, 
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main chat interface with agent capabilities"""
        
        # Get conversation history
        conversation_history = []
        if conversation_id:
            conversation_history = self.memory.get_conversation_history(conversation_id)
        else:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.memory.create_conversation(conversation_id)
        
        # Create function calling prompt
        prompt = self._create_function_calling_prompt(query, conversation_history)
        
        # Get initial LLM response (may contain function calls)
        initial_response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        initial_text = initial_response.content
        
        # Execute any function calls
        response_text, tools_used, function_results = await self._execute_function_calls(initial_text)
        
        # If functions were called, generate final response with results
        final_response = initial_text
        sources = []
        
        if function_results:
            # Create enhanced prompt with function results
            results_context = "Function Results:\n"
            for func_name, result in function_results.items():
                results_context += f"{func_name}: {json.dumps(result, indent=2)}\n"
                
                # Extract sources from search results
                if isinstance(result, dict) and 'sources' in result:
                    sources.extend(result['sources'])
            
            enhanced_prompt = f"""Based on the function results below, answer the user's question.

User Query: {query}

{results_context}

**Audience Detection:**
- If question uses technical terms (class, method, implementation, DTO, API) → provide comprehensive technical analysis
- If question is general (what is, how does, explain) → use plain business language

**Guidelines:**
1. Start with direct answer (1-2 sentences)
2. Match depth to the question - technical questions get technical answers
3. Use step-by-step flows for processes
4. Cite source files at the end
5. SKIP: testing, 'what's missing', config details (unless asked)

Response:"""
            
            final_llm_response = await self.llm.ainvoke([HumanMessage(content=enhanced_prompt)])
            final_response = final_llm_response.content
        
        # Save conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=query,
            ai_response=final_response,
            tools_used=tools_used,
            sources=list(set(sources)),
            reasoning=f"Used functions: {', '.join(tools_used)}" if tools_used else None
        )
        
        self.memory.add_turn(conversation_id, turn)
        
        return {
            "answer": final_response,
            "conversation_id": conversation_id,
            "tools_used": tools_used,
            "sources": list(set(sources)),
            "function_results": function_results if function_results else None,
            "reasoning": turn.reasoning
        }


# FastAPI integration
async def create_conversation_agent() -> SimpleConversationAgent:
    """Factory for creating conversation agent"""
    return SimpleConversationAgent()


# Usage example:
"""
# Add to your main.py:

@app.post("/conversation")
async def conversation_chat(request: dict):
    agent = await create_conversation_agent()
    response = await agent.chat(
        query=request["query"],
        conversation_id=request.get("conversation_id")
    )
    return response
"""