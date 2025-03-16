import os
import json
import csv
import anthropic
from typing import Dict, List, Optional, Any, Tuple
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    user_query: str = Field(description="Original question asked by the user")
    proposed_answer: str = Field(description="Answer proposed by the human agent")
    grammar_fixed_answer: Optional[str] = Field(None, description="Answer with grammar fixed")
    adequacy_assessment: Optional[Dict] = Field(None, description="Assessment of answer adequacy")
    format_assessment: Optional[Dict] = Field(None, description="Assessment of answer formatting")
    final_answer: Optional[str] = Field(None, description="Final processed answer")

def get_claude_client(api_key: Optional[str] = None):
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")
    return anthropic.Anthropic(api_key=api_key)

def load_formatting_examples_from_csv() -> List[Dict]:
    csv_path = "faff_formatting_examples.csv"
    examples = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            change_descriptions = []
            for i in range(1, 5):
                key = f"Change Description {i}"
                if key in row and row[key].strip():
                    change_descriptions.append(row[key].strip())
            
            explanation = "\n".join([f"• {desc}" for desc in change_descriptions])
            
            example = {
                "task": row["Task"],
                "bad_format": row["Bad Formatting"],
                "good_format": row["Good Formatting"],
                "explanation": explanation,
            }
            examples.append(example)
    
    return examples

# Node 1: Fix grammar with minimal changes
def fix_grammar(state: AgentState, client=None) -> Dict:
    if not client:
        client = get_claude_client()
    
    prompt = f"""I need you to fix ONLY critical grammar issues in the following answer.
Make the absolute minimum changes necessary - only fix clear grammatical errors.
Do not:
- Change word choice unless absolutely necessary for grammar
- Alter sentence structure
- Modify punctuation unless it's grammatically incorrect
- Change the style, tone, formality level, or voice
- Add or remove information

The text should read almost identically to the original, just with grammar errors fixed.
Return ONLY the corrected text with no additional explanations.

Original answer: {state.proposed_answer}"""
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        temperature=0.0,
        system="You are a conservative grammar editor. Your job is to make the absolute minimum changes necessary to fix only clear grammatical errors. Preserve the author's original words, style and voice completely.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    grammar_fixed_answer = message.content[0].text
    return {"grammar_fixed_answer": grammar_fixed_answer}

def check_adequacy(state: AgentState, client=None) -> Dict:
    if not client:
        client = get_claude_client()
    
    grammar_fixed = state.grammar_fixed_answer or state.proposed_answer
    
    prompt = f"""Evaluate if the proposed answer adequately and explicitly addresses the user's query.

User Query: {state.user_query}

Proposed Answer: {grammar_fixed}

Your task:
1. Identify if the answer fully addresses all aspects of the user's query
2. Check if any important information is missing
3. Note if the answer contains irrelevant information
4. Suggest specific improvements if needed

Return your evaluation as JSON with these fields:
- "adequately_addressed": boolean
- "missing_aspects": list of strings (empty if none)
- "suggestions": list of specific improvements (empty if none)
- "improved_answer": the answer with your suggested improvements incorporated (if any, otherwise return the original)
"""
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        temperature=0.0,
        system="You are an expert at evaluating customer service responses. Be thorough but practical in your assessment. Only suggest changes when truly necessary to address the user's query.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract JSON from response
    response_text = message.content[0].text
    try:
        # Handle if Claude wraps the JSON in code blocks
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_content = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_content = response_text.strip()
            
        adequacy_assessment = json.loads(json_content)
    except json.JSONDecodeError:
        # Fallback if parsing fails
        adequacy_assessment = {
            "adequately_addressed": False,
            "missing_aspects": ["Unable to parse assessment"],
            "suggestions": ["Error processing assessment"],
            "improved_answer": state.grammar_fixed_answer or state.proposed_answer
        }
    
    return {"adequacy_assessment": adequacy_assessment}

def improve_formatting(state: AgentState, formatting_examples: List[Dict], client=None) -> Dict:
    if not client:
        client = get_claude_client()
    
    current_answer = state.adequacy_assessment.get("improved_answer", "") if state.adequacy_assessment else ""
    if not current_answer:
        current_answer = state.grammar_fixed_answer or state.proposed_answer
 
    examples_to_use = formatting_examples[:5]  # Limit to 5 examples to avoid token issues
    
    examples_text = ""
    for i, example in enumerate(examples_to_use):
        examples_text += f"\nExample {i+1}:\n"
        examples_text += f"User Query: {example['task']}\n"
        examples_text += f"Bad format: {example['bad_format']}\n"
        examples_text += f"Good format: {example['good_format']}\n"
        examples_text += f"Changes made: {example['explanation']}\n"
        
    prompt = f"""Improve the formatting of this customer service answer to make it more readable and user-friendly.
Keep the content largely the same, but apply formatting best practices based on these examples:

{examples_text}

Current Task: {state.user_query}
Current Answer:
{current_answer}

Return your assessment as JSON with these fields:
- "formatting_issues": list of formatting issues identified
- "improvements_made": list of improvements you made
- "improved_answer": the answer with better formatting
"""
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        temperature=0.0,
        system="You are an expert at formatting customer service responses for maximum readability and clarity. Apply the patterns from good examples while preserving the meaning and content of the answer.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract JSON from response
    response_text = message.content[0].text
    try:
        # Handle if Claude wraps the JSON in code blocks
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_content = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_content = response_text.strip()
            
        format_assessment = json.loads(json_content)
    except json.JSONDecodeError:
        # Fallback if parsing fails
        format_assessment = {
            "formatting_issues": ["Unable to parse assessment"],
            "improvements_made": ["Error processing formatting"],
            "improved_answer": current_answer
        }
    
    return {"format_assessment": format_assessment}

# Node 4: Create final answer with summary of changes
def create_final_answer(state: AgentState, client=None) -> Dict:
    if not client:
        client = get_claude_client()
    
    # Get the formatted answer
    formatted_answer = state.format_assessment.get("improved_answer", "") if state.format_assessment else ""
    if not formatted_answer:
        # Fallback chain
        adequacy_answer = state.adequacy_assessment.get("improved_answer", "") if state.adequacy_assessment else ""
        formatted_answer = adequacy_answer or state.grammar_fixed_answer or state.proposed_answer
    
    # Compile a summary of changes
    adequacy_changes = []
    if state.adequacy_assessment and not state.adequacy_assessment.get("adequately_addressed", True):
        adequacy_changes = state.adequacy_assessment.get("suggestions", [])
    
    formatting_changes = []
    if state.format_assessment:
        formatting_changes = state.format_assessment.get("improvements_made", [])
    
    # Create a final answer object
    final_answer = formatted_answer
    
    # Create a summary object for internal use/reporting
    changes_summary = {
        "grammar_changed": state.grammar_fixed_answer != state.proposed_answer if state.grammar_fixed_answer else False,
        "adequacy_issues": adequacy_changes,
        "formatting_improvements": formatting_changes
    }
    
    return {"final_answer": final_answer, "changes_summary": changes_summary}

# Build the LangGraph workflow
def build_answer_quality_graph(formatting_examples: List[Dict]) -> StateGraph:
    # Create Claude client to reuse
    client = get_claude_client()
    
    # Create the graph with our state
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fix_grammar", lambda state: fix_grammar(state, client))
    workflow.add_node("check_adequacy", lambda state: check_adequacy(state, client))
    workflow.add_node("improve_formatting", lambda state: improve_formatting(state, formatting_examples, client))
    workflow.add_node("create_final", lambda state: create_final_answer(state, client))
    
    # Connect the nodes in sequence
    workflow.add_edge("fix_grammar", "check_adequacy")
    workflow.add_edge("check_adequacy", "improve_formatting")
    workflow.add_edge("improve_formatting", "create_final")
    workflow.add_edge("create_final", END)
    
    # Set the entry point
    workflow.set_entry_point("fix_grammar")
    
    # Compile the graph
    return workflow.compile()

# Main function to process an answer
def process_answer(user_query: str, proposed_answer: str, formatting_examples: List[Dict] = None, csv_path: str = None) -> Dict:
    """
    Process an agent's proposed answer to ensure quality.
    
    Args:
        user_query: The original question from the user
        proposed_answer: The agent's proposed answer
        formatting_examples: Optional list of pre-loaded formatting examples
        csv_path: Path to CSV file with formatting examples (used if formatting_examples not provided)
        
    Returns:
        Dict with final answer and analysis
    """
    # Load examples from CSV if not provided directly
    if formatting_examples is None:
        if csv_path is None:
            raise ValueError("Either formatting_examples or csv_path must be provided")
        formatting_examples = load_formatting_examples_from_csv(csv_path)
    
    # Initialize the state
    initial_state = AgentState(
        user_query=user_query,
        proposed_answer=proposed_answer
    )
    
    # Build and run the graph
    graph = build_answer_quality_graph(formatting_examples)
    result = graph.invoke(initial_state)
    
    return {
        "original_answer": proposed_answer,
        "final_answer": result.final_answer,
        "analysis": {
            "grammar_fixed": result.grammar_fixed_answer != proposed_answer if result.grammar_fixed_answer else False,
            "adequacy_assessment": result.adequacy_assessment,
            "format_assessment": result.format_assessment
        }
    }