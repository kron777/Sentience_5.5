import os
import json
import textwrap
from openai import OpenAI

# --- CONFIGURATION ---
# Options: "OPENAI" or "GROQ"
PROVIDER = "OPENAI" 

if PROVIDER == "OPENAI":
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = None  # Uses default OpenAI path
    ANALYZER_MODEL = "gpt-4o"       # High intelligence for analysis
    DEFAULT_NODE_MODEL = "gpt-4o-mini" # Fast/Cheap for nodes
elif PROVIDER == "GROQ":
    api_key = os.getenv("GROQ_API_KEY")
    base_url = "https://api.groq.com/openai/v1"
    ANALYZER_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_NODE_MODEL = "llama3-8b-8192"

# Initialize Client
if not api_key:
    print(f"Error: {PROVIDER}_API_KEY not set.")
    exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)

def analyze_node_functions(node_code: str, description: str = "") -> list:
    prompt = f"""
    You are an expert cognitive architect.
    Analyze this node code and extract the specific cognitive functions it must perform.
    Return ONLY valid JSON: {{"functions": ["func1", "func2"]}}
    
    CODE:
    {node_code}
    
    DESC: {description}
    """
    response = client.chat.completions.create(
        model=ANALYZER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["functions"]

def shape_and_save_node(node_id: str, node_code: str, description: str = ""):
    print(f"\n[!] Analyzing Node: {node_id} using {PROVIDER}...")
    
    functions = analyze_node_functions(node_code, description)
    
    # Logic: Use the bigger model for complex nodes
    model = ANALYZER_MODEL if len(functions) > 5 else DEFAULT_NODE_MODEL
    
    system_prompt = f"""You are Node {node_id}. 
Functions: {', '.join(functions)}.
Output MUST be JSON: {{"response": "...", "confidence": 0.XX}}"""

    config = {
        "node_id": node_id,
        "provider": PROVIDER,
        "model": model,
        "system_prompt": system_prompt,
        "functions": functions
    }

    os.makedirs("shaped_nodes", exist_ok=True)
    with open(f"shaped_nodes/{node_id}.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"[+] Success! Saved to shaped_nodes/{node_id}.json")

if __name__ == "__main__":
    print(f"--- Sentience Node Shaper ({PROVIDER} Mode) ---")
    print("Paste node code below. Type 'END' on a new line to finish:")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END": break
            lines.append(line)
        except EOFError:
            break
    
    code = "\n".join(lines)
    if code.strip():
        nid = input("Enter Node ID (e.g. Decision_Making_Node): ").strip()
        desc = input("Description (optional): ").strip()
        shape_and_save_node(nid, code, desc)
