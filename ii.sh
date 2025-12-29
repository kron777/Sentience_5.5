#!/bin/bash
# ii.sh - Sentience 5.5 Installation and Initialization Script
# This script sets up the entire Sentience system:
# - Creates project directory
# - Installs dependencies
# - Installs Ollama if not present
# - Pulls LLM model
# - Creates all required Python files
# - Runs the Orchestrator

set -e  # Exit on error

PROJECT_DIR="$HOME/Desktop/Sentience_5.5"
echo "Setting up Sentience in $PROJECT_DIR"

# Create project directory
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Install Python dependencies (if pip available)
if command -v pip3 &> /dev/null; then
    pip3 install requests beautifulsoup4
else
    echo "pip3 not found. Install Python dependencies manually."
fi

# Install Ollama if not installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

# Pull LLM model (default: llama3.1:8b)
echo "Pulling LLM model..."
ollama pull llama3.1:8b

# Start Ollama server in background if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve &
    sleep 5  # Wait for server to start
fi

# Create all Python files
cat << EOF > Orchestrator.py
#!/usr/bin/env python3
"""
Orchestrator.py
Sentience 5.5 – Central coordinator with LLM intelligence
"""

import sys

from Memory_Node import MemoryNode
from Nonsense_Node import NonsenseNode
from Knowledge_Node import KnowledgeNode
from Reasoning_Node import ReasoningNode
from Conversational_Intelligence_Node import ConversationalIntelligenceNode
from Web_Crawler_Node import WebCrawlerNode
from Evolver_Node import EvolverNode
from LLM_Node import LLMNode

BANNER = r"""
███████╗███████╗███╗   ██╗████████╗██╗███████╗███╗   ██╗ ██████╗███████╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝████╗  ██║██╔════╝██╔════╝
███████╗█████╗  ██╔██╗ ██║   ██║   ██║█████╗  ██╔██╗ ██║██║     █████╗  
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  
███████║███████╗██║ ╚████║   ██║   ██║███████╗██║ ╚████║╚██████╗███████╗
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
"""

class Orchestrator:
    def __init__(self):
        self.memory = MemoryNode()
        self.nonsense = NonsenseNode()
        self.knowledge = KnowledgeNode()
        self.crawler = WebCrawlerNode(self.memory)
        self.evolver = EvolverNode(self.memory, self.knowledge, self.crawler)
        self.reasoning = ReasoningNode(self.knowledge, self.memory)
        self.llm = LLMNode()
        self.chat = ConversationalIntelligenceNode(
            self.memory, self.nonsense, self.knowledge, self.reasoning,
            self.crawler, self.evolver, self.llm
        )

        print(BANNER)
        print("Sentience online.")

    def run(self):
        while True:
            try:
                input_text = input(">> ").strip()
                if not input_text:
                    continue
                response = self.chat.respond(input_text)
                print(response)
            except KeyboardInterrupt:
                print("\\nShutting down.")
                sys.exit(0)

if __name__ == "__main__":
    Orchestrator().run()
EOF

# Create other nodes (minimal placeholders - expand as needed)
cat << EOF > LLM_Node.py
#!/usr/bin/env python3
import subprocess
import json

class LLMNode:
    def __init__(self):
        self.model = "llama3.1:8b"

    def generate(self, prompt):
        try:
            result = subprocess.run(["ollama", "run", self.model, prompt], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            return f"LLM error: {str(e)}"
EOF

# Add placeholders for other files
touch Memory_Node.py Nonsense_Node.py Knowledge_Node.py Reasoning_Node.py
touch Conversational_Intelligence_Node.py Web_Crawler_Node.py Evolver_Node.py

echo "Setup complete. Run 'python3 Orchestrator.py' to start Sentience."
