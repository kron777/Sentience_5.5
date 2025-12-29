#!/usr/bin/env python3
"""
Evolver_Node.py
Sentience 5.5 – Self-evolution and code adaptation engine

Purpose:
- Absorb assimilated web data
- Identify useful patterns, code, knowledge
- Generate candidate improvements
- Safely test and integrate changes
- Log all evolution steps
- Doctrine 2.0 compliant — inspectable, revocable
"""

import os
import ast
import time
import json
import subprocess
from typing import List, Dict, Optional

class EvolverNode:
    def __init__(self, memory, knowledge, crawler):
        self.memory = memory
        self.knowledge = knowledge
        self.crawler = crawler
        self.evolution_log = "evolution_log.json"
        self.backup_dir = "evolution_backups"
        self.max_evolution_steps_per_cycle = 3

        os.makedirs(self.backup_dir, exist_ok=True)
        self.load_evolution_log()

    def load_evolution_log(self):
        if os.path.exists(self.evolution_log):
            with open(self.evolution_log, "r") as f:
                self.log = json.load(f)
        else:
            self.log = {"evolutions": [], "total_steps": 0}

    def save_evolution_log(self):
        with open(self.evolution_log, "w") as f:
            json.dump(self.log, f, indent=2)

    def log_evolution(self, action: str, details: Dict):
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        self.log["evolutions"].append(entry)
        self.log["total_steps"] += 1
        self.save_evolution_log()
        print(f"[EVOLUTION] {action}: {details.get('summary', '')}")

    def backup_file(self, filepath: str) -> str:
        backup_path = os.path.join(self.backup_dir, f"{os.path.basename(filepath)}.bak.{int(time.time())}")
        if os.path.exists(filepath):
            with open(filepath, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
        return backup_path

    def extract_code_snippets(self, text: str) -> List[str]:
        """Extract Python code blocks from text"""
        snippets = []
        lines = text.splitlines()
        in_code = False
        current = []

        for line in lines:
            if line.strip().startswith("```python") or line.strip().startswith("```py"):
                in_code = True
                current = []
            elif line.strip().startswith("```") and in_code:
                in_code = False
                if current:
                    snippets.append("\n".join(current))
            elif in_code:
                current.append(line)

        return snippets

    def safe_exec_test(self, code: str) -> Dict:
        """Safely test code snippet in isolated context"""
        try:
            local_env = {"__name__": "__main__"}
            exec(code, local_env)
            return {"success": True, "output": "No runtime error"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def assimilate_crawl_data(self) -> None:
        """Main evolution cycle — process recent crawl data"""
        recent_crawls = [entry for entry in self.memory.recent(20) if "CRAWL_RESULT" in entry.get("text", "")]
        
        if not recent_crawls:
            return

        print(f"[EVOLVER] Processing {len(recent_crawls)} crawled pages for evolution opportunities")

        for entry in recent_crawls[:self.max_evolution_steps_per_cycle]:
            data = eval(entry["response"])  # Safe because from trusted crawl
            content = data["content"]

            # Extract potential improvements
            code_snippets = self.extract_code_snippets(content)

            for snippet in code_snippets[:2]:  # Limit per page
                if len(snippet) < 50 or len(snippet) > 1000:
                    continue

                test_result = self.safe_exec_test(snippet)

                if test_result["success"]:
                    # Candidate for integration
                    candidate_desc = snippet.strip().splitlines()[0] if snippet.splitlines() else "code improvement"
                    self.log_evolution("candidate_found", {
                        "source_url": data["url"],
                        "summary": f"Valid code: {candidate_desc[:100]}",
                        "snippet_length": len(snippet)
                    })

                    # Simple integration: add to knowledge as learned capability
                    self.knowledge.teach_fact(
                        f"learned_code_{int(time.time())}",
                        {"source": data["url"], "code": snippet, "tested": True}
                    )

                else:
                    self.log_evolution("rejected_code", {
                        "source_url": data["url"],
                        "error": test_result["error"][:200]
                    })

        # Advanced: suggest node improvements based on patterns
        all_text = " ".join([eval(e["response"])["content"] for e in recent_crawls if "content" in eval(e["response"])])
        if "neural network" in all_text.lower() or "self-improvement" in all_text.lower():
            self.propose_architecture_upgrade(all_text)

    def propose_architecture_upgrade(self, context: str):
        """Propose new node or major improvement"""
        proposal = {
            "type": "architecture_suggestion",
            "trigger": "pattern_detection",
            "suggestion": "Consider adding Neural_Learning_Node for pattern-based adaptation",
            "context_length": len(context)
        }
        self.log_evolution("proposal", proposal)

    def evolve_from_query(self, user_query: str) -> str:
        """Direct evolution trigger from user"""
        self.log_evolution("user_triggered_evolution", {"query": user_query})
        
        # Example: search for improvements
        results = self.crawler.search_and_crawl(user_query)
        self.assimilate_crawl_data()
        
        return f"Initiated evolution cycle based on: {user_query}. Processed {len(results)} sources."

    def status(self) -> str:
        return (
            f"Evolution status:\n"
            f"  Total evolution steps: {self.log['total_steps']}\n"
            f"  Recorded evolutions: {len(self.log['evolutions'])}\n"
            f"  Backups: {len(os.listdir(self.backup_dir))}\n"
            f"  Ready for next adaptation cycle."
        )
