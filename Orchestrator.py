#!/usr/bin/env python3
"""
Orchestrator.py
Sentience 5.5 – Final evolved version with local LLM intelligence
"""

import sys

from Memory_Node import MemoryNode
from Nonsense_Node import NonsenseNode
from Knowledge_Node import KnowledgeNode
from Reasoning_Node import ReasoningNode
from Conversational_Intelligence_Node import ConversationalIntelligenceNode
from Web_Crawler_Node import WebCrawlerNode
from Evolver_Node import EvolverNode
from LLM_Node import LLMNode  # NEW


BANNER = r"""
███████╗███████╗███╗   ██╗████████╗██╗███████╗███╗   ██╗ ██████╗███████╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝████╗  ██║██╔════╝██╔════╝
███████╗█████╗  ██╔██╗ ██║   ██║   ██║█████╗  ██╔██╗ ██║██║     █████╗  
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  
███████║███████╗██║ ╚████║   ██║   ██║███████╗██║ ╚████║╚██████╗███████╗
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
Sentience — Evolved with local LLM intelligence
"""


class Orchestrator:
    def __init__(self):
        self.memory = MemoryNode()
        self.nonsense = NonsenseNode()
        self.knowledge = KnowledgeNode()

        self.crawler = WebCrawlerNode(memory=self.memory)
        self.evolver = EvolverNode(
            memory=self.memory,
            knowledge=self.knowledge,
            crawler=self.crawler
        )

        self.reasoning = ReasoningNode(
            knowledge=self.knowledge,
            memory=self.memory
        )

        # LLM Node — change model_name to whatever you have (e.g. "mistral", "phi3", "llama3.1")
        self.llm = LLMNode(model_name="llama3.1")  # or "llama3.1:8b" if tagged

        self.chat = ConversationalIntelligenceNode(
            memory=self.memory,
            nonsense=self.nonsense,
            knowledge=self.knowledge,
            reasoning=self.reasoning,
            crawler=self.crawler,
            evolver=self.evolver,
            llm=self.llm  # Passed in
        )

        print(BANNER)
        print("Sentience online — local LLM active. Full intelligence enabled.\n")

    def run(self):
        while True:
            try:
                user_input = input(">> ").strip()
                if not user_input:
                    continue

                # Direct commands still work
                if user_input.lower().startswith(("crawl ", "research ", "evolve on ")):
                    if "crawl" in user_input.lower() or "research" in user_input.lower():
                        query = user_input.split(" ", 2)[-1] if len(user_input.split()) > 2 else ""
                        print(f"[CRAWL] Researching: {query}")
                        results = self.crawler.search_and_crawl(query)
                        print(f"[CRAWL] Complete — {len(results)} pages assimilated.")
                        self.evolver.assimilate_crawl_data()
                        continue

                    if "evolve" in user_input.lower():
                        topic = user_input.split(" ", 2)[-1]
                        resp = self.evolver.evolve_from_query(topic)
                        print(resp)
                        continue

                if user_input.lower() in {"evolution status", "llm status"}:
                    print(self.evolver.status())
                    continue

                # Normal conversation → now powered by local LLM
                response = self.chat.respond(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\nShutting down.")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] {str(e)}")


if __name__ == "__main__":
    Orchestrator().run()
