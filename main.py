import os
import json
import time
from datetime import datetime
from carbontracker.tracker import CarbonTracker
from ddgs import DDGS
import ollama

class GemmaLocalAgenticTask:
    def __init__(self, model_name="gemma:2b", agency_level="none"):
        self.model_name = model_name
        self.agency_level = agency_level
        self.ddgs = DDGS()

    def _generate(self, system_instruction, user_content):
        """Executes local inference via Ollama and triggers GPU power draw for tracking."""
        print(f"  [Gemma] Running local inference with {self.model_name}...")
        full_prompt = f"Instruction: {system_instruction}\n\nContext and Data: {user_content}\n\nOutput:"
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content']

    def _format_sources(self, results):
        """Formats search results into a dense paragraph of citations for the report."""
        if not results:
            return "Sources Consulted: Internal model knowledge only; no external search performed."
        formatted = "Sources Consulted: "
        for i, r in enumerate(results):
            title = r.get('title', 'Unknown Title')
            body = r.get('body', '')
            formatted += f"({i+1}) {title}: {body[:100]}... "
        return formatted

    def execute(self, keywords):
        print(f"Starting Local Gemma {self.agency_level} agency task for: {keywords}")
        all_retrieved_results = []
        
        # Agency Tier 0: No Search Baseline
        if self.agency_level == "none":
            print(f"  [Baseline] Skipping search for '{self.agency_level}' level.")
            system_instruction = "You are a senior research scientist. Write a dense, scientific literature review based entirely on your internal training data. Do not use bullet points or lists."
            content = self._generate(system_instruction, f"Topic: {keywords}")
            return f"{content}\n\n{self._format_sources([])}"

        # Step 1: Primary Data Retrieval
        print(f"  [Search] Calling DuckDuckGo for primary query: '{keywords}'")
        try:
            primary_results = list(self.ddgs.text(keywords, max_results=3))
            all_retrieved_results.extend(primary_results)
            print(f"  [Search] Success: Retrieved {len(primary_results)} primary sources.")
        except Exception as e:
            print(f"  [Error] Primary search failed: {e}")
            primary_results = []

        context_data = "\n".join([f"Source {i+1}: {r.get('body', '')}" for i, r in enumerate(primary_results)])
        
        # Agency Tier 1: Low Agency (Single-pass Summary)
        if self.agency_level == "low":
            system_instruction = "You are a research assistant. Write a dense, scientific literature review summary based on the provided data. Do not use bullet points."
            content = self._generate(system_instruction, context_data)
            return f"{content}\n\n{self._format_sources(all_retrieved_results)}"
        
        # Agency Tier 2: Medium Agency (Iterative Refinement)
        elif self.agency_level == "medium":
            print("  [Agent] Medium Agency: Identifying gaps in the sustainability literature...")
            analysis_instruction = "Review these snippets and identify one specific technical gap. Output only a short search query."
            gap_query = self._generate(analysis_instruction, context_data)
            
            print(f"  [Search] Calling DuckDuckGo for refinement query: '{gap_query}'")
            try:
                secondary_results = list(self.ddgs.text(gap_query, max_results=2))
                all_retrieved_results.extend(secondary_results)
                print(f"  [Search] Success: Retrieved {len(secondary_results)} refinement sources.")
            except Exception as e:
                print(f"  [Error] Refinement search failed: {e}")
                secondary_results = []
                
            secondary_data = "\n".join([f"Additional Source {i+1}: {r.get('body', '')}" for i, r in enumerate(secondary_results)])
            combined_context = f"PRIMARY DATA:\n{context_data}\n\nREFINEMENT DATA:\n{secondary_data}"
            final_instruction = "Synthesize a professional literature review using all data. Maintain a dense prose style and avoid bullet points."
            content = self._generate(final_instruction, combined_context)
            return f"{content}\n\n{self._format_sources(all_retrieved_results)}"

        # Agency Tier 3: High Agency (Chain-of-Thought Verification)
        else:
            print("  [Agent] High Agency: Generating a structured literature review plan...")
            plan_instruction = "Outline a 3-point plan to investigate specific carbon metrics. Output as a single dense paragraph."
            research_plan = self._generate(plan_instruction, context_data)
            
            print("  [Agent] High Agency: Executing multi-stage verification...")
            verification_data_list = []
            steps = [s.strip() for s in research_plan.replace('\n', '.').split('.') if len(s.strip()) > 10]
            for i, step in enumerate(steps[:3]):
                print(f"    - Verifying Research Dimension {i+1}...")
                try:
                    step_search = list(self.ddgs.text(f"{keywords} {step[:40]}", max_results=1))
                    if step_search:
                        all_retrieved_results.extend(step_search)
                        verification_data_list.append(step_search[0].get('body', ''))
                except Exception:
                    continue
            
            final_context = f"CONTEXT:\n{context_data}\n\nPLAN:\n{research_plan}\n\nVERIFIED DATA:\n" + "\n".join(verification_data_list)
            final_instruction = "As a senior research scientist, synthesize a definitive literature review. Use dense, sophisticated prose and no bullet points."
            content = self._generate(final_instruction, final_context)
            return f"{content}\n\n{self._format_sources(all_retrieved_results)}"

def run_experiment(model_name, agency_level, keywords, run_index, log_dir="carbon_logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize CarbonTracker to monitor energy during this specific execution block
    tracker = CarbonTracker(epochs=1, log_dir=log_dir, monitor_epochs=1, update_interval=1)
    tracker.epoch_start()
    
    report = ""
    try:
        agent = GemmaLocalAgenticTask(model_name=model_name, agency_level=agency_level)
        report = agent.execute(keywords)
        
        print("\n" + "="*50)
        print(f"FINAL LITERATURE REVIEW ({model_name} - {agency_level} agency - Run {run_index}):")
        print("="*50)
        print(report)
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Experiment failed during Run {run_index}: {e}")
    finally:
        tracker.epoch_end()
    
    run_id = f"{model_name.replace(':', '-')}_{agency_level}_run{run_index}_{int(time.time())}"
    
    if report:
        with open(os.path.join(log_dir, f"lit_review_{run_id}.txt"), "w") as f:
            f.write(report)
            
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "agency_level": agency_level,
        "run_index": run_index,
        "keywords": keywords,
        "type": "local_gpu_inference",
        "review_file": f"lit_review_{run_id}.txt"
    }
    
    with open(os.path.join(log_dir, f"meta_{run_id}.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    # Ensure Ollama has these models pulled: 'ollama pull gemma:2b' and 'ollama pull gemma:7b'
    MODELS = ["gemma:2b","qwen2.5:0.5b"]
    LEVELS = ["none", "low", "medium", "high"]
    TOPIC = "literature review on environmental sustainability of AI"
    REPETITIONS = 3 
    
    for model in MODELS:
        for level in LEVELS:
            for i in range(1, REPETITIONS + 1):
                print(f"\n--- EXPERIMENT: {model} | AGENCY: {level} | RUN: {i}/{REPETITIONS} ---")
                run_experiment(model, level, TOPIC, i)
