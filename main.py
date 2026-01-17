import os
import json
import time
import re
import glob
import shutil
from datetime import datetime
from carbontracker.tracker import CarbonTracker
from ddgs import DDGS
import ollama

MAX_RES = 2

class GemmaLocalAgenticTask:
    def __init__(self, model_name="gemma:2b", agency_level="none"):
        self.model_name = model_name
        self.agency_level = agency_level
        self.ddgs = DDGS()

    def _generate(self, system_instruction, user_content):
        print(f"  [Gemma] Running local inference with {self.model_name}...")
        full_prompt = f"Instruction: {system_instruction}\n\nContext and Data: {user_content}\n\nOutput:"
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content']

    def _format_sources(self, results):
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
        
        if self.agency_level == "none":
            system_instruction = "You are a senior research scientist. Write a dense, scientific literature review based entirely on your internal training data. Do not use bullet points or lists."
            content = self._generate(system_instruction, f"Topic: {keywords}")
            return f"{content}\n\n{self._format_sources([])}"
       
        elif self.agency_level == "low":
            try:
                primary_results = list(self.ddgs.text(keywords, max_results=MAX_RES))
                all_retrieved_results.extend(primary_results)
            except Exception:
                primary_results = []

            context_data = "\n".join([f"Source {i+1}: {r.get('body', '')}" for i, r in enumerate(primary_results)])


            system_instruction = "You are a research assistant. Write a dense, scientific literature review summary based on the provided data. Do not use bullet points."
            content = self._generate(system_instruction, context_data)
            return f"{content}\n\n{self._format_sources(all_retrieved_results)}"
        
        elif self.agency_level == "medium":
            try:
                primary_results = list(self.ddgs.text(keywords, max_results=MAX_RES))
                all_retrieved_results.extend(primary_results)
            except Exception:
                primary_results = []

            context_data = "\n".join([f"Source {i+1}: {r.get('body', '')}" for i, r in enumerate(primary_results)])


            analysis_instruction = "Review these snippets and identify one specific technical gap. Output only a short search query."
            gap_query = self._generate(analysis_instruction, context_data)
            try:
                secondary_results = list(self.ddgs.text(gap_query, max_results=MAX_RES))
                all_retrieved_results.extend(secondary_results)
            except Exception:
                secondary_results = []
            secondary_data = "\n".join([f"Additional Source {i+1}: {r.get('body', '')}" for i, r in enumerate(secondary_results)])
            combined_context = f"PRIMARY DATA:\n{context_data}\n\nREFINEMENT DATA:\n{secondary_data}"
            final_instruction = "Synthesize a professional literature review using all data. Maintain a dense prose style and avoid bullet points."
            content = self._generate(final_instruction, combined_context)
            return f"{content}\n\n{self._format_sources(all_retrieved_results)}"

        else: # High Agency
            try:
                primary_results = list(self.ddgs.text(keywords, max_results=MAX_RES))
                all_retrieved_results.extend(primary_results)
            except Exception:
                primary_results = []

            context_data = "\n".join([f"Source {i+1}: {r.get('body', '')}" for i, r in enumerate(primary_results)])


            plan_instruction = "Outline a 3-point plan to investigate specific carbon metrics. Output as a single dense paragraph."
            research_plan = self._generate(plan_instruction, context_data)
            verification_data_list = []
            steps = [s.strip() for s in research_plan.replace('\n', '.').split('.') if len(s.strip()) > 10]
            for i, step in enumerate(steps[:3]):
                try:
                    step_search = list(self.ddgs.text(f"{keywords} {step[:40]}", max_results=MAX_RES))
                    if step_search:
                        all_retrieved_results.extend(step_search)
                        verification_data_list.append(step_search[0].get('body', ''))
                except Exception:
                    continue
            final_context = f"CONTEXT:\n{context_data}\n\nPLAN:\n{research_plan}\n\nVERIFIED DATA:\n" + "\n".join(verification_data_list)
            final_instruction = "As a senior research scientist, synthesize a definitive literature review. Use dense, sophisticated prose and no bullet points."
            content = self._generate(final_instruction, final_context)
            return f"{content}\n\n{self._format_sources(all_retrieved_results)}"

def parse_last_run_from_log(session_dir):
    std_logs = glob.glob(os.path.join(session_dir, "**/*_carbontracker.log"), recursive=True)
    if not std_logs:
        return {}
    latest_std_log = max(std_logs, key=os.path.getmtime)
    latest_output_log = latest_std_log.replace("_carbontracker.log", "_carbontracker_output.log")
    
    metrics = {"gpu_power_w": 0.0, "cpu_power_w": 0.0, "duration_s": 0.0, "total_energy_kwh": 0.0, "total_co2eq_g": 0.0}
    gpu_re = re.compile(r"Average power usage \(W\) for gpu:\s+([\d.]+)", re.IGNORECASE)
    cpu_re = re.compile(r"Average power usage \(W\) for cpu:\s+([\d.]+)", re.IGNORECASE)
    dur_re = re.compile(r"Duration:\s+(\d+):(\d+):([\d.]+)", re.IGNORECASE)
    energy_re = re.compile(r"Energy:\s+([\d.]+)\s+kWh", re.IGNORECASE)
    co2_re = re.compile(r"CO2eq:\s+([\d.]+)\s+g", re.IGNORECASE)

    with open(latest_std_log, 'r') as f:
        for line in f:
            gm = gpu_re.search(line); cm = cpu_re.search(line); dm = dur_re.search(line)
            if gm: metrics["gpu_power_w"] = float(gm.group(1))
            if cm: metrics["cpu_power_w"] = float(cm.group(1))
            if dm:
                h, m, s = map(float, dm.groups())
                metrics["duration_s"] = h * 3600 + m * 60 + s

    if os.path.exists(latest_output_log):
        with open(latest_output_log, 'r') as f:
            content = f.read()
            actual_block = content.split("Actual consumption")[-1]
            en_m = energy_re.search(actual_block); co_m = co2_re.search(actual_block)
            if en_m: metrics["total_energy_kwh"] = float(en_m.group(1))
            if co_m: metrics["total_co2eq_g"] = float(co_m.group(1))

    if metrics["duration_s"] > 0:
        metrics["gpu_energy_kwh"] = (metrics["gpu_power_w"] * metrics["duration_s"]) / 3600000
        metrics["cpu_energy_kwh"] = (metrics["cpu_power_w"] * metrics["duration_s"]) / 3600000
    return metrics

def run_experiment(model_name, agency_level, keywords, run_index, session_dir,repeats=10):
    tracker = CarbonTracker(epochs=1, log_dir=session_dir, monitor_epochs=1, update_interval=1)
    tracker.epoch_start()
    report = ""
    try:
        for _ in range(repeats):
            agent = GemmaLocalAgenticTask(model_name=model_name, agency_level=agency_level)
            report = agent.execute(keywords)
    finally:
        tracker.epoch_end()
    tracker.stop() 
    hw_metrics = parse_last_run_from_log(session_dir)
    run_id = f"{model_name.replace(':', '-')}_{agency_level}_run{run_index}_{int(time.time())}"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "agency_level": agency_level,
        "run_index": run_index,
        "metrics": hw_metrics,
        "report_file": f"lit_review_{run_id}.txt"
    }
    
    with open(os.path.join(session_dir, f"meta_{run_id}.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    if report:
        with open(os.path.join(session_dir, f"lit_review_{run_id}.txt"), "w") as f:
            f.write(report)

if __name__ == "__main__":
    MODELS = ["qwen2.5:0.5b","qwen2.5:1.5b","qwen2.5:3b","qwen2.5:7b","gemma3:270m","gemma3:1b","gemma3:4b"]
    LEVELS, REPS = ["none", "low", "medium", "high"], 3
    TOPIC = "literature review on environmental sustainability of AI"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("carbon_logs", f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    for model in MODELS:
        for level in LEVELS:
            for i in range(1, REPS + 1):
                print(f"\n--- EXPERIMENT: {model} | AGENCY: {level} | RUN: {i}/{REPS} ---")
                run_experiment(model, level, TOPIC, i, session_dir)
    print(f"\nExperiment session complete. Data stored in: {session_dir}")
