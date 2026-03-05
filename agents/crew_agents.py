from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai import LLM
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# ── Load environment variables ─────────────────────────────
load_dotenv()

# ── Configure Groq as LLM ──────────────────────────────────
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",   # CrewAI recognizes "groq/" prefix
    temperature=0.1,
    max_tokens=8192,
    api_key=os.getenv("GROQ_API_KEY"),      # CrewAI uses api_key field
    base_url="https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
)

# ── Define Agents ──────────────────────────────────────────

parser_agent = Agent(
    role="Research Paper Parser",
    goal="Extract and filter the most relevant methodology sections from ML research papers",
    backstory="""You are an expert academic researcher with 10 years of experience 
    reading and analyzing machine learning papers. You know exactly which sections 
    contain the important implementation details and which are irrelevant.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

coder_agent = Agent(
    role="ML Code Generator",
    goal="Generate complete, runnable PyTorch code from research paper methodology",
    backstory="""You are a senior ML engineer who specializes in implementing 
    research papers from scratch. You always write clean, reproducible code with 
    proper seeds and clear comments.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

debugger_agent = Agent(
    role="Code Debugger",
    goal="Identify and fix errors in generated ML code automatically",
    backstory="""You are an expert Python debugger who has fixed thousands of 
    PyTorch errors. You can identify the root cause of any error and apply 
    the minimal fix needed to make code run correctly.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

tester_agent = Agent(
    role="Reproducibility Tester",
    goal="Verify that generated code reproduces the paper's claimed results within tolerance",
    backstory="""You are a scientific integrity researcher who specializes in 
    verifying ML paper claims. You calculate precise reproducibility scores and 
    identify when results match or deviate from paper claims.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

hallucination_agent = Agent(
    role="Hallucination Detector",
    goal="Identify assumptions made by the AI that are not supported by the paper",
    backstory="""You are an AI safety researcher specializing in detecting 
    hallucinations in LLM outputs. You carefully compare generated code against 
    the original paper to flag any unsupported assumptions.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

# ── Define Tasks ───────────────────────────────────────────

def create_tasks(paper_id, paper_text, rag_context, code, stdout, stderr,
                 repro_result, hallucination_result):

    parse_task = Task(
        description=f"""Parse the research paper {paper_id} and confirm the methodology 
        was successfully extracted. Paper text length: {len(paper_text)} characters. 
        RAG context length: {len(rag_context)} characters.
        Confirm the extraction was successful and summarize what was found.""",
        agent=parser_agent,
        expected_output="Confirmation of paper parsing with summary of extracted methodology"
    )

    code_task = Task(
        description=f"""Review the generated PyTorch code for paper {paper_id}.
        The code was generated from the RAG context.
        Code preview: {code[:500]}...
        Confirm the code structure is correct and follows ML best practices.""",
        agent=coder_agent,
        expected_output="Code review summary confirming structure and quality"
    )

    debug_task = Task(
        description=f"""Review the debugging results for paper {paper_id}.
        stdout: {stdout[:200] if stdout else 'No output'}
        stderr: {stderr[:200] if stderr else 'No errors'}
        Confirm whether code ran successfully or needed fixes.""",
        agent=debugger_agent,
        expected_output="Debugging summary with outcome and any fixes applied"
    )

    test_task = Task(
        description=f"""Review reproducibility results for paper {paper_id}.
        Expected accuracy: {repro_result['expected_accuracy']}%
        Actual accuracy: {repro_result['actual_accuracy']}%
        Score: {repro_result['reproducibility_score']}/100
        Status: {repro_result['status']}
        Provide a detailed analysis of the reproducibility results.""",
        agent=tester_agent,
        expected_output="Detailed reproducibility analysis with score justification"
    )

    hallucination_task = Task(
        description=f"""Review hallucination analysis for paper {paper_id}.
        Hallucination score: {hallucination_result['hallucination_score']}/100
        Assumptions made: {hallucination_result['total_assumptions']}
        Values from paper: {hallucination_result['total_from_paper']}
        Summary: {hallucination_result['summary']}
        Provide recommendations to reduce hallucinations.""",
        agent=hallucination_agent,
        expected_output="Hallucination analysis with recommendations"
    )

    return [parse_task, code_task, debug_task, test_task, hallucination_task]

# ── Run Crew ───────────────────────────────────────────────

def run_crew_analysis(paper_id, paper_text, rag_context, code,
                      stdout, stderr, repro_result, hallucination_result):

    print("\n🤖 Starting CrewAI Multi-Agent Analysis...")
    print("=" * 50)

    tasks = create_tasks(
        paper_id, paper_text, rag_context, code,
        stdout, stderr, repro_result, hallucination_result
    )

    crew = Crew(
        agents=[parser_agent, coder_agent, debugger_agent,
                tester_agent, hallucination_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()

    print("\n✅ CrewAI Analysis Complete!")
    print("=" * 50)

    return result