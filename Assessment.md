# Labs Interview Assessment

## Overview

Meerkat is an enterprise AI chat platform for healthcare analytics. Think ChatGPT, but specifically built for querying healthcare databases, generating insights, and conducting multi-step research workflows.

At its core, Meerkat is powered by:

- **LLM agents** that orchestrate complex analysis tasks
- **Multi-step workflows** where agents autonomously break down and solve complex problems

This assessment simulates a simplified version of Meerkat's architecture to evaluate your skills for working on our production codebase.

---

## What You Have

We've provided a working starter system with:

### 1. Healthcare Database (`healthcare.duckdb`)

Synthetic healthcare data modeling real-world claims and patient records:

**Patient Data:**
- `demographics` - Patient demographics (10,000 patients with DOB, sex)
- `geography` - Patient location data (state, ZIP, validity periods)
- `mortality` - Death records (~5% of patients)
- `mx_events` - Medical/claims events (~2.5M rows: procedures, diagnoses, billing)
- `rx_events` - Pharmacy events (~2.5M rows: prescriptions, fills, drug info)

**Code Lookup Tables:**
- `icd10_codes` - ICD-10 diagnosis codes (~80K codes)
- `procedure_codes` - CPT/HCPCS procedure codes (~20K codes)
- `ndc_products` - NDC drug products (~114K products)

All patient tables are joinable via `PATIENT_NUMBER`. Medical events share NPI pools for provider joinability.

### 2. SQL Query Module (`tools/db_query.py`)

A production-ready tool that provides:

- `query()` - Execute SQL, return raw results
- `query_pretty()` - Display formatted tables
- `query_to_csv()` / `query_to_json()` - Export results
- `list_tables()`, `describe_table()`, `table_info()` - Database exploration

### 3. Example Chat Application (`chat.py`)

A basic LLM agent with tool calling that can:

- Query the database through natural language
- Execute multiple tools in sequence
- Maintain conversation history
- Display results to users

**This already works.** You can run it and chat with the database right now.

---

## Your Mission

**Show us how creative you can be.**

We want to see what you think would make this system better. How would you extend it? What features would you add? What interesting capabilities could an AI healthcare analyst have?

### The Basic Requirements

Your solution should:

- Use LLM tool/function calling (OpenAI, Anthropic, or similar)
- Interact with the healthcare database
- Demonstrate some level of agent orchestration
- Show clean, readable code

Beyond that? **Surprise us.**

---

## Ideas to Spark Your Creativity

*Note: These are just suggestions. Don't feel constrained by this list. Build what excites you.*

### Agent Architecture Ideas

- **Multi-agent systems** - Different agents for different tasks (data analyst, visualizer, report writer)
- **Planning & execution** - Agent creates a plan before executing queries
- **Self-correction** - Agent detects errors and fixes them autonomously
- **Memory systems** - Agent remembers important facts across conversations
- **Sub-agents** - Main agent spawns specialized sub-agents for specific tasks

### Tool Ideas

- **Data visualization** - Generate charts/graphs from query results
- **Statistical analysis** - Automatic correlation detection, significance testing
- **Report generation** - Create formatted markdown/PDF reports
- **Data quality checks** - Detect missing data, outliers, inconsistencies
- **Query optimization** - Suggest more efficient SQL queries
- **Natural language explanations** - Translate SQL and results into plain English
- **Cross-dataset analysis** - Join and analyze across multiple tables intelligently
- **Caching layer** - Store and reuse previous query results
- **Query history** - Track and reference previous queries

### UX/Interface Ideas

- **Streaming responses** - Show results as they come in
- **Interactive refinement** - "Show me more details about patient X"
- **Query suggestions** - Proactively suggest interesting queries
- **Data summaries** - Automatic overview when exploring new tables
- **Error handling** - Graceful failures with helpful messages
- **Progress indicators** - Show what the agent is doing
- **Export workflows** - Save entire analysis sessions

### Architecture Ideas

- **Conversation state** - Track SQL views, intermediate results, user context
- **Tool chaining** - Results from one tool feed into another
- **Extensible tool system** - Easy to add new capabilities
- **Async execution** - Handle long-running queries elegantly
- **Testing framework** - How would you test an agent system?
- **Observability** - Logging, tracing, debugging agent decisions

### Healthcare-Specific Ideas

- **Cohort analysis** - Find patients matching specific criteria across tables
- **Drug interaction detection** - Identify potentially dangerous combinations in rx_events
- **Care gap analysis** - Find patients missing expected follow-up events
- **Cost analysis** - Analyze billing patterns, plan payments, patient responsibility
- **Provider network analysis** - Map relationships between NPIs across events
- **Diagnosis trending** - Track ICD-10 code patterns over time
- **Prescription patterns** - Analyze prescribing behaviors by provider, region, or patient demographics

---

## What We're Evaluating

### Creativity & Problem Solving (Most Important)

- How interesting/novel are your ideas?
- Do you go beyond the obvious?
- Do you think about user experience?
- Do you demonstrate product thinking?

### Technical Implementation

- **LLM Integration** - Tool calling, prompt design, response handling
- **Python Architecture** - Code organization, patterns, structure
- **Database Integration** - SQL generation, safety, result processing
- **Tool System Design** - How tools are defined, executed, composed
- **State Management** - Tracking context across conversation turns

### Engineering Quality

- **Code Quality** - Clean, readable, well-organized
- **Error Handling** - Graceful failures, helpful messages
- **Extensibility** - Easy to add new features
- **Documentation** - Clear explanation of your design
- **Testing (bonus)** - How you'd validate your system

---

## Time Guidance

**2-4 hours.**

We'd rather see a few well-implemented, creative features than many half-finished ideas.

Focus on showing your strengths. If you're great at UX, make it delightful. If you love architecture, show us elegant patterns. If you're passionate about ML, do something clever with the LLM.

---

## Deliverables

### 1. Your Code

- Well-organized and documented
- Easy to understand and run
- Demonstrates your best work

### 2. README Updates

Should include:

- **What you built** - High-level overview of features
- **How to run it** - Setup and usage instructions
- **Architecture decisions** - Why you made key choices
- **What you'd add next** - Given more time, what would you build?
- **Demo/examples** - Screenshots, code samples, or example sessions

### 3. Show It Working

- Example conversation or workflow
- Screenshots (if relevant)
- Sample output files (if you generate reports/visualizations)

---

## Technical Constraints

- Python 3.11+ (starter code uses 3.13)
- Use the provided `tools/db_query.py` module for database access (don't modify it)
- Use an LLM with tool calling support (OpenAI, Anthropic, etc.)
- Keep the existing data as-is (don't need to modify the database)

Beyond these constraints, you have complete freedom.

---

## Getting Started

The starter code already works! Try it:

```bash
# Setup (one command)
make all

# Add your OPENAI_API_KEY to .env
cp env.template .env
# Edit .env and add your key

# Run the example chat
uv run chat.py
```

Ask it:
- "What tables are available?"
- "How many patients are in the database?"
- "What are the most common diagnosis codes?"
- "Show me the average age of patients by state"

Then ask yourself: **How can I make this 10x better?**

---

## Tips

### Think About Real Use Cases

Imagine you're a healthcare data analyst. What would make your job easier? What insights would be valuable? Consider questions like:

- "Which patients have both diabetes diagnoses and metformin prescriptions?"
- "What's the average time between diagnosis and first treatment?"
- "Which providers have the highest patient volumes?"
- "Are there regional patterns in prescription behavior?"

### Start Simple, Then Iterate

Get something working, then make it better. Don't try to build everything at once.

### Show Your Personality

We want to see how you think. There's no single "right answer."

### Consider the Meerkat Context

In production, Meerkat handles:

- Complex multi-step analysis workflows
- Multiple users with different needs
- Large datasets with expensive queries
- Need for reproducibility and auditability

How might these constraints influence your design?

### Have Fun!

This is a chance to build something cool. Enjoy it.

---

## Questions?

If anything is unclear about the requirements (not the code - we want to see how you debug!), reach out to your interview coordinator.

Otherwise, the floor is yours. **Show us what you've got.**

