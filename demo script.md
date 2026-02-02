# 🎥 CodeBot – Hackathon Demo Script

## 0:00–0:30 — Opening: the real problem

> In large banking systems, one of the hardest problems isn’t writing new code — it’s understanding how existing systems actually work.

> When a change request, incident, audit question, or new feature comes in, people often don’t know where to start.  
> So they ask around, or spend days doing spikes just to find the right area.

---

## 0:30–0:55 — What CodeBot is (core metaphor)

> CodeBot is like Wikipedia for source code.

> It reads our application code and lets people ask questions in plain English — with answers grounded in the real code.

> Instead of manually searching repositories, people can ask the system directly.

---

## 0:55–1:20 — Who it’s for (without labels)

> Different people ask different questions.

> Architects want to understand end-to-end flows or response payloads.  
> Risk teams want early signals about potential issues.  
> Delivery teams want to know where to look before making changes.

> CodeBot gives answers based on what the person is trying to understand.

---

## 1:20–2:20 — Demo: cross-project understanding (main showcase)

> Let me show you how this works with our indexed codebase.

> *"How do these services connect together?"*

*(Demo runs - showing live query and response)*

> CodeBot searches through multiple projects and shows how they're actually connected — what APIs they call, what data structures they share, and where the integration points are.

> This isn't just documentation — it's reading the actual code and understanding the relationships between different parts of the system.

> Instead of hunting through repositories, you get the full picture with references to the exact files and methods.

---

## 2:20–2:45 — Reuse & consistency

> Another common question teams ask is whether something already exists.

> By surfacing shared utilities and patterns, CodeBot helps teams avoid duplicate code and inconsistent logic.

> This improves delivery speed and overall system quality.

---

## 2:45–3:15 — Security & deployment (technical approach)

> We built this as a RAG system using vector embeddings and semantic search.

> For production banks, we offer fully local deployment with Ollama — complete air-gapped operation with no external API calls.

> For development environments, cloud deployment with Google AI or OpenAI works well, though this does involve bulk indexing your codebase for embedding generation.

> The vector database and knowledge base always stay on your infrastructure, giving you complete control over your data.


---

## 3:15–3:30 — What's next

> This is an MVP focused on proving the core concept.

> The next steps are scaling to larger codebases and adding role-based access controls for production deployment.

---

## 3:30–3:45 — Why this matters

> Tools like GitHub Copilot help you write code faster.

> CodeBot helps you understand existing systems faster.

> Together, they transform how teams work with complex codebases.

---

## 3:45–4:00 — Close

> We've built something that turns your codebase into a searchable knowledge base.

> It's RAG applied to source code, with the flexibility to run locally or in the cloud.

> This could change how teams onboard, how they understand systems, and how they make changes safely.

> Thanks for watching!
