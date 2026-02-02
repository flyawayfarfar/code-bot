# 🎥 CodeBot – 3-Minute Hackathon Pitch Script

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

## 1:20–2:20 — Demo: cross-project lending flow (main showcase)

> Let’s look at a real example using our lending systems.

> When a third-party submits a request to the Lending Gateway, what actually happens inside our platform?

*(Demo runs)*

> CodeBot identifies the entry endpoint in Lending Gateway, follows the service logic, and traces the call into Lending Apply.

> What’s important here isn’t just the explanation — it’s that CodeBot understands how multiple projects work together, not just a single repository.

> Instead of jumping between codebases, we get the full end-to-end picture in one place, with references to the actual classes and methods.

---

## 2:20–2:45 — Reuse & consistency

> Another common question teams ask is whether something already exists.

> By surfacing shared utilities and patterns, CodeBot helps teams avoid duplicate code and inconsistent logic.

> This improves delivery speed and overall system quality.

---

## 2:45–3:15 — Trust, risk, and hallucination (explicit)

> A natural question with any AI system is trust.

> CodeBot runs locally and answers questions using retrieved code fragments, not free-form guessing.

> Answers are grounded in the code and can point to exact files and methods, so users can verify them directly in their IDE.

> If stricter requirements apply, the same architecture can run with fully local models.

---

## 3:15–3:45 — Current limitations (honest and controlled)

> This is an MVP, and there are clear limitations today.

> At the moment, users may need to guide the response — for example, whether they want an architectural overview or a technical deep dive.

> We’ve focused on two related projects to demonstrate cross-project understanding, and we still need to validate performance with much larger repositories.

> Response accuracy and depth will continue to improve through better retrieval strategies and prompt configuration.

---

## 3:45–4:20 — Verification & future confidence improvements

> Today, answers can be verified in two ways:  
> by asking CodeBot to show the code locations, and by opening the code directly in the IDE.

> A possible future enhancement is a secondary verifier agent that checks whether answers are fully supported by retrieved code and flags uncertainty.

> This would add an extra confidence layer rather than replacing human judgement.

---

## 4:20–4:50 — Access control, scope, and real usage

> Access control and role-based scoping are not implemented yet.

> Currently, anyone can ask questions within the indexed projects.  
> For production, role-based access control would ensure teams only see what they’re authorised to see.

> A login system could also tailor responses based on team or role, so answers match the audience automatically.

---

## 4:50–5:10 — Cost & rollout awareness

> Cost and scalability need to be validated as part of rollout planning.

> Based on early indications, the value gained from faster onboarding, reduced spikes, and better knowledge sharing makes this worth exploring further.

---

## 5:10–5:30 — Positioning vs Copilot (clear differentiation)

> Tools like GitHub Copilot are great for writing code.

> CodeBot solves a different problem — understanding how systems behave across multiple projects.

> They complement each other rather than replace each other.

---

## 5:30–5:45 — Close

> CodeBot turns tribal knowledge into something searchable.

> It helps teams onboard faster, understand systems more clearly, and spend less time searching or asking around.

> That’s what we wanted to demonstrate today.
