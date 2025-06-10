
# 🔍 Lecture Summary: Continuous Evaluation and CI at Groq (Aarush Sah)

## 🧠 Introduction
- **Speaker:** Aarush Sah (Head of Evals at Groq)
- **Host:** Hamel Husain
- **Topic:** How Groq builds and scales **CI pipelines** and **Evals** for LLM reliability
- **Groq Overview:**
  - Builds fastest AI inference hardware (custom silicon)
  - Focus on large-scale inference with high performance

---

## 🔬 What Are Evals?
- Evals = **Tests** for model behavior and quality
- Used for:
  - Tracking quality
  - Measuring iteration improvements
  - Ensuring reliability

---

## 🧨 Real Incident: Why CI Evals Matter
- Optimization improved model speed by 20%
- CI & unit tests passed, **BUT**:
  - Config change limited tokens to 16K
  - Model supported 131K tokens
  - Long prompts failed with 503 errors
- **Lesson:** Only **end-to-end evals** would have caught it

---

## 🧱 Layered Testing Strategy
- Groq uses **mono-repo**, multiple layers:
  - Firmware, compilers, inference engine
  - Models, APIs, load balancers
- Each layer has tests:
  - Firmware → simulation
  - Compiler → numerical stability
  - Model → throughput/quality evals
  - API → system-wide reliability
- **Key Insight:** Evals = part of unified testing strategy

---

## 📏 SLOs (Service Level Objectives)
- Guide all tests and evals
- Examples:
  - Latency, availability, error rate
  - Accuracy, coherence, safety
- SLO = **North Star**
- If SLO is violated → **No deploy**
- SLOs are **automated**, not static docs

---

## 🌐 Testing External Dependencies
- LLMs rely on third-party services
- Groq’s "Compound Beta" combines:
  - Web search
  - Code execution
- Problem: External services drift
- **Solution:**
  - Run **cron job evals** on known queries
  - Trigger full evals if behavior changes

---

## 🛠 Stratified CI Test Design
1. **Smoke Tests** (on every push)
   - Run in ~30s
   - Just checks “does it work at all?”
2. **Regression Tests** (post-merge)
   - MMLU, long prompts, stress cases
3. **Pre-Deployment Tests**
   - End-to-end stress & throughput
   - Simulates real traffic

---

## 🧠 Best Practices for CI & LLMs
- **Fail fast:** Run cheap tests first
- **Make failures actionable:**
  - E.g. “Tool accuracy dropped 5%” > “Quality degraded”
- **Visualize + monitor over time:**
  - Dashboards (e.g. Grafana)
  - Slack alerts

---

## 🔧 Infra & Tooling
- **Inspect:** Eval framework of choice
  - Multi-provider support (OpenAI, Anthropic, Cohere, etc.)
  - Better UX than LM Eval Harness
  - Great UI: `inspect you` command
  - Fast dev team
- **CI Platforms:**
  - GitHub + GitLab
  - Sensitive code (compilers) → GitLab
  - SaaS infra → GitHub

---

## ⚙️ Eval Suite for Compound Beta
- **Quick tests:** Tool call success
- **Medium tests:** Tool selection correctness
- **Full evals:** Complex reasoning, real-time evals
- Hierarchical setup minimizes cost & waste

---

## 💡 Lessons from Production
1. **Test the test:** Ensure evals themselves are robust
2. **Sample smartly:**
   - Edge cases
   - Customer-critical paths (e.g. Dropbox use case)
   - Past failures
3. **Alert & visualize everything:**
   - Slack, dashboards, regressions
   - Alert on *drift*, not just hard failures

---

## ✅ Actionable Takeaways
1. Define SLOs early
2. Add tests to CI now, even basic ones
3. Monitor all dependencies
4. Grow CI maturity **incrementally** with your product

---

## 🚀 What’s Coming from Groq
- **Open-sourcing Evals Toolkit**
  - Built on Inspect
  - CI integration + automation + UI
- **Dynamic LoRA support**
  - User-uploaded adapters on Groq API
  - High-speed inference

---

## 💼 They're Hiring: Evals Engineer
- Work on full-stack eval infra
- Integrate evals into CI
- Reliability + tooling + model quality
- “Like a founding engineer role”

---

## 📎 Connect
- Follow Groq: [Twitter @Grok_Inc](https://twitter.com/Grok_Inc)
- Aarush: Open to DMs on LinkedIn/Twitter
