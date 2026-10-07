# Product
<!-- impeccable:product-schema 1 -->

## Platform
web

## Users
People building and evaluating retrieval-augmented generation systems. Confirmed priority: compare experiments by answer quality, speed, and measured usage.

## Product Purpose
Compare retrieval configurations on shared documents and reference questions, then inspect answers and retrieved evidence to understand differences.

## Operating Context
A private Streamlit workspace backed by FastAPI. Personal access keys isolate saved work. Benchmarks consume provider quota; planning and viewing saved results do not initiate generation.

## Capabilities and Constraints
Document and reference uploads, two to four benchmark configurations, four Ragas metrics, saved runs, profiling, bounded optimization, retrieval debugging, and investigations. Preserve server-side authentication, isolation, retention, and cost controls. Missing measurements remain unknown; incomplete scores cannot establish a winner. Dollar costs cannot be inferred from request counts.

## Brand Commitments
Keep RAG Bench and the existing lavender, teal, apricot, and muted rose palette. User requests a whole-site overhaul with a cleaner, predominantly visual experience.

## Evidence on Hand
Saved backend benchmark summaries and performance records are authoritative. Demonstration fixtures must be explicitly labelled synthetic and never displayed as live measurements.

## Product Principles
- Lead with comparisons and their evidence.
- Make the next task easy to find.
- Explain missing data instead of inventing it.
- Keep risky and quota-consuming actions explicit.
