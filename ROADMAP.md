# RAG Solution Roadmap

## Current Baseline

The project already includes:

- PDF ingestion from a local folder
- chunking and OpenAI embeddings
- local ChromaDB vector storage
- hybrid retrieval with vector search and keyword scoring
- a Gradio chat interface
- local reset and reindex commands

## Near-Term Priorities

These are the highest-leverage upgrades for turning the current system into a more complete RAG solution:

1. OCR and better PDF parsing
2. metadata extraction and filterable retrieval
3. reranking after initial retrieval
4. evaluation and observability
5. admin controls in the UI

## Phase 1: Retrieval Quality

### OCR and document parsing

- add OCR support for scanned PDFs
- improve table extraction
- handle images and captions where useful
- parse sections and headings instead of relying only on flat page text

### Rich metadata

- extract document title, author, date, tags, and section headers
- support metadata filters such as document type, date range, and source folder
- allow retrieval constrained to a selected subset of the corpus

### Smarter chunking

- move from basic fixed-size chunking to heading-aware or semantic chunking
- support parent-child chunk retrieval
- improve chunking for tables, lists, and structured sections

### Reranking

- add a second-stage reranker after Chroma and keyword retrieval
- prioritize highest-relevance passages before generation
- reduce noisy context sent to the model

### Query enhancement

- rewrite ambiguous user questions before retrieval
- support multi-query retrieval for broader recall
- add optional query expansion for difficult searches

## Phase 2: User Experience and Answer Quality

### Stronger citations

- show exact quoted spans, not just chunk previews
- provide clickable document and page references
- add a “why this answer” explanation panel

### Answer controls

- add confidence or evidence-strength indicators
- support strict grounded mode
- return “insufficient evidence” when retrieval is weak
- add structured answer templates for common question types

### Conversation handling

- improve follow-up question resolution
- maintain cleaner chat memory
- prevent old conversation turns from degrading retrieval quality

### Admin UI

- inspect indexed files and chunk counts
- reindex or delete one document at a time
- inspect retrieved chunks for a question
- tune retrieval settings from the interface

### Background indexing

- watch the PDF folder for changes
- queue and process indexing jobs asynchronously
- expose indexing progress and failures in the UI

## Phase 3: Reliability, Operations, and Cost

### Evaluation suite

- create benchmark questions and expected source documents
- measure retrieval hit rate and answer quality
- add regression checks when prompts, chunking, or models change

### LangSmith evaluation and agentic verification

- add LangSmith tracing for retrieval, context assembly, and final answer generation
- build curated evaluation datasets from real document questions and expected evidence
- use LLM-as-judge evaluations for:
  - answer relevance to the user question
  - groundedness against retrieved chunks
  - factual accuracy relative to retrieved evidence or reference answers
- compare experiments across chunking strategies, prompts, models, and retrieval settings
- add asynchronous post-response evaluation for production traces
- route weak or failed answers into human review workflows
- optionally add a higher-cost inline verification pass for sensitive or low-confidence queries
- avoid code-evaluation scope here; focus only on RAG answer quality and evidence support

### Observability

- log retrieved chunks per query
- track latency, token usage, and model costs
- add trace views for retrieval and generation steps

### Cost controls

- cache embeddings and responses where safe
- deduplicate near-identical chunks
- reduce prompt size with smarter context packing
- route tasks to cheaper or stronger models as appropriate

### Security

- improve API key management
- add authentication if the app becomes multi-user
- support access control by collection or document
- add audit logging and optional PII redaction

## Phase 4: Product-Level Expansion

### Multi-collection support

- separate knowledge bases by team, project, or document class
- allow the user to select a corpus at query time
- support collection-specific prompts and retrieval settings

### Feedback loops

- collect user feedback on answer quality
- flag low-confidence or low-satisfaction responses
- use feedback to tune prompts, retrieval settings, and evaluation sets

### Advanced workflows

- support structured extraction tasks in addition to chat
- add summarization and comparison modes across documents
- support report generation backed by retrieved evidence

## Suggested Implementation Order

1. OCR and improved parsing
2. metadata extraction and retrieval filters
3. reranking
4. stronger citations
5. evaluation suite
6. observability
7. LangSmith evaluation and agentic verification
8. admin UI
9. background indexing
10. cost controls
11. security and multi-collection support
