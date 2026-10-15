-- FinRAG schema. Idempotent: safe to run on every startup.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filings (
    accession_number text PRIMARY KEY,
    ticker           text NOT NULL,
    company          text NOT NULL,
    form             text NOT NULL,
    filing_date      date NOT NULL,
    period_end       date NOT NULL,
    fiscal_year      int  NOT NULL,
    fiscal_quarter   int,             -- NULL for 10-K
    url              text NOT NULL
);

-- One row per (chunk, embedding model). Several chunking strategies and models
-- live side by side so experiments can be compared on the same database.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         text NOT NULL,   -- "{accession}:{strategy}:{n}"
    embed_model      text NOT NULL,   -- e.g. "voyage-4"
    strategy         text NOT NULL,   -- e.g. "section-512"
    accession_number text NOT NULL REFERENCES filings ON DELETE CASCADE,
    ticker           text NOT NULL,
    form             text NOT NULL,
    fiscal_year      int  NOT NULL,
    fiscal_quarter   int,
    section_key      text NOT NULL,
    section_title    text NOT NULL,
    text             text NOT NULL,   -- shown to the LLM and cited
    embed_text       text NOT NULL,   -- what was embedded (may include a header)
    n_tokens         int  NOT NULL,
    -- halfvec stores 16-bit floats: half the space of vector, which matters on
    -- Neon's free tier. The retrieval-quality difference is negligible.
    embedding        halfvec(1024) NOT NULL,
    -- Full-text index for BM25-style hybrid search (Week 2).
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED,
    PRIMARY KEY (chunk_id, embed_model)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_filters
    ON chunks (strategy, embed_model, ticker, fiscal_year, form);

-- User feedback on answers (thumbs down / "Report an issue").
CREATE TABLE IF NOT EXISTS feedback (
    id          bigserial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    rating      smallint NOT NULL CHECK (rating IN (-1, 1)),
    issue_type  text,                -- e.g. "incorrect_answer"; NULL for a thumbs up
    question    text NOT NULL,
    answer      text,
    details     text,
    citations   jsonb,               -- what was cited, for debugging
    model       text,
    client_hash text                 -- salted hash of the IP: abuse control without storing IPs
);
CREATE INDEX IF NOT EXISTS feedback_created ON feedback (created_at DESC);

-- Bookkeeping for scheduled jobs (e.g. the daily new-filing check).
CREATE TABLE IF NOT EXISTS job_runs (
    id          bigserial PRIMARY KEY,
    job         text NOT NULL,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    ok          boolean,
    summary     jsonb
);
CREATE INDEX IF NOT EXISTS job_runs_job ON job_runs (job, started_at DESC);
