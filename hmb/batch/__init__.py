"""Amazon Bedrock batch inference as a second transport for `hmb.providers.call_model` (2026-09-26).

A batch job answers exactly the requests `call_model` would have sent synchronously: the record body is
the synchronous Converse request (`providers.converse_request`), and every answered record is written as
the ordinary cache entry of that request (same cache key, same `ModelResponse`), with the ledger settled
at the dated batch-lane rate. Callers are then replayed from the cache, so prompts, parsers, scoring and
gates never see which transport answered. Only routes whose capability is `SUPPORTED_LIVE_COMPLETED` in
the versioned matrix (`capability_matrix.json`) are batched; every other call stays synchronous.

    capability.py  versioned capability matrix and fail-closed route selection
    jobs.py        batch config, record building, S3/Bedrock job backend, job manifests, ingestion
    scheduler.py   router installed into call_model, dependency-aware hybrid scheduler, policies
    probe.py       minimal live capability probes (engineering evidence, separate ledger phase)
    projection.py  granular cost projection (on-demand vs hybrid) from ledgers and the rate card
    cli.py         batch-plan / batch-run / batch-status / batch-collect / batch-cancel / batch-cost
"""
