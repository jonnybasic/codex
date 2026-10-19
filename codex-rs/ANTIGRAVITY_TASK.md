# ANTIGRAVITY TASK SPEC: Enterprise Air-Gapped Multi-Provider & OpenCode TUI Fork (`release/0.160`)

## Objective
Transform `codex-rs` (`release/0.160`) into an air-gapped, enterprise-safe binary with zero outbound dependencies on `openai.com` or `chatgpt.com`, full support for both `WireApi::Responses` and `WireApi::Chat`, first-party compatibility with local `llama.cpp`/`vLLM`, Google Cloud Gemini IL5 (`thought_signature` preservation + schema sanitization), and AskSage (`x-access-tokens` custom header + `stream: false` synchronous JSON fallback), plus OpenCode TUI parity (`/model` populated from `config.toml` and `Ctrl+T` live variant cycling).

## Strict Architectural Invariants
1. **Zero OpenAI Cloud Leaks**: Purge hardcoded `api.openai.com` / `chatgpt.com` default providers, disable OTLP network exporters in `otel_init.rs`, and disable self-update in `cli/src/main.rs`.
2. **Dual Wire Protocol**: Restore `WireApi::Chat` alongside `WireApi::Responses` in `codex-model-provider-info`.
3. **Gemini IL5 Compatibility**: Preserve `extra_content.google.thought_signature` across tool call turns by packing/unpacking `|sig:<token>` inside `ResponseItem::FunctionCall.call_id`, and recursively strip `$schema`, `additionalProperties`, and nullable array union types (`["string", "null"]`) from tool JSON schemas in `WireApi::Chat`.
4. **AskSage & Sync Fallback**: Support `env_key_header` (e.g. `"x-access-tokens"`), `http_headers`, and `supports_streaming = false` (converting synchronous JSON `/chat/completions` responses into the internal `ResponseEvent` stream).
5. **OpenCode Variants & Live TUI Hot-Swap**: Support `[models."<name>"]` and `[models."<name>".variants.<variant>]` in `config.toml`, resolve `<model>:<variant>` strings, wire them into the TUI `/model` picker, and bind `Ctrl+T` to cycle variants mid-session via `Op::OverrideTurnContext`.

## Execution Order (Verify with `cargo check` after each phase)
- [ ] **Phase 1**: `codex-rs/model-provider-info/src/lib.rs` (`cargo check -p codex-model-provider-info`)
- [ ] **Phase 2**: `codex-rs/core/src/otel_init.rs`, `codex-rs/core/src/auth.rs`, `codex-rs/cli/src/main.rs`
- [ ] **Phase 3**: `codex-rs/core/src/config/mod.rs` (Model & Variant registry + TUI catalog builder)
- [ ] **Phase 4**: `codex-rs/core/src/client.rs` (Dual `WireApi::Chat` SSE + Sync engine, Gemini `thought_signature`, Schema Sanitizer)
- [ ] **Phase 5**: `codex-rs/protocol/src/protocol.rs` & `codex-rs/core/src/codex.rs` (`Op::OverrideTurnContext` cross-provider hot-swap)
- [ ] **Phase 6**: `codex-rs/tui/src/chatwidget.rs` & status bar (`Ctrl+T` variant cycler + dynamic `/model` modal)