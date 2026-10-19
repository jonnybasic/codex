use codex_model_provider_info::WireApi;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::collections::HashMap;

/// OpenCode-compatible model specification.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ModelSpec {
    /// Optional human-friendly display name for the model.
    #[serde(default)]
    pub name: Option<String>,
    /// Provider ID key in `model_providers`
    pub provider: String,
    /// Upstream wire model name sent to the provider. Defaults to the table key if omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Size of the context window in tokens (e.g. 1000000 for 1M).
    #[serde(default)]
    pub context_window: Option<u64>,
    /// Maximum input tokens supported by the model.
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens supported by the model.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Whether the model supports streaming SSE. Defaults to true.
    #[serde(default)]
    pub supports_streaming: Option<bool>,
    /// Reasoning effort level (e.g. "low", "medium", "high").
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Wire protocol (WireApi::Responses or WireApi::Chat).
    #[serde(default)]
    pub wire_api: Option<WireApi>,
    /// Additional HTTP headers to send with inference requests.
    #[serde(default)]
    pub http_headers: Option<HashMap<String, String>>,
    /// Optional extra JSON body fields for inference requests.
    #[serde(default)]
    pub extra_body: Option<BTreeMap<String, serde_json::Value>>,
    /// Variants of this model (e.g. [models."gemini-2.5-pro".variants.high]).
    #[serde(default)]
    pub variants: BTreeMap<String, ModelVariant>,
}

/// A variant configuration for a specific model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ModelVariant {
    /// Optional human-friendly display name for this variant.
    #[serde(default)]
    pub name: Option<String>,
    /// Upstream wire model override for this variant.
    #[serde(default)]
    pub model: Option<String>,
    /// Maximum input tokens override for this variant.
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens override for this variant.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Reasoning effort override for this variant.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Streaming override for this variant.
    #[serde(default)]
    pub supports_streaming: Option<bool>,
    /// Wire protocol override for this variant.
    #[serde(default)]
    pub wire_api: Option<WireApi>,
    /// Additional HTTP headers for this variant.
    #[serde(default)]
    pub http_headers: Option<HashMap<String, String>>,
    /// Optional extra JSON body fields for this variant.
    #[serde(default)]
    pub extra_body: Option<BTreeMap<String, serde_json::Value>>,
}

/// Fully resolved model configuration after evaluating base model and variant overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelConfig {
    /// Canonical identifier requested by user (e.g. "gemini-2.5-pro:high" or "gemma-4-12b")
    pub identifier: String,
    /// Base model name (e.g. "gemini-2.5-pro")
    pub base_model: String,
    /// Active variant name if any (e.g. Some("high"))
    pub variant: Option<String>,
    /// Target provider ID
    pub provider: String,
    /// Wire model sent to API
    pub wire_model: String,
    /// Context window limit
    pub context_window: Option<u64>,
    /// Streaming enabled
    pub supports_streaming: bool,
    /// Reasoning effort if set
    pub reasoning_effort: Option<String>,
    /// Wire protocol
    pub wire_api: WireApi,
    /// Merged HTTP headers
    pub http_headers: HashMap<String, String>,
}

/// A formatted model entry for the TUI model picker and variant switcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiModelEntry {
    /// Full model identifier (e.g. "gemini-2.5-pro:high" or "local-gemma")
    pub id: String,
    /// Base model key
    pub base_model: String,
    /// Variant name if any
    pub variant: Option<String>,
    /// Provider ID
    pub provider: String,
    /// Formatted display label for menus
    pub display_label: String,
    /// Context window tokens
    pub context_window: Option<u64>,
    /// Effective reasoning effort level if configured
    pub reasoning_effort: Option<String>,
    /// Streaming enabled
    pub supports_streaming: bool,
    /// Wire protocol
    pub wire_api: WireApi,
}

impl ResolvedModelConfig {
    /// Returns the effective model string to store in session state (the canonical identifier).
    pub fn session_model_name(&self) -> &str {
        &self.identifier
    }
}

/// Resolves an effective model configuration from the catalog given a raw model string.
pub fn resolve_effective_model(
    catalog: &BTreeMap<String, ModelSpec>,
    raw_model: &str,
) -> ResolvedModelConfig {
    if let Some((base, variant)) = raw_model.split_once(':')
        && let Some(spec) = catalog.get(base)
    {
        let matched_variant = spec.variants.get_key_value(variant).or_else(|| {
            spec.variants
                .iter()
                .find(|(_, v)| v.reasoning_effort.as_deref() == Some(variant))
        });

        if let Some((matched_name, var)) = matched_variant {
            let wire_model = var
                .model
                .as_deref()
                .or(spec.model.as_deref())
                .unwrap_or(base)
                .to_string();
            let reasoning_effort = var
                .reasoning_effort
                .clone()
                .or_else(|| spec.reasoning_effort.clone());
            let supports_streaming = var
                .supports_streaming
                .or(spec.supports_streaming)
                .unwrap_or(true);
            let wire_api = var.wire_api.or(spec.wire_api).unwrap_or(WireApi::Chat);
            let mut http_headers = spec.http_headers.clone().unwrap_or_default();
            if let Some(var_headers) = &var.http_headers {
                http_headers.extend(var_headers.clone());
            }

            return ResolvedModelConfig {
                identifier: format!("{base}:{matched_name}"),
                base_model: base.to_string(),
                variant: Some(matched_name.clone()),
                provider: spec.provider.clone(),
                wire_model,
                context_window: spec.context_window,
                supports_streaming,
                reasoning_effort,
                wire_api,
                http_headers,
            };
        }

        // Ad-hoc `:effort` suffix on a known base model (e.g. `gemini-2.5-flash:low`)
        let wire_model = spec.model.as_deref().unwrap_or(base).to_string();
        let supports_streaming = spec.supports_streaming.unwrap_or(true);
        let wire_api = spec.wire_api.unwrap_or(WireApi::Chat);
        let http_headers = spec.http_headers.clone().unwrap_or_default();

        return ResolvedModelConfig {
            identifier: raw_model.to_string(),
            base_model: base.to_string(),
            variant: Some(variant.to_string()),
            provider: spec.provider.clone(),
            wire_model,
            context_window: spec.context_window,
            supports_streaming,
            reasoning_effort: Some(variant.to_string()),
            wire_api,
            http_headers,
        };
    }

    if let Some(spec) = catalog.get(raw_model) {
        let wire_model = spec.model.as_deref().unwrap_or(raw_model).to_string();
        let reasoning_effort = spec.reasoning_effort.clone();
        let supports_streaming = spec.supports_streaming.unwrap_or(true);
        let wire_api = spec.wire_api.unwrap_or(WireApi::Chat);
        let http_headers = spec.http_headers.clone().unwrap_or_default();

        return ResolvedModelConfig {
            identifier: raw_model.to_string(),
            base_model: raw_model.to_string(),
            variant: None,
            provider: spec.provider.clone(),
            wire_model,
            context_window: spec.context_window,
            supports_streaming,
            reasoning_effort,
            wire_api,
            http_headers,
        };
    }

    // Fallback for ad-hoc models not in the catalog
    ResolvedModelConfig {
        identifier: raw_model.to_string(),
        base_model: raw_model.to_string(),
        variant: None,
        provider: "default".to_string(),
        wire_model: raw_model.to_string(),
        context_window: None,
        supports_streaming: true,
        reasoning_effort: None,
        wire_api: WireApi::Chat,
        http_headers: HashMap::new(),
    }
}

/// Builds the TUI model catalog entries from the `[models]` configuration table.
pub fn build_tui_model_catalog(catalog: &BTreeMap<String, ModelSpec>) -> Vec<TuiModelEntry> {
    let mut entries = Vec::new();

    for (name, spec) in catalog {
        let ctx_str = spec
            .context_window
            .map_or_else(String::new, |ctx| format!(" {}k ctx", ctx / 1000));
        let stream_str = if spec.supports_streaming.unwrap_or(true) {
            "SSE"
        } else {
            "Sync"
        };
        let effort_str = spec
            .reasoning_effort
            .as_deref()
            .map_or_else(String::new, |e| format!(" | effort: {e}"));
        let p = &spec.provider;
        let base_title = match spec.name.as_deref() {
            Some(friendly) => format!("{friendly} ({name})"),
            None => name.clone(),
        };
        let display_label = format!("{base_title} [{p}]{ctx_str} | {stream_str}{effort_str}");

        entries.push(TuiModelEntry {
            id: name.clone(),
            base_model: name.clone(),
            variant: None,
            provider: spec.provider.clone(),
            display_label,
            context_window: spec.context_window,
            reasoning_effort: spec.reasoning_effort.clone(),
            supports_streaming: spec.supports_streaming.unwrap_or(true),
            wire_api: spec.wire_api.unwrap_or(WireApi::Chat),
        });

        for (variant_name, var) in &spec.variants {
            let var_id = format!("{name}:{variant_name}");
            let var_streaming = var
                .supports_streaming
                .or(spec.supports_streaming)
                .unwrap_or(true);
            let var_stream_str = if var_streaming { "SSE" } else { "Sync" };
            let var_effort = var
                .reasoning_effort
                .as_deref()
                .or(spec.reasoning_effort.as_deref());
            let var_effort_str = var_effort.map_or_else(String::new, |e| format!(" | effort: {e}"));
            let var_title = match var.name.as_deref().or(spec.name.as_deref()) {
                Some(friendly) => format!("{friendly} ({var_id})"),
                None => var_id.clone(),
            };
            let var_label =
                format!("{var_title} [{p}]{ctx_str} | {var_stream_str}{var_effort_str}");

            entries.push(TuiModelEntry {
                id: var_id,
                base_model: name.clone(),
                variant: Some(variant_name.clone()),
                provider: spec.provider.clone(),
                display_label: var_label,
                context_window: spec.context_window,
                reasoning_effort: var_effort.map(str::to_string),
                supports_streaming: var_streaming,
                wire_api: var.wire_api.or(spec.wire_api).unwrap_or(WireApi::Chat),
            });
        }
    }

    entries
}

/// Finds the next variant for the currently active model (powering Ctrl+T cycling).
pub fn next_variant_for_model(
    catalog: &BTreeMap<String, ModelSpec>,
    current_model: &str,
) -> Option<TuiModelEntry> {
    next_variant_for_model_and_effort(catalog, current_model, /*current_effort*/ None)
}

/// Finds the next variant for the currently active model and reasoning effort.
pub fn next_variant_for_model_and_effort(
    catalog: &BTreeMap<String, ModelSpec>,
    current_model: &str,
    current_effort: Option<&str>,
) -> Option<TuiModelEntry> {
    let raw_base = current_model.split(':').next().unwrap_or(current_model);
    let (base_name, spec) = catalog
        .get_key_value(raw_base)
        .map(|(k, v)| (k.as_str(), v))
        .or_else(|| {
            catalog
                .iter()
                .find(|(_, s)| s.model.as_deref() == Some(raw_base))
                .map(|(k, v)| (k.as_str(), v))
        })?;
    if spec.variants.is_empty() {
        return None;
    }

    // Sequence of variants: base (None) followed by each named variant in order
    let mut variant_keys: Vec<Option<&str>> = vec![None];
    variant_keys.extend(spec.variants.keys().map(|k| Some(k.as_str())));

    let current_variant = current_model.split_once(':').map(|(_, v)| v);
    let current_idx = if let Some(var_suffix) = current_variant {
        variant_keys
            .iter()
            .position(|&v| v == Some(var_suffix))
            .or_else(|| {
                spec.variants
                    .iter()
                    .position(|(_, var)| var.reasoning_effort.as_deref() == Some(var_suffix))
                    .map(|idx| idx + 1)
            })
            .unwrap_or(0)
    } else if let Some(eff) = current_effort {
        if spec.reasoning_effort.as_deref() == Some(eff) {
            0
        } else {
            spec.variants
                .iter()
                .position(|(_, var)| var.reasoning_effort.as_deref() == Some(eff))
                .map(|idx| idx + 1)
                .unwrap_or(0)
        }
    } else {
        0
    };

    let next_idx = (current_idx + 1) % variant_keys.len();
    let next_variant = variant_keys[next_idx];

    let full_catalog = build_tui_model_catalog(catalog);
    let target_id = match next_variant {
        Some(v) => format!("{base_name}:{v}"),
        None => base_name.to_string(),
    };

    full_catalog.into_iter().find(|e| e.id == target_id)
}

#[cfg(test)]
#[path = "model_spec_tests.rs"]
mod tests;
