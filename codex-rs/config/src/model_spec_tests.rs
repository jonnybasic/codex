use super::*;
use pretty_assertions::assert_eq;

#[test]
fn test_model_spec_deserialization() {
    let toml_str = r#"
provider = "google"
model = "gemini-2.5-pro-preview"
context_window = 1000000
supports_streaming = true
reasoning_effort = "medium"
wire_api = "chat"

[http_headers]
"X-Goog-Api-Client" = "codex-test"

[variants.high]
reasoning_effort = "high"

[variants.sync]
supports_streaming = false
"#;

    let spec: ModelSpec = toml::from_str(toml_str).expect("should deserialize ModelSpec");
    assert_eq!(spec.provider, "google");
    assert_eq!(spec.model.as_deref(), Some("gemini-2.5-pro-preview"));
    assert_eq!(spec.context_window, Some(1000000));
    assert_eq!(spec.supports_streaming, Some(true));
    assert_eq!(spec.reasoning_effort.as_deref(), Some("medium"));
    assert_eq!(spec.wire_api, Some(WireApi::Chat));
    assert_eq!(spec.variants.len(), 2);
    assert_eq!(
        spec.variants
            .get("high")
            .and_then(|v| v.reasoning_effort.as_deref()),
        Some("high")
    );
    assert_eq!(
        spec.variants.get("sync").and_then(|v| v.supports_streaming),
        Some(false)
    );
}

#[test]
fn test_resolve_effective_model_base_and_variants() {
    let toml_catalog = r#"
[models."gemini-2.5-pro"]
provider = "google"
model = "gemini-2.5-pro-002"
context_window = 1000000
supports_streaming = true
reasoning_effort = "medium"
wire_api = "chat"

[models."gemini-2.5-pro".http_headers]
"X-Base" = "1"

[models."gemini-2.5-pro".variants.high]
reasoning_effort = "high"

[models."gemini-2.5-pro".variants.high.http_headers]
"X-Variant" = "2"

[models."llama-3"]
provider = "local"
model = "llama-3-8b"
"#;

    #[derive(Deserialize)]
    struct Wrapper {
        models: BTreeMap<String, ModelSpec>,
    }

    let wrapper: Wrapper = toml::from_str(toml_catalog).unwrap();
    let catalog = wrapper.models;

    // 1. Resolve base model
    let base_res = resolve_effective_model(&catalog, "gemini-2.5-pro");
    assert_eq!(base_res.identifier, "gemini-2.5-pro");
    assert_eq!(base_res.base_model, "gemini-2.5-pro");
    assert_eq!(base_res.variant, None);
    assert_eq!(base_res.provider, "google");
    assert_eq!(base_res.wire_model, "gemini-2.5-pro-002");
    assert_eq!(base_res.context_window, Some(1000000));
    assert!(base_res.supports_streaming);
    assert_eq!(base_res.reasoning_effort.as_deref(), Some("medium"));
    assert_eq!(base_res.wire_api, WireApi::Chat);
    assert_eq!(
        base_res.http_headers.get("X-Base").map(String::as_str),
        Some("1")
    );
    assert_eq!(base_res.http_headers.get("X-Variant"), None);

    // 2. Resolve variant model
    let var_res = resolve_effective_model(&catalog, "gemini-2.5-pro:high");
    assert_eq!(var_res.identifier, "gemini-2.5-pro:high");
    assert_eq!(var_res.base_model, "gemini-2.5-pro");
    assert_eq!(var_res.variant.as_deref(), Some("high"));
    assert_eq!(var_res.provider, "google");
    assert_eq!(var_res.wire_model, "gemini-2.5-pro-002");
    assert_eq!(var_res.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        var_res.http_headers.get("X-Base").map(String::as_str),
        Some("1")
    );
    assert_eq!(
        var_res.http_headers.get("X-Variant").map(String::as_str),
        Some("2")
    );

    // 3. Fallback for uncatalogued model
    let fallback = resolve_effective_model(&catalog, "custom-model");
    assert_eq!(fallback.identifier, "custom-model");
    assert_eq!(fallback.wire_model, "custom-model");
    assert_eq!(fallback.provider, "default");
}

#[test]
fn test_build_tui_model_catalog_and_variant_cycling() {
    let mut variants = BTreeMap::new();
    variants.insert(
        "high".to_string(),
        ModelVariant {
            name: None,
            model: None,
            max_input_tokens: None,
            max_output_tokens: Some(65536),
            reasoning_effort: Some("high".to_string()),
            supports_streaming: None,
            wire_api: None,
            http_headers: None,
            extra_body: None,
        },
    );
    variants.insert(
        "fast".to_string(),
        ModelVariant {
            name: None,
            model: None,
            max_input_tokens: None,
            max_output_tokens: Some(16384),
            reasoning_effort: Some("low".to_string()),
            supports_streaming: None,
            wire_api: None,
            http_headers: None,
            extra_body: None,
        },
    );

    let mut catalog = BTreeMap::new();
    catalog.insert(
        "gemini-2.5-pro".to_string(),
        ModelSpec {
            name: Some("Gemini 2.5 Pro".to_string()),
            provider: "google".to_string(),
            model: None,
            context_window: Some(1000000),
            max_input_tokens: Some(1048576),
            max_output_tokens: Some(65536),
            supports_streaming: Some(true),
            reasoning_effort: Some("medium".to_string()),
            wire_api: Some(WireApi::Chat),
            http_headers: None,
            extra_body: None,
            variants,
        },
    );
    catalog.insert(
        "gemma-4-12b".to_string(),
        ModelSpec {
            name: Some("Gemma 4 12B".to_string()),
            provider: "local".to_string(),
            model: None,
            context_window: Some(128000),
            max_input_tokens: None,
            max_output_tokens: None,
            supports_streaming: Some(true),
            reasoning_effort: None,
            wire_api: Some(WireApi::Chat),
            http_headers: None,
            extra_body: None,
            variants: BTreeMap::new(),
        },
    );

    let tui_entries = build_tui_model_catalog(&catalog);
    assert_eq!(tui_entries.len(), 4); // gemini-2.5-pro, gemini-2.5-pro:fast, gemini-2.5-pro:high, gemma-4-12b

    // Resolve by reasoning_effort alias (`:low` matches `.variants.fast`)
    let resolved_low = resolve_effective_model(&catalog, "gemini-2.5-pro:low");
    assert_eq!(resolved_low.identifier, "gemini-2.5-pro:fast");
    assert_eq!(resolved_low.reasoning_effort.as_deref(), Some("low"));

    // Test cycling for gemini-2.5-pro (BTreeMap keys: "fast", "high"):
    // Base -> fast -> high -> Base
    let cycle1 =
        next_variant_for_model(&catalog, "gemini-2.5-pro").expect("should cycle to first variant");
    assert_eq!(cycle1.id, "gemini-2.5-pro:fast");

    let cycle2 = next_variant_for_model(&catalog, "gemini-2.5-pro:fast")
        .expect("should cycle to second variant");
    assert_eq!(cycle2.id, "gemini-2.5-pro:high");

    let cycle3 = next_variant_for_model(&catalog, "gemini-2.5-pro:high")
        .expect("should wrap around to base");
    assert_eq!(cycle3.id, "gemini-2.5-pro");

    // Test cycling when session holds wire model + current effort ("low" -> fast -> next is high)
    let cycle_by_effort =
        next_variant_for_model_and_effort(&catalog, "gemini-2.5-pro", Some("low"))
            .expect("should cycle from fast(low) to high");
    assert_eq!(cycle_by_effort.id, "gemini-2.5-pro:high");

    // Test model without variants
    assert_eq!(next_variant_for_model(&catalog, "gemma-4-12b"), None);
}
