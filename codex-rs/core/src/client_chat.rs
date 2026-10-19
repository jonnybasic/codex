//! OpenAI/Gemini/AskSage-compatible `/v1/chat/completions` streaming and synchronous client engine.
//!
//! Supports:
//! - Standard SSE streaming (`supports_streaming = true`)
//! - Synchronous JSON fallback (`supports_streaming = false`, e.g., AskSage)
//! - Custom API key headers (`env_key_header`, e.g., `x-access-tokens`)
//! - Gemini 2.5/3.x `thought_signature` lossless round-trip packing/unpacking via `call_id`
//! - Recursive JSON Schema sanitization for strict OpenAPI 3.0 gateways

use std::collections::BTreeMap;

use codex_http_client::ClientRouteClass;
use codex_login::default_client::ClientRedirectPolicy;
use codex_login::default_client::create_client_for_route;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ToolSpec;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::HeaderName;
use http::HeaderValue;
use serde_json::Value;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::client::ModelClientSession;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;

const THOUGHT_SIG_SEPARATOR: &str = "|sig:";
const CHAT_STREAM_CHANNEL_CAPACITY: usize = 1600;

/// Recursively strips keywords unsupported by Gemini / strict OpenAPI 3.0 gateways
/// (`$schema`, `additionalProperties`) and collapses nullable type arrays (`["string", "null"]` -> `"string"`).
pub(crate) fn sanitize_json_schema(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            map.remove("$schema");
            map.remove("additionalProperties");

            if let Some(Value::Array(types)) = map.get("type") {
                let non_null: Vec<&str> = types
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|t| *t != "null")
                    .collect();
                if let Some(primary) = non_null.first() {
                    map.insert("type".to_string(), Value::String((*primary).to_string()));
                }
            }

            for val in map.values_mut() {
                sanitize_json_schema(val);
            }
        }
        Value::Array(arr) => {
            for val in arr {
                sanitize_json_schema(val);
            }
        }
        _ => {}
    }
}

fn freeform_to_chat_tool(name: &str, description: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": description
                    }
                },
                "required": ["input"]
            }
        }
    })
}

/// Converts Codex `ToolSpec` items into `/v1/chat/completions` function tool definitions
/// with sanitized parameter schemas.
pub(crate) fn convert_tools_to_chat_format(tools: &[ToolSpec]) -> Vec<Value> {
    let mut out = Vec::new();
    for spec in tools {
        match spec {
            ToolSpec::Function(tool) => {
                let mut params = serde_json::to_value(&tool.parameters)
                    .unwrap_or_else(|_| json!({"type": "object", "properties": {}}));
                sanitize_json_schema(&mut params);
                out.push(json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params,
                    }
                }));
            }
            ToolSpec::Namespace(ns) => {
                for ns_tool in &ns.tools {
                    match ns_tool {
                        ResponsesApiNamespaceTool::Function(tool) => {
                            let mut params = serde_json::to_value(&tool.parameters)
                                .unwrap_or_else(|_| json!({"type": "object", "properties": {}}));
                            sanitize_json_schema(&mut params);
                            out.push(json!({
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": params,
                                }
                            }));
                        }
                        ResponsesApiNamespaceTool::Custom(freeform) => {
                            out.push(freeform_to_chat_tool(&freeform.name, &freeform.description));
                        }
                    }
                }
            }
            ToolSpec::Freeform(freeform) => {
                out.push(freeform_to_chat_tool(&freeform.name, &freeform.description));
            }
            ToolSpec::ToolSearch { .. } | ToolSpec::WebSearch { .. } => {}
        }
    }
    out
}

fn append_assistant_tool_call(messages: &mut Vec<Value>, tool_call: Value) {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("assistant")
        && let Some(obj) = last.as_object_mut()
    {
        let calls = obj
            .entry("tool_calls".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(arr) = calls {
            arr.push(tool_call);
            return;
        }
    }
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": [tool_call]
    }));
}

fn build_tool_call_json(call_id: &str, name: &str, arguments: &str) -> Value {
    let (clean_id, thought_sig) = match call_id.split_once(THOUGHT_SIG_SEPARATOR) {
        Some((id, sig)) => (id, Some(sig)),
        None => (call_id, None),
    };

    let mut tool_call = json!({
        "id": clean_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments
        }
    });

    if let Some(sig) = thought_sig
        && !sig.is_empty()
    {
        tool_call["extra_content"] = json!({
            "google": {
                "thought_signature": sig
            }
        });
    }
    tool_call
}

/// Converts Codex `ResponseItem` history into `/v1/chat/completions` messages,
/// unpacking any Gemini `thought_signature` packed inside `call_id`.
pub(crate) fn convert_items_to_chat_messages(
    base_instructions: &str,
    items: &[ResponseItem],
) -> Vec<Value> {
    let mut messages = Vec::new();
    if !base_instructions.trim().is_empty() {
        messages.push(json!({
            "role": "system",
            "content": base_instructions
        }));
    }

    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let mut text_parts = Vec::new();
                for c in content {
                    match c {
                        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                            text_parts.push(text.as_str());
                        }
                        ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => {}
                    }
                }
                let mapped_role = if role == "developer" {
                    "system"
                } else {
                    role.as_str()
                };
                messages.push(json!({
                    "role": mapped_role,
                    "content": text_parts.join("\n")
                }));
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                let tool_call = build_tool_call_json(call_id, name, arguments);
                append_assistant_tool_call(&mut messages, tool_call);
            }
            ResponseItem::FunctionCallOutput {
                call_id: Some(call_id),
                output,
                ..
            } => {
                let clean_id = call_id
                    .split_once(THOUGHT_SIG_SEPARATOR)
                    .map_or(call_id.as_str(), |(id, _)| id);
                let text = output.body.to_text().unwrap_or_else(|| output.to_string());
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": clean_id,
                    "content": text
                }));
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let args = json!({ "input": input }).to_string();
                let tool_call = build_tool_call_json(call_id, name, &args);
                append_assistant_tool_call(&mut messages, tool_call);
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let clean_id = call_id
                    .split_once(THOUGHT_SIG_SEPARATOR)
                    .map_or(call_id.as_str(), |(id, _)| id);
                let text = output.body.to_text().unwrap_or_else(|| output.to_string());
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": clean_id,
                    "content": text
                }));
            }
            _ => {}
        }
    }

    messages
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
    thought_signature: String,
}

fn pack_call_id(id: &str, thought_signature: &str) -> String {
    if thought_signature.is_empty() {
        id.to_string()
    } else {
        format!("{id}{THOUGHT_SIG_SEPARATOR}{thought_signature}")
    }
}

/// Parses a synchronous `/v1/chat/completions` JSON response into `ResponseEvent` items.
pub(crate) fn parse_sync_chat_response(json: &Value) -> Vec<ResponseEvent> {
    let mut events = Vec::new();
    let response_id = json
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("sync-chat-resp")
        .to_string();

    if let Some(choice) = json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        && let Some(msg) = choice.get("message")
    {
        if let Some(reasoning) = msg
            .get("reasoning_content")
            .or_else(|| msg.get("reasoning"))
            .and_then(Value::as_str)
            && !reasoning.is_empty()
        {
            events.push(ResponseEvent::ReasoningContentDelta {
                delta: reasoning.to_string(),
                content_index: 0,
            });
        }

        if let Some(content) = msg.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            events.push(ResponseEvent::OutputTextDelta(content.to_string()));
            events.push(ResponseEvent::OutputItemDone(ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: content.to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            }));
        }

        if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for (idx, tc) in tool_calls.iter().enumerate() {
                let raw_id = tc
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("call_{idx}"));
                let name = tc
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let args = tc
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_string();
                let sig = tc
                    .pointer("/extra_content/google/thought_signature")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let call_id = pack_call_id(&raw_id, sig);

                events.push(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                    id: None,
                    name,
                    namespace: None,
                    arguments: args,
                    encrypted_function_args: None,
                    call_id,
                    internal_chat_message_metadata_passthrough: None,
                }));
            }
        }
    }

    events.push(ResponseEvent::Completed {
        response_id,
        token_usage: None,
        usage_metadata: None,
        end_turn: None,
    });
    events
}

impl ModelClientSession {
    /// Streams or synchronously executes a turn via `/v1/chat/completions`.
    pub(crate) async fn stream_chat_completions(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        effort: Option<ReasoningEffortConfig>,
    ) -> Result<ResponseStream> {
        let provider = self.client.state.provider.info();
        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or("http://127.0.0.1:8080/v1");

        let lower_url = base_url.to_ascii_lowercase();
        if lower_url.contains("openai.com") || lower_url.contains("chatgpt.com") {
            return Err(CodexErr::Fatal(format!(
                "Air-Gap Violation: Outbound traffic to OpenAI/ChatGPT ({base_url}) is strictly prohibited in this build."
            )));
        }

        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let api_provider = provider.to_api_provider(/*auth_mode*/ None)?;
        let mut headers = api_provider.headers;

        let resolved_key = ModelProviderInfo::resolve_api_key(provider)?.or_else(|| {
            provider
                .experimental_bearer_token
                .as_ref()
                .map(|t| t.as_str().to_string())
        });
        if let Some(key) = resolved_key
            && !key.trim().is_empty()
        {
            if let Some(custom_header) = &provider.env_key_header {
                if let (Ok(name), Ok(val)) = (
                    HeaderName::try_from(custom_header.as_str()),
                    HeaderValue::try_from(key.as_str()),
                ) {
                    headers.insert(name, val);
                }
            } else if let Ok(val) = HeaderValue::from_str(&format!("Bearer {key}")) {
                headers.insert(http::header::AUTHORIZATION, val);
            }
        }

        let input = prompt.get_formatted_input_for_request(model_info);
        let messages = convert_items_to_chat_messages(&prompt.base_instructions.text, &input);
        let tools = convert_tools_to_chat_format(&prompt.tools);
        let supports_streaming = provider.supports_streaming;

        let mut payload = json!({
            "model": model_info.slug,
            "messages": messages,
            "stream": supports_streaming,
        });

        if !tools.is_empty() {
            payload["tools"] = Value::Array(tools);
            payload["tool_choice"] = json!("auto");
        }

        if let Some(eff) = effort {
            payload["reasoning_effort"] = json!(eff.to_string().to_lowercase());
        }

        let http_client = create_client_for_route(
            &self.client.http_client_factory,
            &url,
            ClientRouteClass::Api,
            ClientRedirectPolicy::Default,
        )
        .map_err(std::io::Error::from)?;

        let (tx_event, rx_event) = mpsc::channel(CHAT_STREAM_CHANNEL_CAPACITY);
        let consumer_dropped = CancellationToken::new();

        tokio::spawn(async move {
            let resp = match http_client
                .post(&url)
                .headers(headers)
                .json(&payload)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx_event
                        .send(Err(CodexErr::Fatal(format!("Chat request failed: {e}"))))
                        .await;
                    return;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let _ = tx_event
                    .send(Err(CodexErr::Fatal(format!(
                        "Chat API Error ({status}): {body}"
                    ))))
                    .await;
                return;
            }

            let _ = tx_event
                .send(Ok(ResponseEvent::Created { response_id: None }))
                .await;

            if !supports_streaming {
                let text = match resp.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx_event
                            .send(Err(CodexErr::Fatal(format!(
                                "Failed to read sync chat response: {e}"
                            ))))
                            .await;
                        return;
                    }
                };

                let json: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx_event
                            .send(Err(CodexErr::Fatal(format!(
                                "Invalid JSON in sync chat response: {e}"
                            ))))
                            .await;
                        return;
                    }
                };

                for event in parse_sync_chat_response(&json) {
                    let _ = tx_event.send(Ok(event)).await;
                }
                return;
            }

            let mut stream = resp.bytes_stream().eventsource();
            let mut accumulated_text = String::new();
            let mut tool_calls: BTreeMap<usize, AccumulatedToolCall> = BTreeMap::new();
            let mut response_id = "sse-chat-resp".to_string();

            while let Some(event_res) = stream.next().await {
                let event = match event_res {
                    Ok(ev) => ev,
                    Err(e) => {
                        let _ = tx_event
                            .send(Err(CodexErr::Fatal(format!("SSE stream error: {e}"))))
                            .await;
                        return;
                    }
                };

                if event.data.trim() == "[DONE]" {
                    break;
                }

                let Ok(chunk) = serde_json::from_str::<Value>(&event.data) else {
                    continue;
                };

                if let Some(id) = chunk.get("id").and_then(Value::as_str) {
                    response_id = id.to_string();
                }

                if let Some(choice) = chunk
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|c| c.first())
                    && let Some(delta) = choice.get("delta")
                {
                    if let Some(reasoning) = delta
                        .get("reasoning_content")
                        .or_else(|| delta.get("reasoning"))
                        .and_then(Value::as_str)
                        && !reasoning.is_empty()
                    {
                        let _ = tx_event
                            .send(Ok(ResponseEvent::ReasoningContentDelta {
                                delta: reasoning.to_string(),
                                content_index: 0,
                            }))
                            .await;
                    }

                    if let Some(content) = delta.get("content").and_then(Value::as_str)
                        && !content.is_empty()
                    {
                        accumulated_text.push_str(content);
                        let _ = tx_event
                            .send(Ok(ResponseEvent::OutputTextDelta(content.to_string())))
                            .await;
                    }

                    if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                        for tc in tcs {
                            let index =
                                tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                            let entry = tool_calls.entry(index).or_default();

                            if let Some(id) = tc.get("id").and_then(Value::as_str)
                                && !id.is_empty()
                            {
                                entry.id = id.to_string();
                            }
                            if let Some(name) = tc.pointer("/function/name").and_then(Value::as_str)
                            {
                                entry.name.push_str(name);
                            }
                            if let Some(args) =
                                tc.pointer("/function/arguments").and_then(Value::as_str)
                            {
                                entry.arguments.push_str(args);
                            }
                            if let Some(sig) = tc
                                .pointer("/extra_content/google/thought_signature")
                                .and_then(Value::as_str)
                                && !sig.is_empty()
                            {
                                entry.thought_signature = sig.to_string();
                            }
                        }
                    }
                }
            }

            if !accumulated_text.is_empty() {
                let _ = tx_event
                    .send(Ok(ResponseEvent::OutputItemDone(ResponseItem::Message {
                        id: None,
                        role: "assistant".to_string(),
                        content: vec![ContentItem::OutputText {
                            text: accumulated_text,
                        }],
                        phase: None,
                        internal_chat_message_metadata_passthrough: None,
                    })))
                    .await;
            }

            for (idx, tc) in tool_calls {
                let raw_id = if tc.id.is_empty() {
                    format!("call_{idx}")
                } else {
                    tc.id
                };
                let call_id = pack_call_id(&raw_id, &tc.thought_signature);

                let _ = tx_event
                    .send(Ok(ResponseEvent::OutputItemDone(
                        ResponseItem::FunctionCall {
                            id: None,
                            name: tc.name,
                            namespace: None,
                            arguments: tc.arguments,
                            encrypted_function_args: None,
                            call_id,
                            internal_chat_message_metadata_passthrough: None,
                        },
                    )))
                    .await;
            }

            let _ = tx_event
                .send(Ok(ResponseEvent::Completed {
                    response_id,
                    token_usage: None,
                    usage_metadata: None,
                    end_turn: None,
                }))
                .await;
        });

        Ok(ResponseStream {
            rx_event,
            interrupt: None,
            consumer_dropped,
        })
    }
}

#[cfg(test)]
#[path = "client_chat_tests.rs"]
mod tests;
