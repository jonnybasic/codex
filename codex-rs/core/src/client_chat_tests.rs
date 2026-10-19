use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::convert_items_to_chat_messages;
use super::parse_sync_chat_response;
use super::sanitize_json_schema;
use crate::client_common::ResponseEvent;

#[test]
fn test_sanitize_json_schema_strips_forbidden_keys_and_collapses_nullable_unions() {
    let mut raw_schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "command": {
                "type": ["string", "null"],
                "description": "The command to execute"
            },
            "nested": {
                "type": "object",
                "additionalProperties": true,
                "properties": {
                    "timeout": {
                        "type": ["null", "integer"]
                    }
                }
            }
        }
    });

    sanitize_json_schema(&mut raw_schema);

    assert_eq!(
        raw_schema,
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute"
                },
                "nested": {
                    "type": "object",
                    "properties": {
                        "timeout": {
                            "type": "integer"
                        }
                    }
                }
            }
        })
    );
}

#[test]
fn test_thought_signature_round_trip_in_chat_messages() {
    let items = vec![
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Run ls".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCall {
            id: None,
            name: "shell".to_string(),
            namespace: None,
            arguments: r#"{"cmd":"ls"}"#.to_string(),
            encrypted_function_args: None,
            call_id: "call_42|sig:gemini_encrypted_sig_xyz".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some("call_42|sig:gemini_encrypted_sig_xyz".to_string()),
            name: Some("shell".to_string()),
            namespace: None,
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("file1.txt\nfile2.txt".to_string()),
                success: Some(true),
            },
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    let messages =
        convert_items_to_chat_messages("You are an air-gapped coding assistant.", &items);

    assert_eq!(
        messages,
        vec![
            json!({
                "role": "system",
                "content": "You are an air-gapped coding assistant."
            }),
            json!({
                "role": "user",
                "content": "Run ls"
            }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": r#"{"cmd":"ls"}"#
                        },
                        "extra_content": {
                            "google": {
                                "thought_signature": "gemini_encrypted_sig_xyz"
                            }
                        }
                    }
                ]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_42",
                "content": "file1.txt\nfile2.txt"
            })
        ]
    );
}

#[test]
fn test_parse_sync_chat_response_packs_thought_signature_and_emits_events() {
    let sync_json = json!({
        "id": "asksage-sync-001",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Checking directory contents first.",
                    "content": "Running shell command.",
                    "tool_calls": [
                        {
                            "id": "call_99",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": r#"{"cmd":"pwd"}"#
                            },
                            "extra_content": {
                                "google": {
                                    "thought_signature": "sig_abc_123"
                                }
                            }
                        }
                    ]
                }
            }
        ]
    });

    let events = parse_sync_chat_response(&sync_json);
    assert_eq!(events.len(), 5);

    match &events[0] {
        ResponseEvent::ReasoningContentDelta {
            delta,
            content_index,
        } => {
            assert_eq!(delta, "Checking directory contents first.");
            assert_eq!(*content_index, 0);
        }
        other => panic!("Expected ReasoningContentDelta, got {other:?}"),
    }

    match &events[1] {
        ResponseEvent::OutputTextDelta(text) => {
            assert_eq!(text, "Running shell command.");
        }
        other => panic!("Expected OutputTextDelta, got {other:?}"),
    }

    match &events[3] {
        ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        }) => {
            assert_eq!(name, "shell");
            assert_eq!(arguments, r#"{"cmd":"pwd"}"#);
            assert_eq!(call_id, "call_99|sig:sig_abc_123");
        }
        other => panic!("Expected FunctionCall OutputItemDone, got {other:?}"),
    }

    match &events[4] {
        ResponseEvent::Completed { response_id, .. } => {
            assert_eq!(response_id, "asksage-sync-001");
        }
        other => panic!("Expected Completed, got {other:?}"),
    }
}
