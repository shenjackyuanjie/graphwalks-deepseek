//! DeepSeek API 客户端：请求/响应的类型定义和 HTTP 调用（含 streaming）。

use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

// ── 请求类型 ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Serialize)]
pub struct Thinking {
    #[serde(rename = "type")]
    pub type_: String,
}

#[derive(Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

// ── 响应类型 ──────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    #[serde(default)]
    pub prompt_cache_hit_tokens: u32,
    #[serde(default)]
    pub prompt_cache_miss_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
pub struct ChoiceMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

// ── Streaming 进度通知 ───────────────────────────────────────────────────

/// 每次收到 SSE delta 时发送的进度通知。
#[derive(Debug, Clone)]
pub struct StreamTick {
    pub sample_index: usize,
    /// 本次 content 增量的字符数（不含 reasoning）。
    pub content_delta_chars: usize,
    /// 本次 reasoning 增量的字符数。
    pub reasoning_delta_chars: usize,
}

// ── 非 streaming API 调用 ────────────────────────────────────────────────

/// 调用 DeepSeek Chat Completions API，返回 (content, reasoning_content, usage)。
pub async fn call_api(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    thinking_effort: Option<&str>,
    prompt: &str,
) -> Result<(String, Option<String>, Option<Usage>)> {
    let (reasoning_effort, thinking) = build_thinking(thinking_effort);
    let req_body = build_request(model, prompt, reasoning_effort, thinking, false);

    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&req_body)
        .send()
        .await
        .context("HTTP 请求失败")?;

    let status = resp.status();
    let body_text = resp.text().await.context("读取响应内容失败")?;

    if !status.is_success() {
        return Err(anyhow!("API 返回错误 {status}: {body_text}"));
    }

    let chat_resp: ChatResponse =
        serde_json::from_str(&body_text).context("解析 API 响应失败")?;

    let choice = chat_resp
        .choices
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("API 返回了空的 choices"));

    Ok((
        choice.message.content.unwrap_or_default(),
        choice.message.reasoning_content,
        chat_resp.usage,
    ))
}

// ── Streaming API 调用 ───────────────────────────────────────────────────

/// 调用 DeepSeek Chat Completions API（streaming 模式），
/// 每收到一个 delta 就通过 `tick_tx` 发送进度通知。
pub async fn call_api_streaming(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    thinking_effort: Option<&str>,
    prompt: &str,
    sample_index: usize,
    tick_tx: &mpsc::UnboundedSender<StreamTick>,
) -> Result<(String, Option<String>, Option<Usage>)> {
    let (reasoning_effort, thinking) = build_thinking(thinking_effort);
    let req_body = build_request(model, prompt, reasoning_effort, thinking, true);

    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&req_body)
        .send()
        .await
        .context("HTTP 请求失败")?;

    let status = resp.status();
    if !status.is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("API 返回错误 {status}: {body_text}"));
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage: Option<Usage> = None;
    let mut line_buf = String::new();

    let mut byte_stream = resp.bytes_stream();
    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("读取 streaming chunk 失败")?;
        line_buf.push_str(&String::from_utf8_lossy(&chunk));

        // 逐行解析 SSE
        while let Some(line_end) = line_buf.find('\n') {
            let line = line_buf[..line_end].trim_end_matches('\r').to_owned();
            line_buf = line_buf[line_end + 1..].to_owned();

            let data = match line.strip_prefix("data: ") {
                Some(d) => d,
                None => continue,
            };

            if data == "[DONE]" {
                break;
            }

            let v: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // 提取 delta
            if let Some(choices) = v["choices"].as_array() {
                for choice in choices {
                    if let Some(delta) = choice.get("delta") {
                        let c = delta["content"].as_str().unwrap_or("");
                        let r = delta["reasoning_content"].as_str().unwrap_or("");

                        if !c.is_empty() || !r.is_empty() {
                            let _ = tick_tx.send(StreamTick {
                                sample_index,
                                content_delta_chars: c.chars().count(),
                                reasoning_delta_chars: r.chars().count(),
                            });
                            if !c.is_empty() {
                                content.push_str(c);
                            }
                            if !r.is_empty() {
                                reasoning.push_str(r);
                            }
                        }
                    }
                }
            }

            // 最后一个 chunk 通常包含 usage
            if let Some(u) = v.get("usage") {
                if u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) > 0 {
                    usage = Some(serde_json::from_value(u.clone()).unwrap_or_else(|_| {
                        panic!("解析 streaming usage 失败")
                    }));
                }
            }
        }
    }

    Ok((
        content,
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        usage,
    ))
}

// ── 内部辅助 ──────────────────────────────────────────────────────────────

fn build_thinking(thinking_effort: Option<&str>) -> (Option<String>, Option<Thinking>) {
    thinking_effort
        .map(|e| {
            (
                Some(e.to_owned()),
                Some(Thinking {
                    type_: "enabled".to_owned(),
                }),
            )
        })
        .unwrap_or((None, None))
}

fn build_request(
    model: &str,
    prompt: &str,
    reasoning_effort: Option<String>,
    thinking: Option<Thinking>,
    stream: bool,
) -> ChatRequest {
    ChatRequest {
        model: model.to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: prompt.to_owned(),
        }],
        temperature: Some(0.0),
        reasoning_effort,
        thinking,
        stream: if stream { Some(true) } else { None },
    }
}
