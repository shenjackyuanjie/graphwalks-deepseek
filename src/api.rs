//! DeepSeek API 客户端：请求/响应的类型定义和 HTTP 调用。

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

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

// ── API 调用 ──────────────────────────────────────────────────────────────

/// 调用 DeepSeek Chat Completions API，返回 (content, reasoning_content, usage)。
pub async fn call_api(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    thinking_effort: Option<&str>,
    prompt: &str,
) -> Result<(String, Option<String>, Option<Usage>)> {
    let (reasoning_effort, thinking) = thinking_effort
        .map(|e| {
            (
                Some(e.to_owned()),
                Some(Thinking {
                    type_: "enabled".to_owned(),
                }),
            )
        })
        .unwrap_or((None, None));

    let req_body = ChatRequest {
        model: model.to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: prompt.to_owned(),
        }],
        temperature: Some(0.0),
        reasoning_effort,
        thinking,
    };

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
