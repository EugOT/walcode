//! OpenAI-compatible embeddings HTTP client.

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct EmbedConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl EmbedConfig {
    pub fn from_env() -> Result<Self, String> {
        let api_key = std::env::var("CLAW_RAG_EMBEDDING_API_KEY")
            .map_err(|_| "set CLAW_RAG_EMBEDDING_API_KEY for embeddings".to_string())?;
        let base_url = std::env::var("CLAW_RAG_EMBEDDING_BASE_URL").map_err(|_| {
            "set CLAW_RAG_EMBEDDING_BASE_URL to an approved local/private embedding endpoint"
                .to_string()
        })?;
        if is_forbidden_provider_base_url(&base_url) {
            return Err(
                "direct provider embedding endpoints are disabled by runtime policy".to_string(),
            );
        }
        let model = std::env::var("CLAW_RAG_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".into());
        Ok(Self {
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
        })
    }

    /// Deterministic fake vectors for tests / dry-run (1536 dims match common `OpenAI` models;
    /// truncated scan still works if dim mismatches — ingest uses same mock for all).
    #[must_use]
    pub fn mock_from_env() -> Option<Self> {
        if std::env::var("CLAW_RAG_MOCK_PROVIDERS").ok().as_deref() != Some("1") {
            return None;
        }
        Some(Self {
            api_key: "mock".into(),
            base_url: "mock://".into(),
            model: "mock-embedding".into(),
        })
    }
}

fn is_forbidden_provider_base_url(base_url: &str) -> bool {
    let lower = base_url.to_ascii_lowercase();
    [
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "aiplatform.googleapis.com",
    ]
    .iter()
    .any(|host| lower.contains(host))
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

pub async fn embed_batch(
    client: &Client,
    cfg: &EmbedConfig,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    if cfg.base_url.starts_with("mock://") {
        return Ok(texts
            .iter()
            .map(|s| mock_vector_for_text(s.as_str()))
            .collect());
    }

    let url = format!("{}/embeddings", cfg.base_url);
    let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let body = EmbeddingsRequest {
        model: &cfg.model,
        input: inputs,
    };
    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", cfg.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        let t = res.text().await.unwrap_or_default();
        return Err(format!("embeddings HTTP error: {t}"));
    }
    let parsed: EmbeddingsResponse = res.json().await.map_err(|e| e.to_string())?;
    if parsed.data.len() != texts.len() {
        return Err(format!(
            "embeddings count mismatch: got {} for {} inputs",
            parsed.data.len(),
            texts.len()
        ));
    }
    Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
}

fn mock_vector_for_text(s: &str) -> Vec<f32> {
    const DIM: usize = 16;
    let mut v = vec![0f32; DIM];
    for (i, b) in s.bytes().enumerate().take(DIM * 4) {
        v[i % DIM] += f32::from(b) / 255.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct EnvGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var_os(key);
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn from_env_requires_explicit_embedding_endpoint() {
        let _lock = env_lock();
        let _key = EnvGuard::set("CLAW_RAG_EMBEDDING_API_KEY", Some("test-key"));
        let _base = EnvGuard::set("CLAW_RAG_EMBEDDING_BASE_URL", None);

        let err = EmbedConfig::from_env().expect_err("base URL should be required");
        assert!(err.contains("CLAW_RAG_EMBEDDING_BASE_URL"), "{err}");
    }

    #[test]
    fn from_env_rejects_direct_provider_endpoint() {
        let _lock = env_lock();
        let _key = EnvGuard::set("CLAW_RAG_EMBEDDING_API_KEY", Some("test-key"));
        let _base = EnvGuard::set(
            "CLAW_RAG_EMBEDDING_BASE_URL",
            Some("https://api.openai.com/v1"),
        );

        let err = EmbedConfig::from_env().expect_err("direct provider URL should be rejected");
        assert!(err.contains("disabled by runtime policy"), "{err}");
    }

    #[test]
    fn from_env_accepts_local_embedding_endpoint() {
        let _lock = env_lock();
        let _key = EnvGuard::set("CLAW_RAG_EMBEDDING_API_KEY", Some("test-key"));
        let _base = EnvGuard::set(
            "CLAW_RAG_EMBEDDING_BASE_URL",
            Some("http://127.0.0.1:11434/v1/"),
        );

        let cfg = EmbedConfig::from_env().expect("local endpoint should be accepted");
        assert_eq!(cfg.base_url, "http://127.0.0.1:11434/v1");
    }
}
