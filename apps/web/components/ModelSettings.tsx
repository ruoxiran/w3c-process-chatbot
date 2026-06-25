"use client";

import { FormEvent, useEffect, useState } from "react";
import type { ProviderOverride } from "@/lib/api";

const STORAGE_KEY = "w3c-process.provider-override";

export type ProviderChoice = "default" | "openai-compatible" | "ollama";

export type SavedConfig =
  | { kind: "default" }
  | {
      kind: "openai-compatible";
      base_url: string;
      api_key: string;
      model: string;
    }
  | {
      kind: "ollama";
      base_url: string;
      model: string;
    };

/**
 * Load the saved provider config from localStorage.
 *
 * The api_key is stored unencrypted in localStorage by design — the user is
 * the only owner of this browser, and we explicitly tell them so in the UI.
 * The alternative (asking the user to re-enter the key on every refresh)
 * trades real ergonomic cost for marginal protection that any XSS would
 * already defeat anyway.
 */
export function loadSavedConfig(): SavedConfig {
  if (typeof window === "undefined") return { kind: "default" };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { kind: "default" };
    const parsed = JSON.parse(raw) as SavedConfig;
    if (!parsed || typeof parsed !== "object" || !("kind" in parsed)) {
      return { kind: "default" };
    }
    return parsed;
  } catch {
    return { kind: "default" };
  }
}

function saveConfig(config: SavedConfig): void {
  if (typeof window === "undefined") return;
  if (config.kind === "default") {
    window.localStorage.removeItem(STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export function configToOverride(config: SavedConfig): ProviderOverride | undefined {
  if (config.kind === "default") return undefined;
  if (config.kind === "openai-compatible") {
    return {
      kind: "openai-compatible",
      base_url: config.base_url,
      api_key: config.api_key || undefined,
      model: config.model,
    };
  }
  return {
    kind: "ollama",
    base_url: config.base_url,
    model: config.model,
  };
}

export function describeConfig(config: SavedConfig): string {
  if (config.kind === "default") return "Default (server)";
  if (config.kind === "openai-compatible") {
    let host: string;
    try {
      host = new URL(config.base_url).hostname;
    } catch {
      host = config.base_url;
    }
    return `${host} · ${config.model}`;
  }
  return `Ollama · ${config.model}`;
}

interface ModelSettingsProps {
  open: boolean;
  current: SavedConfig;
  onClose: () => void;
  onSave: (config: SavedConfig) => void;
}

export function ModelSettings({ open, current, onClose, onSave }: ModelSettingsProps) {
  const [choice, setChoice] = useState<ProviderChoice>(current.kind);
  const [openaiBaseUrl, setOpenaiBaseUrl] = useState(
    current.kind === "openai-compatible" ? current.base_url : "https://api.openai.com/v1"
  );
  const [openaiApiKey, setOpenaiApiKey] = useState(
    current.kind === "openai-compatible" ? current.api_key : ""
  );
  const [openaiModel, setOpenaiModel] = useState(
    current.kind === "openai-compatible" ? current.model : "gpt-4.1"
  );
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState(
    current.kind === "ollama" ? current.base_url : "http://localhost:11434"
  );
  const [ollamaModel, setOllamaModel] = useState(
    current.kind === "ollama" ? current.model : "qwen3:8b"
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setChoice(current.kind);
    setError(null);
    if (current.kind === "openai-compatible") {
      setOpenaiBaseUrl(current.base_url);
      setOpenaiApiKey(current.api_key);
      setOpenaiModel(current.model);
    } else if (current.kind === "ollama") {
      setOllamaBaseUrl(current.base_url);
      setOllamaModel(current.model);
    }
  }, [open, current]);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = buildConfig(choice, {
      openaiBaseUrl,
      openaiApiKey,
      openaiModel,
      ollamaBaseUrl,
      ollamaModel,
    });
    const validationError = validateConfig(next);
    if (validationError) {
      setError(validationError);
      return;
    }
    saveConfig(next);
    onSave(next);
    onClose();
  }

  return (
    <div
      className="settings-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-settings-title"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <form className="settings-modal" onSubmit={onSubmit}>
        <header className="settings-header">
          <h2 id="model-settings-title">Model</h2>
          <button
            type="button"
            className="button-quiet"
            onClick={onClose}
            aria-label="Close settings"
          >
            Close
          </button>
        </header>

        <p className="settings-privacy">
          Your API key and endpoint are saved <strong>only in this browser</strong> and
          sent with each chat request. The server forwards them to your provider
          and discards them when the request ends — they are not written to
          logs, audit, or the feedback record.
        </p>

        <fieldset className="settings-choice-group">
          <legend className="visually-hidden">Provider</legend>

          <label className={`settings-choice ${choice === "default" ? "active" : ""}`}>
            <input
              type="radio"
              name="provider"
              value="default"
              checked={choice === "default"}
              onChange={() => setChoice("default")}
            />
            <span>
              <strong>Default (server)</strong>
              <small>Use the model the server is configured with. No setup needed.</small>
            </span>
          </label>

          <label className={`settings-choice ${choice === "openai-compatible" ? "active" : ""}`}>
            <input
              type="radio"
              name="provider"
              value="openai-compatible"
              checked={choice === "openai-compatible"}
              onChange={() => setChoice("openai-compatible")}
            />
            <span>
              <strong>OpenAI-compatible</strong>
              <small>
                Works with OpenAI, OpenRouter, Kimi, Groq, vLLM and other endpoints
                that implement <code>/v1/chat/completions</code>.
              </small>
            </span>
          </label>

          <label className={`settings-choice ${choice === "ollama" ? "active" : ""}`}>
            <input
              type="radio"
              name="provider"
              value="ollama"
              checked={choice === "ollama"}
              onChange={() => setChoice("ollama")}
            />
            <span>
              <strong>Ollama</strong>
              <small>
                Local model via Ollama. The endpoint must be reachable from the
                W3C server — your laptop's <code>localhost</code> usually is not.
              </small>
            </span>
          </label>
        </fieldset>

        {choice === "openai-compatible" ? (
          <div className="settings-fields">
            <label>
              Base URL
              <input
                type="url"
                required
                value={openaiBaseUrl}
                onChange={(event) => setOpenaiBaseUrl(event.target.value)}
                placeholder="https://api.openai.com/v1"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label>
              API key
              <input
                type="password"
                value={openaiApiKey}
                onChange={(event) => setOpenaiApiKey(event.target.value)}
                placeholder="sk-..."
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label>
              Model
              <input
                type="text"
                required
                value={openaiModel}
                onChange={(event) => setOpenaiModel(event.target.value)}
                placeholder="gpt-4.1"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
          </div>
        ) : null}

        {choice === "ollama" ? (
          <div className="settings-fields">
            <label>
              Base URL
              <input
                type="url"
                required
                value={ollamaBaseUrl}
                onChange={(event) => setOllamaBaseUrl(event.target.value)}
                placeholder="http://localhost:11434"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label>
              Model
              <input
                type="text"
                required
                value={ollamaModel}
                onChange={(event) => setOllamaModel(event.target.value)}
                placeholder="qwen3:8b"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
          </div>
        ) : null}

        {error ? <p className="settings-error">{error}</p> : null}

        <footer className="settings-actions">
          <button type="button" className="button-quiet" onClick={onClose}>
            Cancel
          </button>
          <button type="submit">Save</button>
        </footer>
      </form>
    </div>
  );
}

interface FormState {
  openaiBaseUrl: string;
  openaiApiKey: string;
  openaiModel: string;
  ollamaBaseUrl: string;
  ollamaModel: string;
}

function buildConfig(choice: ProviderChoice, state: FormState): SavedConfig {
  if (choice === "openai-compatible") {
    return {
      kind: "openai-compatible",
      base_url: state.openaiBaseUrl.trim(),
      api_key: state.openaiApiKey.trim(),
      model: state.openaiModel.trim(),
    };
  }
  if (choice === "ollama") {
    return {
      kind: "ollama",
      base_url: state.ollamaBaseUrl.trim(),
      model: state.ollamaModel.trim(),
    };
  }
  return { kind: "default" };
}

function validateConfig(config: SavedConfig): string | null {
  if (config.kind === "default") return null;
  if (!config.base_url) return "Base URL is required.";
  let url: URL;
  try {
    url = new URL(config.base_url);
  } catch {
    return "Base URL must be a valid http(s) URL.";
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return "Base URL must use http or https.";
  }
  if (!config.model) return "Model is required.";
  return null;
}
