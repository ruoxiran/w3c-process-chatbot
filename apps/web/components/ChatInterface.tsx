"use client";

import { FormEvent, KeyboardEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import { CitationPanel } from "@/components/CitationPanel";
import {
  ModelSettings,
  configToOverride,
  describeConfig,
  loadSavedConfig,
  type SavedConfig,
} from "@/components/ModelSettings";
import { WorkflowPanel } from "@/components/WorkflowPanel";
import {
  listModels,
  runEval,
  sendChat,
  sendChatStream,
  submitFeedback,
  type ChatResponse,
  type Citation,
  type CompiledContext,
  type DraftContext,
  type EvalRunResponse,
  type ChatTurn,
  type FeedbackRating,
  type ModelInfo,
  type W3CEntity
} from "@/lib/api";

const starterQuestions = [
  "What should a CSS specification do next to move from CR to REC?",
  "How does the W3C Process handle a Formal Objection?",
  "What should a Working Group check before updating its charter?"
];

type ConversationMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
  status?: "pending" | "error";
  question?: string;
  model?: string;
};

type InspectorTab = "workflow" | "sources" | "entities" | "quality" | "version";

export function ChatInterface() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("qwen3:8b");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedResponse, setSelectedResponse] = useState<ChatResponse | null>(null);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("workflow");
  const [evalRun, setEvalRun] = useState<EvalRunResponse | null>(null);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  // Provider override (user's own LLM endpoint + key, stored in localStorage
  // only). ``null`` while we hydrate from storage; ``{kind: "default"}`` once
  // we've checked. This avoids sending an override on the very first server
  // render in case localStorage had stale data.
  const [providerConfig, setProviderConfig] = useState<SavedConfig | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    setProviderConfig(loadSavedConfig());
  }, []);

  const latestResponse = useMemo(
    () => [...messages].reverse().find((item) => item.response)?.response ?? null,
    [messages]
  );
  const activeResponse = selectedResponse ?? latestResponse;

  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      try {
        const result = await listModels();
        if (cancelled) return;
        const chatModels = result.models.filter((model) => !model.is_embedding);
        setModels(chatModels);
        setSelectedModel(result.default_model || chatModels[0]?.name || "qwen3:8b");
      } catch {
        if (!cancelled) {
          setModels([]);
        }
      }
    }

    loadModels();

    return () => {
      cancelled = true;
    };
  }, []);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitQuestion();
  }

  async function submitQuestion() {
    const trimmed = message.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ConversationMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed
    };
    const pendingMessage: ConversationMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "Checking W3C Process and Guidebook sources...",
      status: "pending",
      question: trimmed,
      model: selectedModel
    };
    const history = toChatHistory(messages);

    setMessages((current) => [...current, userMessage, pendingMessage]);
    setMessage("");
    setIsLoading(true);
    setSelectedResponse(null);
    setInspectorTab("workflow");

    try {
      const override = providerConfig ? configToOverride(providerConfig) : undefined;
      // Streaming path: update the bubble + inspector progressively as the
      // server emits meta then delta chunks. The first ``meta`` carries the
      // workflow trace and citations so the right panel populates immediately;
      // the chunks fill in the answer text with a typing effect.
      const result = await sendChatStream(
        trimmed,
        {
          onMeta: (meta) => {
            const partial = { ...meta, answer: "" } as ChatResponse;
            setSelectedResponse(partial);
            setMessages((current) =>
              current.map((item) =>
                item.id === pendingMessage.id
                  ? { ...item, content: "", response: partial, status: undefined }
                  : item
              )
            );
          },
          onChunk: (accumulated) => {
            setMessages((current) =>
              current.map((item) =>
                item.id === pendingMessage.id
                  ? { ...item, content: accumulated }
                  : item
              )
            );
          },
        },
        selectedModel,
        history,
        override
      );
      setSelectedResponse(result);
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingMessage.id
            ? { ...item, content: result.answer, response: result, status: undefined }
            : item
        )
      );
    } catch (err) {
      const content = err instanceof Error ? err.message : "Request failed";
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingMessage.id
            ? {
                ...item,
                content,
                status: "error"
              }
            : item
        )
      );
    } finally {
      setIsLoading(false);
    }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }
    event.preventDefault();
    void submitQuestion();
  }

  function startQuestion(question: string) {
    setMessage(question);
  }

  function clearConversation() {
    setMessages([]);
    setMessage("");
    setSelectedResponse(null);
    setInspectorTab("workflow");
  }

  function focusResponse(response: ChatResponse, tab: InspectorTab) {
    setSelectedResponse(response);
    setInspectorTab(tab);
  }

  async function runQualityEval() {
    if (isEvalLoading) return;
    setIsEvalLoading(true);
    setEvalError(null);
    setInspectorTab("quality");
    try {
      const result = await runEval();
      setEvalRun(result);
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : "Eval request failed");
    } finally {
      setIsEvalLoading(false);
    }
  }

  return (
    <main className="chat-layout">
      <section className="chat-main" aria-labelledby="page-title">
        <div className="chat-topbar">
          <div>
            <p className="eyebrow">W3C internal workflow assistant</p>
            <h1 id="page-title">W3C Process Assistant</h1>
          </div>
          <div className="topbar-actions">
            {providerConfig && providerConfig.kind !== "default" ? (
              <span className="provider-chip" title="Using your own provider">
                {describeConfig(providerConfig)}
              </span>
            ) : (
              <>
                <label htmlFor="model">Model</label>
                <select
                  id="model"
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                >
                  {models.length ? (
                    models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name}
                      </option>
                    ))
                  ) : (
                    <option value={selectedModel}>{selectedModel}</option>
                  )}
                </select>
              </>
            )}
            <button
              className="button-quiet"
              type="button"
              onClick={() => setSettingsOpen(true)}
              aria-haspopup="dialog"
            >
              Model settings
            </button>
            <button className="button-secondary" type="button" onClick={clearConversation}>
              New chat
            </button>
          </div>
        </div>

        <ModelSettings
          open={settingsOpen}
          current={providerConfig ?? { kind: "default" }}
          onClose={() => setSettingsOpen(false)}
          onSave={setProviderConfig}
        />

        <div
          className="conversation"
          role="log"
          aria-live="polite"
          aria-relevant="additions"
          aria-atomic="false"
        >
          {messages.length ? (
            messages.map((item) => (
              <ChatBubble key={item.id} message={item} onInspect={focusResponse} />
            ))
          ) : (
            <section className="empty-chat" aria-label="Start a conversation">
              <h2>Ask about W3C Process workflow</h2>
              <p>
                This page keeps the current conversation in memory while it stays open. Refreshing
                or closing the page clears the conversation.
              </p>
              <div className="starter-list" aria-label="Example questions">
                {starterQuestions.map((question) => (
                  <button key={question} type="button" onClick={() => startQuestion(question)}>
                    {question}
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <label className="sr-only" htmlFor="question">
            Question
          </label>
          <textarea
            id="question"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={onComposerKeyDown}
            placeholder="Ask a W3C Process or Guidebook workflow question"
            rows={3}
          />
          <button type="submit" disabled={isLoading || !message.trim()}>
            {isLoading ? "Checking" : "Ask"}
          </button>
        </form>
      </section>

      <aside className="workflow-sidebar" aria-label="Workflow and sources">
        <div className="inspector-tabs" role="tablist" aria-label="Response inspector">
          <button
            type="button"
            role="tab"
            id="inspector-tab-workflow"
            aria-selected={inspectorTab === "workflow"}
            aria-controls="inspector-panel-workflow"
            className={inspectorTab === "workflow" ? "active" : ""}
            onClick={() => setInspectorTab("workflow")}
          >
            Workflow
          </button>
          <button
            type="button"
            role="tab"
            id="inspector-tab-sources"
            aria-selected={inspectorTab === "sources"}
            aria-controls="inspector-panel-sources"
            className={inspectorTab === "sources" ? "active" : ""}
            onClick={() => setInspectorTab("sources")}
          >
            Sources
          </button>
          <button
            type="button"
            role="tab"
            id="inspector-tab-entities"
            aria-selected={inspectorTab === "entities"}
            aria-controls="inspector-panel-entities"
            className={inspectorTab === "entities" ? "active" : ""}
            onClick={() => setInspectorTab("entities")}
          >
            Entities
          </button>
          <button
            type="button"
            role="tab"
            id="inspector-tab-quality"
            aria-selected={inspectorTab === "quality"}
            aria-controls="inspector-panel-quality"
            className={inspectorTab === "quality" ? "active" : ""}
            onClick={() => setInspectorTab("quality")}
          >
            Quality
          </button>
          <button
            type="button"
            role="tab"
            id="inspector-tab-version"
            aria-selected={inspectorTab === "version"}
            aria-controls="inspector-panel-version"
            className={inspectorTab === "version" ? "active" : ""}
            onClick={() => setInspectorTab("version")}
          >
            Version
          </button>
        </div>

        {inspectorTab === "workflow" ? (
          <div role="tabpanel" id="inspector-panel-workflow" aria-labelledby="inspector-tab-workflow">
            <WorkflowPanel response={activeResponse ?? undefined} steps={activeResponse?.workflow_trace} isLoading={isLoading} />
          </div>
        ) : null}
        {inspectorTab === "entities" ? (
          <div role="tabpanel" id="inspector-panel-entities" aria-labelledby="inspector-tab-entities">
            <EntityPanel
              compiledContext={activeResponse?.compiled_context ?? null}
              entities={activeResponse?.resolved_entities ?? []}
              draftContexts={activeResponse?.draft_contexts ?? []}
            />
          </div>
        ) : null}
        {inspectorTab === "sources" ? (
          <div role="tabpanel" id="inspector-panel-sources" aria-labelledby="inspector-tab-sources">
            <CitationPanel response={activeResponse} mode="sources" />
          </div>
        ) : null}
        {inspectorTab === "quality" ? (
          <div role="tabpanel" id="inspector-panel-quality" aria-labelledby="inspector-tab-quality">
            <QualityPanel
              result={evalRun}
              isLoading={isEvalLoading}
              error={evalError}
              onRun={runQualityEval}
            />
          </div>
        ) : null}
        {inspectorTab === "version" ? (
          <div role="tabpanel" id="inspector-panel-version" aria-labelledby="inspector-tab-version">
            <CitationPanel response={activeResponse} mode="version" />
          </div>
        ) : null}
      </aside>
    </main>
  );
}

function QualityPanel({
  result,
  isLoading,
  error,
  onRun
}: {
  result: EvalRunResponse | null;
  isLoading: boolean;
  error: string | null;
  onRun: () => void;
}) {
  const failed = result?.results.filter((item) => !item.passed) ?? [];
  const warnings = result?.results.filter((item) => item.warnings.length) ?? [];

  return (
    <aside className="source-panel quality-panel" aria-label="Quality evaluation">
      <div className="quality-header">
        <div>
          <p className="eyebrow">Regression harness</p>
          <h2>Quality</h2>
        </div>
        <button className="button-secondary" type="button" onClick={onRun} disabled={isLoading}>
          {isLoading ? "Running" : "Run eval"}
        </button>
      </div>

      {error ? <div className="callout danger">{error}</div> : null}

      {result ? (
        <>
          <section className={`quality-score ${result.passed ? "passed" : "failed"}`} aria-label="Eval score">
            <div>
              <strong>{Math.round(result.score * 100)}%</strong>
              <span>{result.passed_count} / {result.total_count} passed</span>
            </div>
            <span>{result.passed ? "Passing" : "Needs attention"}</span>
          </section>

          {failed.length ? (
            <section className="quality-section" aria-label="Failed eval cases">
              <h3>Failures</h3>
              <ul className="quality-case-list">
                {failed.map((item) => (
                  <li key={item.name} className="quality-case failed">
                    <strong>{item.name}</strong>
                    <p>{item.details}</p>
                    <QualityTags tags={item.tags} />
                  </li>
                ))}
              </ul>
            </section>
          ) : (
            <div className="callout success">All golden cases are passing.</div>
          )}

          {warnings.length ? (
            <section className="quality-section" aria-label="Eval warnings">
              <h3>Warnings</h3>
              <ul className="quality-case-list">
                {warnings.slice(0, 6).map((item) => (
                  <li key={item.name} className="quality-case warning">
                    <strong>{item.name}</strong>
                    <p>{item.warnings.join("; ")}</p>
                    <QualityTags tags={item.tags} />
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          <details className="diagnostic-details">
            <summary>All cases</summary>
            <ul className="quality-case-list compact">
              {result.results.map((item) => (
                <li key={item.name} className={`quality-case ${item.passed ? "passed" : "failed"}`}>
                  <span>{item.passed ? "Pass" : "Fail"}</span>
                  <strong>{item.name}</strong>
                  <small>{item.actual_intent ?? "no intent"}</small>
                </li>
              ))}
            </ul>
          </details>
        </>
      ) : (
        <p className="muted">
          Run the golden-question harness to check scope, intent, citations, entity grounding,
          compiled context, next-step focus, and injection resistance.
        </p>
      )}
    </aside>
  );
}

function QualityTags({ tags }: { tags: string[] }) {
  if (!tags.length) return null;
  return (
    <div className="diagnostic-chip-group" aria-label="Eval case tags">
      {tags.map((tag) => (
        <span key={tag}>{tag}</span>
      ))}
    </div>
  );
}

function EntityPanel({
  compiledContext,
  entities,
  draftContexts
}: {
  compiledContext: CompiledContext | null;
  entities: W3CEntity[];
  draftContexts: DraftContext[];
}) {
  return (
    <aside className="source-panel" aria-label="Resolved W3C API entities">
      <h2>Entities</h2>
      {compiledContext ? <CompiledContextCard context={compiledContext} /> : null}
      {entities.length ? (
        <ul className="source-list">
          {entities.map((entity) => (
            <li className="source-item" key={entity.api_url}>
              <span className="source-badge source-related_policy">
                {entity.entity_type === "specification" ? "Specification" : "Group"}
              </span>
              <a href={entity.public_url ?? entity.api_url} target="_blank" rel="noreferrer">
                {entity.title}
              </a>
              <dl className="entity-meta">
                {entity.shortname ? (
                  <>
                    <dt>Shortname</dt>
                    <dd>{entity.shortname}</dd>
                  </>
                ) : null}
                {entity.status ? (
                  <>
                    <dt>Status</dt>
                    <dd>{entity.status}</dd>
                  </>
                ) : null}
                {entity.latest_version_date ? (
                  <>
                    <dt>Latest</dt>
                    <dd>{entity.latest_version_date}</dd>
                  </>
                ) : null}
                {entity.editor_draft_url ? (
                  <>
                    <dt>Editor draft</dt>
                    <dd>
                      <a href={entity.editor_draft_url} target="_blank" rel="noreferrer">
                        Open
                      </a>
                    </dd>
                  </>
                ) : null}
                {entity.group_type ? (
                  <>
                    <dt>Type</dt>
                    <dd>{entity.group_type}</dd>
                  </>
                ) : null}
                {entity.deliverers.length ? (
                  <>
                    <dt>Deliverer</dt>
                    <dd>{entity.deliverers.join(", ")}</dd>
                  </>
                ) : null}
                {entity.charter_end ? (
                  <>
                    <dt>Charter end</dt>
                    <dd>{entity.charter_end}</dd>
                  </>
                ) : null}
                {entity.team_contacts.length ? (
                  <>
                    <dt>Team contact</dt>
                    <dd>{entity.team_contacts.join(", ")}</dd>
                  </>
                ) : null}
                <dt>Confidence</dt>
                <dd>{Math.round(entity.confidence * 100)}%</dd>
              </dl>
              {entity.retrieval_hints.length ? (
                <div className="retrieval-hints" aria-label="Fields used for retrieval">
                  <strong>Used for retrieval</strong>
                  <div>
                    {entity.retrieval_hints.map((hint) => (
                      <span key={hint}>{hint}</span>
                    ))}
                  </div>
                </div>
              ) : null}
              {entity.description ? <p>{entity.description}</p> : null}
              <div className="entity-links">
                <a href={entity.api_url} target="_blank" rel="noreferrer">
                  W3C API record
                </a>
                {entity.latest_version_url ? (
                  <a href={entity.latest_version_url} target="_blank" rel="noreferrer">
                    Latest version API
                  </a>
                ) : null}
                {entity.editor_draft_url ? (
                  <a href={entity.editor_draft_url} target="_blank" rel="noreferrer">
                    Editor draft
                  </a>
                ) : null}
                {entity.charter_url ? (
                  <a href={entity.charter_url} target="_blank" rel="noreferrer">
                    Active charter
                  </a>
                ) : null}
                {entity.patent_policy_url ? (
                  <a href={entity.patent_policy_url} target="_blank" rel="noreferrer">
                    Patent Policy
                  </a>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No strong specification or group match from the public W3C API.</p>
      )}
      {draftContexts.length ? (
        <>
          <h2>Draft repositories</h2>
          <ul className="source-list">
            {draftContexts.map((context) => (
              <li className="source-item" key={context.repo_full_name}>
                <span className="source-badge source-repo">Draft context</span>
                <a href={context.repo_url} target="_blank" rel="noreferrer">
                  {context.repo_full_name}
                </a>
                <dl className="entity-meta">
                  {context.default_branch ? (
                    <>
                      <dt>Branch</dt>
                      <dd>{context.default_branch}</dd>
                    </>
                  ) : null}
                  {context.latest_commit_sha ? (
                    <>
                      <dt>Commit</dt>
                      <dd>{context.latest_commit_sha}</dd>
                    </>
                  ) : null}
                  {typeof context.open_issues_count === "number" ? (
                    <>
                      <dt>Open issues</dt>
                      <dd>{context.open_issues_count}</dd>
                    </>
                  ) : null}
                  <dt>Confidence</dt>
                  <dd>{Math.round(context.confidence * 100)}%</dd>
                </dl>
                {context.description ? <p>{context.description}</p> : null}
                {context.snippets.length ? (
                  <ul className="workflow-references" aria-label="Draft context snippets">
                    {context.snippets.slice(0, 4).map((snippet) => (
                      <li key={snippet.path}>
                        {snippet.url ? (
                          <a href={snippet.url} target="_blank" rel="noreferrer">
                            {snippet.path}
                          </a>
                        ) : (
                          <span>{snippet.path}</span>
                        )}
                        <span>{snippet.title ?? "source"}</span>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </aside>
  );
}

function CompiledContextCard({ context }: { context: CompiledContext }) {
  return (
    <section className="source-item compiled-context-card" aria-label="Compiled spec context">
      <span className="source-badge source-compiled">Compiled context</span>
      <h3>{context.title}</h3>
      <dl className="entity-meta">
        <dt>Key</dt>
        <dd>{context.key}</dd>
        {context.current_state ? (
          <>
            <dt>State</dt>
            <dd>{context.current_state}</dd>
          </>
        ) : null}
        {context.freshness.compiled_at ? (
          <>
            <dt>Compiled</dt>
            <dd>{new Date(context.freshness.compiled_at).toLocaleString()}</dd>
          </>
        ) : null}
        <dt>Confidence</dt>
        <dd>{Math.round(context.confidence * 100)}%</dd>
      </dl>
      <p>{context.summary}</p>
      {context.next_step_candidates.length ? (
        <details className="diagnostic-details" open>
          <summary>Next step candidates</summary>
          <ul>
            {context.next_step_candidates.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </details>
      ) : null}
      {context.guide_signals.length || context.horizontal_review_signals.length || context.charter_signals.length ? (
        <details className="diagnostic-details">
          <summary>Workflow signals</summary>
          <ul>
            {[...context.guide_signals, ...context.horizontal_review_signals, ...context.charter_signals]
              .slice(0, 8)
              .map((signal) => (
                <li key={signal}>{signal}</li>
              ))}
          </ul>
        </details>
      ) : null}
      <details className="diagnostic-details">
        <summary>Provenance</summary>
        <dl className="entity-meta compiled-provenance">
          <dt>Process</dt>
          <dd>{context.provenance.normative_urls.length}</dd>
          <dt>Guidebook</dt>
          <dd>{context.provenance.guide_urls.length}</dd>
          <dt>Operational</dt>
          <dd>{context.provenance.operational_urls.length}</dd>
        </dl>
        <div className="entity-links">
          {context.provenance.normative_urls.slice(0, 2).map((url) => (
            <a href={url} key={url} target="_blank" rel="noreferrer">
              Process
            </a>
          ))}
          {context.provenance.guide_urls.slice(0, 2).map((url) => (
            <a href={url} key={url} target="_blank" rel="noreferrer">
              Guidebook
            </a>
          ))}
          {context.provenance.operational_urls.slice(0, 2).map((url) => (
            <a href={url} key={url} target="_blank" rel="noreferrer">
              Operational
            </a>
          ))}
        </div>
      </details>
    </section>
  );
}

function ChatBubble({
  message,
  onInspect
}: {
  message: ConversationMessage;
  onInspect: (response: ChatResponse, tab: InspectorTab) => void;
}) {
  const response = message.response;
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    if (!response) return;
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }

  return (
    <article className={`chat-message message-${message.role} ${message.status ? `message-${message.status}` : ""}`}>
      <div className="message-meta">{message.role === "user" ? "You" : "W3C Process Assistant"}</div>
      <div className="message-bubble">
        <AnswerContent text={message.content} citations={response?.citations} />
        {response ? (
          <div className="message-actions" aria-label="Answer actions">
            <button className="button-quiet" type="button" onClick={copyAnswer}>
              {copied ? "Copied" : "Copy answer"}
            </button>
            <button className="button-quiet" type="button" onClick={() => onInspect(response, "workflow")}>
              View workflow
            </button>
            <button className="button-quiet" type="button" onClick={() => onInspect(response, "sources")}>
              View sources
            </button>
          </div>
        ) : null}
        {response ? (
          <FeedbackControls
            response={response}
            question={message.question ?? ""}
            messageId={message.id}
            model={message.model}
          />
        ) : null}
        {response ? <ResponseDetails response={response} /> : null}
      </div>
    </article>
  );
}

interface FeedbackControlsProps {
  response: ChatResponse;
  question: string;
  messageId: string;
  model?: string;
}

function FeedbackControls({ response, question, messageId, model }: FeedbackControlsProps) {
  const [submitted, setSubmitted] = useState<FeedbackRating | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(rating: FeedbackRating, extraComment?: string) {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      // Audit blob is reconstructed server-side from the trusted ChatResponse
      // for the message. The client does not (and cannot) send a trusted audit.
      await submitFeedback({
        rating,
        question,
        answer: response.answer,
        comment: extraComment?.trim() || undefined,
        messageId,
        model,
        inScope: response.in_scope,
        confidence: response.confidence,
        citationUrls: response.citations.map((c) => c.url)
      });
      setSubmitted(rating);
      if (rating === "up") {
        setShowComment(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit feedback");
    } finally {
      setSubmitting(false);
    }
  }

  function onThumbsUp() {
    void send("up");
  }

  function onThumbsDown() {
    if (submitted === "down") {
      setShowComment((current) => !current);
      return;
    }
    setShowComment(true);
    void send("down");
  }

  async function onCommentSubmit() {
    if (!comment.trim()) {
      setShowComment(false);
      return;
    }
    try {
      await send(submitted ?? "down", comment);
      setComment("");
      setShowComment(false);
    } catch {
      // ``send`` already populated ``error`` via state; nothing else to do here.
    }
  }

  return (
    <div className="feedback-controls" aria-label="Answer feedback">
      <div className="feedback-buttons">
        <button
          type="button"
          className={`button-quiet ${submitted === "up" ? "active" : ""}`}
          aria-pressed={submitted === "up"}
          disabled={submitting}
          onClick={onThumbsUp}
          title="This answer was helpful"
        >
          {submitted === "up" ? "Thanks!" : "👍 Helpful"}
        </button>
        <button
          type="button"
          className={`button-quiet ${submitted === "down" ? "active" : ""}`}
          aria-pressed={submitted === "down"}
          disabled={submitting}
          onClick={onThumbsDown}
          title="This answer was not helpful or was incorrect"
        >
          {submitted === "down" ? "Recorded" : "👎 Inaccurate"}
        </button>
      </div>
      {showComment ? (
        <div className="feedback-comment">
          <label htmlFor={`feedback-${messageId}`} className="sr-only">
            Optional comment for the W3C Process team
          </label>
          <textarea
            id={`feedback-${messageId}`}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="What was wrong? (optional, sent to the W3C Process team)"
            rows={2}
            maxLength={4000}
          />
          <button
            type="button"
            className="button-quiet"
            onClick={onCommentSubmit}
            disabled={submitting}
          >
            Send comment
          </button>
        </div>
      ) : null}
      {error ? <p className="feedback-error">{error}</p> : null}
    </div>
  );
}

function ResponseDetails({ response }: { response: ChatResponse }) {
  return (
    <div className="response-details">
      <div className={`callout ${response.in_scope ? "success" : "warning"}`}>
        {response.in_scope ? "In scope: W3C Process workflow question" : "Out of scope"}
      </div>
      {response.process_state ? <ProcessStateSummary response={response} /> : null}
      {response.refusal_reason ? <p className="muted">{response.refusal_reason}</p> : null}
    </div>
  );
}

function ProcessStateSummary({ response }: { response: ChatResponse }) {
  const state = response.process_state;
  if (!state) return null;

  const facts = [
    ["Workflow", formatStateValue(state.likely_workflow)],
    ["Intent", formatStateValue(state.intent)],
    ["Stage", [state.current_stage, state.target_stage].filter(Boolean).join(" -> ") || "Not specified"],
    ["Group", state.group_type || "Not specified"],
    ["Deliverable", state.deliverable_type || "Not specified"]
  ];

  return (
    <section className="state-summary" aria-label="Process state">
      <div className="state-summary-head">
        <h2>Process state</h2>
        <span>{Math.round(state.confidence * 100)}%</span>
      </div>
      <dl>
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {state.risk_flags.length ? (
        <div className="state-tags" aria-label="Risk flags">
          {state.risk_flags.map((risk) => (
            <span key={risk}>{risk}</span>
          ))}
        </div>
      ) : null}
      {state.missing_information.length ? (
        <p className="muted">
          Missing: {state.missing_information.map(formatStateValue).join(", ")}
        </p>
      ) : null}
    </section>
  );
}

function formatStateValue(value: string) {
  return value.replaceAll("_", " ");
}

function toChatHistory(messages: ConversationMessage[]): ChatTurn[] {
  return messages
    .filter((item) => !item.status)
    .map((item) => ({
      role: item.role,
      content: item.content
    }))
    .slice(-8);
}

type AnswerBlock =
  | { type: "paragraph"; text: string }
  | { type: "ordered"; items: string[] }
  | { type: "unordered"; items: string[] };

function AnswerContent({ text, citations }: { text: string; citations?: Citation[] }) {
  const blocks = parseAnswerBlocks(text);
  const sources = citations ?? [];

  return (
    <div className="answer-body">
      {blocks.map((block, index) => {
        if (block.type === "paragraph") {
          return (
            <p key={`${block.type}-${index}`}>{renderInline(block.text, sources)}</p>
          );
        }

        const ListTag = block.type === "ordered" ? "ol" : "ul";
        return (
          <ListTag key={`${block.type}-${index}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`${itemIndex}-${item.slice(0, 32)}`}>{renderInline(item, sources)}</li>
            ))}
          </ListTag>
        );
      })}
    </div>
  );
}

/**
 * Render the limited inline markup we tolerate inside a paragraph or list
 * item: ``**bold**``, ``` `code` ```, and citation labels like ``[S1]`` which
 * resolve to clickable links pointing at the matching citation URL when
 * available. We deliberately do NOT parse general markdown links — citation
 * URLs are the only links we want to surface, and keeping the inline grammar
 * narrow keeps the XSS surface at zero.
 */
function renderInline(text: string, citations: Citation[]): ReactNode {
  const parts: ReactNode[] = [];
  const pattern = /\*\*(.+?)\*\*|`([^`]+)`|\[S(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[1] !== undefined) {
      parts.push(<strong key={`b-${key++}`}>{match[1]}</strong>);
    } else if (match[2] !== undefined) {
      parts.push(<code key={`c-${key++}`}>{match[2]}</code>);
    } else if (match[3] !== undefined) {
      const index = Number.parseInt(match[3], 10);
      const citation = citations[index - 1];
      if (citation?.url) {
        parts.push(
          <a
            className={`citation-ref source-${citation.source_type ?? "repo"}`}
            key={`s-${key++}`}
            href={citation.url}
            target="_blank"
            rel="noreferrer"
            title={citation.heading_path ?? citation.title ?? `Source ${index}`}
          >
            S{index}
          </a>
        );
      } else {
        parts.push(<span className="citation-ref-missing" key={`s-${key++}`}>S{index}</span>);
      }
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts.length ? parts : text;
}

function parseAnswerBlocks(text: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  const paragraphLines: string[] = [];
  let currentList: Extract<AnswerBlock, { type: "ordered" | "unordered" }> | null = null;

  function flushParagraph() {
    if (!paragraphLines.length) return;
    blocks.push({ type: "paragraph", text: paragraphLines.join(" ").trim() });
    paragraphLines.length = 0;
  }

  function flushList() {
    if (!currentList) return;
    blocks.push(currentList);
    currentList = null;
  }

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      // A blank line on its own does NOT close an in-progress list. Many
      // models (including Kimi) emit blank lines between list items, and
      // treating that as a list boundary makes each item render as its own
      // single-item <ol>, which the browser dutifully numbers "1." every
      // time. Only a non-list, non-empty line (handled below) closes the list.
      continue;
    }

    const ordered = line.match(/^\d+[\.)]\s+(.+)$/);
    const unordered = line.match(/^[-*•]\s+(.+)$/);

    if (ordered || unordered) {
      flushParagraph();
      const type = ordered ? "ordered" : "unordered";
      const item = (ordered?.[1] || unordered?.[1] || "").trim();
      if (!currentList || currentList.type !== type) {
        flushList();
        currentList = { type, items: [] };
      }
      currentList?.items.push(item);
      continue;
    }

    flushList();
    paragraphLines.push(line);
  }

  flushParagraph();
  flushList();

  return blocks.length ? blocks : [{ type: "paragraph", text }];
}
