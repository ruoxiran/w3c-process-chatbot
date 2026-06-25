export type Citation = {
  title: string;
  url: string;
  source_type: "process" | "guide" | "related_policy" | "repo";
  section_id?: string | null;
  heading_path?: string | null;
  commit_sha?: string | null;
  published_version_date?: string | null;
  quote?: string | null;
};

export type ChatResponse = {
  answer: string;
  in_scope: boolean;
  citations: Citation[];
  next_steps: string[];
  next_step_details: NextStep[];
  task_plan?: TaskPlan | null;
  evidence_coverage?: EvidenceCoverage | null;
  process_state?: ProcessState | null;
  compiled_context?: CompiledContext | null;
  compiled_context_used: boolean;
  resolved_entities: W3CEntity[];
  draft_contexts: DraftContext[];
  confidence: number;
  refusal_reason?: string | null;
  source_version: {
    process_version_date?: string | null;
    process_commit_sha?: string | null;
    guide_commit_sha?: string | null;
    indexed_at?: string | null;
  };
  workflow_trace: WorkflowStep[];
};

export type CompiledContext = {
  kind: "spec";
  key: string;
  title: string;
  summary: string;
  current_state?: string | null;
  next_step_candidates: string[];
  guide_signals: string[];
  horizontal_review_signals: string[];
  charter_signals: string[];
  freshness: {
    compiled_at?: string | null;
    source_snapshot: string[];
    is_stale: boolean;
  };
  provenance: {
    normative_urls: string[];
    guide_urls: string[];
    operational_urls: string[];
  };
  source_path?: string | null;
  confidence: number;
};

export type W3CEntity = {
  entity_type: "specification" | "group";
  title: string;
  shortname?: string | null;
  api_url: string;
  public_url?: string | null;
  editor_draft_url?: string | null;
  status?: string | null;
  latest_version_url?: string | null;
  latest_version_date?: string | null;
  process_rules_url?: string | null;
  deliverers: string[];
  charter_url?: string | null;
  charter_end?: string | null;
  patent_policy_url?: string | null;
  team_contacts: string[];
  group_type?: string | null;
  description?: string | null;
  retrieval_hints: string[];
  confidence: number;
};

export type DraftSnippet = {
  path: string;
  title?: string | null;
  text: string;
  url?: string | null;
};

export type DraftContext = {
  repo_full_name: string;
  repo_url: string;
  resolved_from?: string | null;
  default_branch?: string | null;
  description?: string | null;
  homepage?: string | null;
  latest_commit_sha?: string | null;
  open_issues_count?: number | null;
  snippets: DraftSnippet[];
  retrieval_hints: string[];
  confidence: number;
};

export type ProcessState = {
  intent: string;
  current_stage?: string | null;
  target_stage?: string | null;
  group_type?: string | null;
  deliverable_type?: string | null;
  likely_workflow: string;
  missing_information: string[];
  risk_flags: string[];
  confidence: number;
};

export type TaskPlan = {
  intent_type: string;
  user_goal: string;
  current_stage?: string | null;
  target_stage?: string | null;
  spec_or_group?: string | null;
  needed_sources: Array<"process" | "guide" | "related_policy" | "repo">;
  answer_shape: string;
  search_queries: string[];
  risk_flags: string[];
  confidence: number;
};

export type EvidenceCoverage = {
  status: "sufficient" | "needs_more_evidence" | "insufficient";
  has_compiled_context: boolean;
  has_process: boolean;
  has_guide: boolean;
  has_entity_status: boolean;
  missing_evidence: string[];
  targeted_queries: string[];
  summary: string;
  confidence: number;
};

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

/**
 * Per-request LLM provider override.
 *
 * The user supplies this in the Settings modal; the values live only in this
 * browser's localStorage. They are sent with each chat request and forwarded
 * to the user's chosen provider. The server does NOT persist them — they are
 * absent from the audit blob, the feedback log, and all other server-side
 * storage.
 */
export type ProviderOverride = {
  kind: "openai-compatible" | "ollama";
  base_url: string;
  api_key?: string;
  model: string;
};

export type NextStep = {
  text: string;
  source_title?: string | null;
  source_url?: string | null;
  source_type?: "process" | "guide" | "related_policy" | "repo" | null;
  source_heading?: string | null;
};

export type WorkflowStep = {
  id: string;
  label: string;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  detail: string;
  references: Citation[];
};

export type ModelInfo = {
  name: string;
  provider: string;
  size?: number | null;
  modified_at?: string | null;
  family?: string | null;
  is_embedding: boolean;
};

export type ModelsResponse = {
  default_model: string;
  models: ModelInfo[];
  error?: string | null;
};

export type EvalCaseResult = {
  name: string;
  passed: boolean;
  details: string;
  tags: string[];
  expected_in_scope?: boolean | null;
  actual_in_scope?: boolean | null;
  expected_intent?: string | null;
  actual_intent?: string | null;
  expected_source_types: string[];
  actual_source_types: string[];
  expected_url_substrings: string[];
  actual_urls: string[];
  expected_entity_shortname?: string | null;
  actual_entity_shortnames: string[];
  expected_compiled_context?: boolean | null;
  actual_compiled_context?: boolean | null;
  confidence?: number | null;
  warnings: string[];
};

export type EvalRunResponse = {
  passed: boolean;
  score: number;
  passed_count: number;
  total_count: number;
  results: EvalCaseResult[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const DEFAULT_TIMEOUT_MS = 15_000;
const CHAT_TIMEOUT_MS = 120_000;

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

export async function listModels(): Promise<ModelsResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/models`);

  if (!response.ok) {
    throw new Error(`Model request failed with ${response.status}`);
  }

  return response.json() as Promise<ModelsResponse>;
}

export async function sendChat(
  message: string,
  model?: string,
  history: ChatTurn[] = [],
  providerOverride?: ProviderOverride
): Promise<ChatResponse> {
  const body: Record<string, unknown> = {
    message,
    locale: "en",
    model,
    history: history.slice(-8),
  };
  if (providerOverride) {
    body.provider_override = providerOverride;
  }
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    CHAT_TIMEOUT_MS
  );

  if (!response.ok) {
    throw new Error(`Chat request failed with ${response.status}`);
  }

  return response.json() as Promise<ChatResponse>;
}

export async function runEval(): Promise<EvalRunResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/eval/run`, { method: "POST" }, CHAT_TIMEOUT_MS);

  if (!response.ok) {
    throw new Error(`Eval request failed with ${response.status}`);
  }

  return response.json() as Promise<EvalRunResponse>;
}

export type FeedbackRating = "up" | "down";

export type FeedbackInput = {
  rating: FeedbackRating;
  question: string;
  answer: string;
  comment?: string;
  conversationId?: string;
  messageId?: string;
  model?: string;
  inScope?: boolean;
  confidence?: number;
  citationUrls?: string[];
  audit?: Record<string, unknown>;
};

export type FeedbackResponse = {
  status: string;
  received_at: string;
};

export async function submitFeedback(input: FeedbackInput): Promise<FeedbackResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      rating: input.rating,
      question: input.question,
      answer: input.answer,
      comment: input.comment ?? null,
      conversation_id: input.conversationId ?? null,
      message_id: input.messageId ?? null,
      model: input.model ?? null,
      in_scope: input.inScope ?? null,
      confidence: input.confidence ?? null,
      citation_urls: input.citationUrls ?? [],
      audit: input.audit ?? {},
    }),
  });

  if (!response.ok) {
    throw new Error(`Feedback submission failed with ${response.status}`);
  }

  return response.json() as Promise<FeedbackResponse>;
}
