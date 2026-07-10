/**
 * Unit tests for ``lib/api`` — the SSE protocol parser and the
 * end-to-end streaming flow that the chat UI depends on.
 *
 * The streaming protocol is the highest-leverage contract between
 * frontend and backend. A regression here breaks every chat request,
 * so it's worth a small, focused suite.
 */

import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import {
  parseSseEvent,
  sendChatStream,
  type WorkflowStep,
} from "./api";

// ---------- parseSseEvent --------------------------------------------------

describe("parseSseEvent", () => {
  test("parses a standard event + JSON data line", () => {
    const result = parseSseEvent('event: delta\ndata: {"text":"hello"}');
    expect(result).toEqual({ event: "delta", data: { text: "hello" } });
  });

  test("defaults to event name 'message' when no event line is present", () => {
    const result = parseSseEvent('data: {"text":"hi"}');
    expect(result).toEqual({ event: "message", data: { text: "hi" } });
  });

  test("returns null when no data lines are present", () => {
    expect(parseSseEvent("event: ping")).toBeNull();
    expect(parseSseEvent("")).toBeNull();
  });

  test("returns null when the data payload is not valid JSON", () => {
    expect(parseSseEvent("event: delta\ndata: not-json")).toBeNull();
  });

  test("joins multiple data: lines with a newline before parsing", () => {
    const raw = 'event: meta\ndata: {"answer":\ndata: "split"}';
    const result = parseSseEvent(raw);
    expect(result).toEqual({ event: "meta", data: { answer: "split" } });
  });

  test("trims leading whitespace from data values (per SSE spec)", () => {
    // The spec says "data:" with no space and "data: " both strip ONE leading
    // space. Our implementation uses .trimStart which is more lenient — that's
    // intentional, because Python's SSE serializer sometimes pads with two.
    const result = parseSseEvent('event: stage\ndata:   {"id":"scope"}');
    expect(result).toEqual({ event: "stage", data: { id: "scope" } });
  });
});


// ---------- sendChatStream end-to-end -------------------------------------

/**
 * Build a fake Response whose ``body`` is a ReadableStream yielding
 * the provided SSE chunks in order. Mirrors what the API returns.
 */
function fakeStreamingResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}


describe("sendChatStream", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("dispatches stage, delta, and meta callbacks in order then returns assembled response", async () => {
    const stages: WorkflowStep[] = [];
    const deltas: string[] = [];
    let metaSeen: unknown = null;

    const stream = [
      'event: stage\ndata: {"id":"scope_classifier","label":"Scope","status":"completed","detail":"ok"}\n\n',
      'event: delta\ndata: {"text":"Hello "}\n\n',
      'event: delta\ndata: {"text":"world."}\n\n',
      'event: meta\ndata: {"in_scope":true,"citations":[],"next_steps":[],"next_step_details":[],"compiled_context_used":false,"resolved_entities":[],"draft_contexts":[],"confidence":0.5,"source_version":{},"workflow_trace":[]}\n\n',
      "event: done\ndata: {}\n\n",
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    const response = await sendChatStream(
      "What is a WD?",
      {
        onStage: (step) => stages.push(step),
        onMeta: (meta) => {
          metaSeen = meta;
        },
        onChunk: (_acc, delta) => deltas.push(delta),
      },
    );

    expect(stages.map((s) => s.id)).toEqual(["scope_classifier"]);
    expect(deltas).toEqual(["Hello ", "world."]);
    expect(metaSeen).not.toBeNull();
    expect(response.answer).toBe("Hello world.");
    expect(response.in_scope).toBe(true);
  });

  test("uses done.answer when no deltas arrive (LLM failed, template fallback)", async () => {
    const deltas: string[] = [];
    const stream = [
      'event: meta\ndata: {"in_scope":true,"citations":[],"next_steps":[],"next_step_details":[],"compiled_context_used":false,"resolved_entities":[],"draft_contexts":[],"confidence":0.5,"source_version":{},"workflow_trace":[]}\n\n',
      'event: done\ndata: {"answer":"Template fallback answer [S1]."}\n\n',
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    const response = await sendChatStream(
      "How is a Formal Objection handled?",
      {
        onMeta: () => undefined,
        onChunk: (acc) => deltas.push(acc),
      },
    );

    expect(response.answer).toBe("Template fallback answer [S1].");
    expect(deltas).toContain("Template fallback answer [S1].");
  });

  test("prefers done.answer over accumulated deltas when they differ", async () => {
    const stream = [
      'event: delta\ndata: {"text":"raw streamed text"}\n\n',
      'event: meta\ndata: {"in_scope":true,"citations":[],"next_steps":[],"next_step_details":[],"compiled_context_used":false,"resolved_entities":[],"draft_contexts":[],"confidence":0.5,"source_version":{},"workflow_trace":[]}\n\n',
      'event: done\ndata: {"answer":"post-processed text"}\n\n',
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    const response = await sendChatStream(
      "What is a WD?",
      { onMeta: () => undefined, onChunk: () => undefined },
    );

    expect(response.answer).toBe("post-processed text");
  });

  test("throws when the stream emits an error event", async () => {
    const stream = [
      'event: error\ndata: {"message":"upstream LLM down"}\n\n',
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    await expect(
      sendChatStream(
        "anything",
        { onMeta: () => undefined, onChunk: () => undefined },
      ),
    ).rejects.toThrow(/upstream LLM down/);
  });

  test("throws when the stream ends without a meta event", async () => {
    const stream = [
      'event: delta\ndata: {"text":"only deltas"}\n\n',
      "event: done\ndata: {}\n\n",
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    await expect(
      sendChatStream(
        "anything",
        { onMeta: () => undefined, onChunk: () => undefined },
      ),
    ).rejects.toThrow(/meta event/);
  });

  test("throws when the response is not OK", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("server error", { status: 500 }),
    );

    await expect(
      sendChatStream(
        "anything",
        { onMeta: () => undefined, onChunk: () => undefined },
      ),
    ).rejects.toThrow(/500/);
  });

  test("handles SSE events split across multiple chunks", async () => {
    // Real network reads frequently split an SSE event across two
    // TCP segments. The parser MUST buffer until \n\n lands; if it
    // tries to parse half an event we drop data.
    const deltas: string[] = [];
    const stream = [
      'event: delta\ndata: {"text":"A',
      '"}\n\nevent: delta\ndata: {"text":"B"}\n\n',
      'event: meta\ndata: {"in_scope":true,"citations":[],"next_steps":[],"next_step_details":[],"compiled_context_used":false,"resolved_entities":[],"draft_contexts":[],"confidence":0.5,"source_version":{},"workflow_trace":[]}\n\n',
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    await sendChatStream(
      "anything",
      {
        onMeta: () => undefined,
        onChunk: (_acc, delta) => deltas.push(delta),
      },
    );

    expect(deltas).toEqual(["A", "B"]);
  });

  test("passes the provider_choice to the backend when supplied", async () => {
    const stream = [
      'event: meta\ndata: {"in_scope":true,"citations":[],"next_steps":[],"next_step_details":[],"compiled_context_used":false,"resolved_entities":[],"draft_contexts":[],"confidence":0.5,"source_version":{},"workflow_trace":[]}\n\n',
    ];
    vi.mocked(fetch).mockResolvedValue(fakeStreamingResponse(stream));

    await sendChatStream(
      "anything",
      { onMeta: () => undefined, onChunk: () => undefined },
      "us.anthropic.claude-sonnet-5",
      [],
      "bedrock",
    );

    const call = vi.mocked(fetch).mock.calls[0];
    const init = call[1] as RequestInit;
    const body = JSON.parse(init.body as string);
    expect(body.provider_choice).toBe("bedrock");
    expect(body.model).toBe("us.anthropic.claude-sonnet-5");
    expect(body.provider_override).toBeUndefined();
  });
});
