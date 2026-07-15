import type { Citation, ChatResponse } from "@/lib/api";

function sourceLabel(type: Citation["source_type"]) {
  switch (type) {
    case "process":
      return "Process";
    case "guide":
      return "Guidebook";
    case "related_policy":
      return "Policy";
    case "repo":
      return "Repository";
  }
}

export function CitationPanel({
  response,
  mode = "all"
}: {
  response: ChatResponse | null;
  mode?: "all" | "sources" | "version";
}) {
  return (
    <aside className="source-panel" aria-label="Sources and version">
      {mode !== "version" ? (
        <>
          <h2>Sources</h2>
          {response?.citations.length ? (
            <ul className="source-list">
              {response.citations.map((citation, index) => (
                <li className="source-item" key={`${citation.url}-${citation.heading_path}`}>
                  {/* Matches the [S1], [S2] ... labels the answer text uses:
                      the answer's [Sn] is the nth source in this list. */}
                  <span className="source-tag">S{index + 1}</span>
                  <span className={`source-badge source-${citation.source_type}`}>
                    {sourceLabel(citation.source_type)}
                  </span>
                  <a href={citation.url} target="_blank" rel="noreferrer">
                    {citation.title}
                  </a>
                  {citation.heading_path ? <p>{citation.heading_path}</p> : null}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">Ask a W3C Process question to see authoritative sources.</p>
          )}
        </>
      ) : null}

      {mode !== "sources" ? (
        <>
          <h2>Version</h2>
          {response?.source_version.indexed_at ? (
            <dl className="version-list">
              <dt>Indexed</dt>
              <dd>{new Date(response.source_version.indexed_at).toLocaleString()}</dd>
              <dt>Confidence</dt>
              <dd>{Math.round(response.confidence * 100)}%</dd>
              {response.source_version.process_version_date ? (
                <>
                  <dt>Process date</dt>
                  <dd>{response.source_version.process_version_date}</dd>
                </>
              ) : null}
              {response.source_version.process_commit_sha ? (
                <>
                  <dt>Process commit</dt>
                  <dd>{response.source_version.process_commit_sha.slice(0, 12)}</dd>
                </>
              ) : null}
              {response.source_version.guide_commit_sha ? (
                <>
                  <dt>Guide commit</dt>
                  <dd>{response.source_version.guide_commit_sha.slice(0, 12)}</dd>
                </>
              ) : null}
            </dl>
          ) : (
            <p className="muted">No response yet.</p>
          )}
        </>
      ) : null}
    </aside>
  );
}
