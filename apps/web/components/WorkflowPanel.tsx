import type { ChatResponse, EvidenceCoverage, TaskPlan, WorkflowStep } from "@/lib/api";

const loadingSteps: WorkflowStep[] = [
  {
    id: "scope_classifier",
    label: "Scope classifier",
    status: "completed",
    detail: "Checking whether the question belongs to W3C Process scope.",
    references: []
  },
  {
    id: "retriever",
    label: "Authoritative source retrieval",
    status: "running",
    detail: "Looking for trusted W3C Process and Guidebook sources.",
    references: []
  },
  {
    id: "answer_generator",
    label: "Answer generation",
    status: "pending",
    detail: "Waiting to generate a constrained answer.",
    references: []
  },
  {
    id: "citation_check",
    label: "Citation and source check",
    status: "pending",
    detail: "Waiting to verify the answer against trusted sources.",
    references: []
  },
  {
    id: "final_response",
    label: "Final conclusion",
    status: "pending",
    detail: "Waiting for the final conclusion.",
    references: []
  }
];

export function WorkflowPanel({
  response,
  steps,
  isLoading
}: {
  response?: ChatResponse;
  steps: WorkflowStep[] | undefined;
  isLoading: boolean;
}) {
  const displaySteps = isLoading ? loadingSteps : steps ?? [];

  if (!displaySteps.length) {
    return null;
  }

  return (
    <section className="workflow-panel" aria-labelledby="workflow-title" aria-live="polite">
      <h2 id="workflow-title">Workflow trace</h2>
      {!isLoading && response?.task_plan ? <TaskPlanSummary plan={response.task_plan} /> : null}
      {!isLoading && response?.evidence_coverage ? (
        <EvidenceCoverageSummary coverage={response.evidence_coverage} />
      ) : null}
      <ol className="workflow-list">
        {displaySteps.map((step, index) => (
          <li className={`workflow-step status-${step.status}`} key={`${step.id}-${index}`}>
            <div className="workflow-marker" aria-hidden="true">
              {index + 1}
            </div>
            <div className="workflow-content">
              <div className="workflow-heading">
                <h3>{step.label}</h3>
                <span className="workflow-status">{step.status}</span>
              </div>
              <p>{step.detail}</p>
              {step.references.length ? (
                <ul className="workflow-references" aria-label={`${step.label} references`}>
                  {step.references.map((reference) => (
                    <li key={`${step.id}-${reference.url}`}>
                      <a href={reference.url} target="_blank" rel="noreferrer">
                        {reference.title}
                      </a>
                      <span>{reference.source_type}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function TaskPlanSummary({ plan }: { plan: TaskPlan }) {
  const fields = [
    ["Intent", plan.intent_type],
    ["Goal", plan.user_goal],
    ["Subject", plan.spec_or_group],
    ["Current stage", plan.current_stage],
    ["Target stage", plan.target_stage],
    ["Answer shape", plan.answer_shape],
    ["Needed sources", plan.needed_sources.join(", ")],
    ["Risk flags", plan.risk_flags.join(", ") || "none"]
  ].filter(([, value]) => value);

  return (
    <section className="workflow-diagnostic" aria-label="Task plan">
      <div className="diagnostic-heading">
        <h3>Task plan</h3>
        <span>{Math.round(plan.confidence * 100)}%</span>
      </div>
      <dl className="diagnostic-grid">
        {fields.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {plan.search_queries.length ? (
        <details className="diagnostic-details">
          <summary>Focused retrieval queries</summary>
          <ul>
            {plan.search_queries.map((query) => (
              <li key={query}>{query}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

function EvidenceCoverageSummary({ coverage }: { coverage: EvidenceCoverage }) {
  const checks = [
    ["Compiled context", coverage.has_compiled_context],
    ["Process", coverage.has_process],
    ["Guidebook", coverage.has_guide],
    ["W3C API status", coverage.has_entity_status]
  ];

  return (
    <section className={`workflow-diagnostic coverage-${coverage.status}`} aria-label="Evidence coverage">
      <div className="diagnostic-heading">
        <h3>Evidence coverage</h3>
        <span>{coverage.status.replaceAll("_", " ")}</span>
      </div>
      <p>{coverage.summary}</p>
      <ul className="coverage-checks" aria-label="Coverage checks">
        {checks.map(([label, passed]) => (
          <li className={passed ? "passed" : "missing"} key={label as string}>
            <span aria-hidden="true">{passed ? "✓" : "!"}</span>
            {label as string}
          </li>
        ))}
      </ul>
      {coverage.missing_evidence.length ? (
        <div className="diagnostic-chip-group" aria-label="Missing evidence">
          {coverage.missing_evidence.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      ) : null}
      {coverage.targeted_queries.length ? (
        <details className="diagnostic-details">
          <summary>Targeted retrieval</summary>
          <ul>
            {coverage.targeted_queries.map((query) => (
              <li key={query}>{query}</li>
            ))}
          </ul>
        </details>
      ) : null}
      {responseCompiledNote(coverage)}
    </section>
  );
}

function responseCompiledNote(coverage: EvidenceCoverage) {
  if (!coverage.has_compiled_context) {
    return null;
  }
  return <p className="diagnostic-note">Compiled spec context was used to sharpen the answer outline.</p>;
}
