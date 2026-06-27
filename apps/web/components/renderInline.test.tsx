/**
 * Tests for the answer-body inline renderer.
 *
 * The renderer is the only place the frontend turns model text into
 * clickable links, so it carries the XSS surface for the whole
 * application. These tests pin the grammar AND the safe-scheme
 * whitelist so a future change can't widen the surface accidentally.
 */

import { describe, expect, test } from "vitest";
import { isValidElement, type ReactElement, type ReactNode } from "react";
import {
  isSafeActionLink,
  renderInline,
} from "./ChatInterface";

// ---------- isSafeActionLink whitelist -------------------------------------

describe("isSafeActionLink", () => {
  test.each([
    ["https://github.com/w3c/i18n-request/issues/new/choose", true],
    ["http://example.com", true],
    ["mailto:chairs@example.org", true],
    ["HTTPS://www.w3.org", true],
    ["MailTo:user@example.com", true],
    // Hostile schemes — must be rejected.
    ["javascript:alert(1)", false],
    ["data:text/html,<script>alert(1)</script>", false],
    ["vbscript:msgbox", false],
    ["file:///etc/passwd", false],
    ["//evil.com/path", false],
    ["/relative/path", false],
    ["", false],
  ])("isSafeActionLink(%j) → %j", (input, expected) => {
    expect(isSafeActionLink(input)).toBe(expected);
  });
});

// ---------- renderInline grammar -------------------------------------------

// React's typed ``props`` is ``unknown`` on the generic ReactElement;
// every assertion below needs to look up specific props (href, className,
// children, target). Narrow once here so the test bodies can use a
// plain anchor-prop shape.
type AnchorElement = ReactElement<{
  href?: string;
  className?: string;
  children?: ReactNode;
  target?: string;
  rel?: string;
  title?: string;
}>;

function elementsOfType(nodes: ReactNode[], tag: string): AnchorElement[] {
  return nodes.filter(
    (node): node is AnchorElement =>
      isValidElement(node) && (node.type as unknown) === tag
  );
}

function toArray(nodes: ReactNode): ReactNode[] {
  return Array.isArray(nodes) ? nodes : [nodes];
}

describe("renderInline", () => {
  test("tolerates whitespace around the URL inside markdown link parens", () => {
    // Models sometimes wrap the URL with leading or trailing
    // whitespace, especially after a line break:
    // ``[label]( https://… )``. The renderer must still produce an
    // anchor with the trimmed URL — otherwise these render as inert
    // literal text.
    const nodes = toArray(renderInline(
      "Open [the i18n tracker]( https://github.com/w3c/i18n-request/issues/new/choose ).",
      [],
    ));
    const anchors = elementsOfType(nodes, "a");
    expect(anchors).toHaveLength(1);
    expect(anchors[0].props.href).toBe(
      "https://github.com/w3c/i18n-request/issues/new/choose",
    );
  });

  test("renders a safe markdown link as an <a> with the label text", () => {
    const nodes = toArray(renderInline(
      "File at [the i18n tracker](https://github.com/w3c/i18n-request/issues/new/choose).",
      [],
    ));
    const anchors = elementsOfType(nodes, "a");
    expect(anchors).toHaveLength(1);
    expect(anchors[0].props.href).toBe(
      "https://github.com/w3c/i18n-request/issues/new/choose",
    );
    expect(anchors[0].props.children).toBe("the i18n tracker");
    expect(anchors[0].props.className).toBe("action-link");
    expect(anchors[0].props.target).toBe("_blank");
    expect(anchors[0].props.rel).toBe("noreferrer");
  });

  test("renders mailto: links without target=_blank (opens in the user's mail app)", () => {
    const nodes = toArray(renderInline(
      "Email [the chairs](mailto:chairs@example.org).",
      [],
    ));
    const anchors = elementsOfType(nodes, "a");
    expect(anchors).toHaveLength(1);
    expect(anchors[0].props.href).toBe("mailto:chairs@example.org");
    expect(anchors[0].props.target).toBeUndefined();
  });

  test("does NOT render javascript: as an anchor — falls back to literal text", () => {
    const nodes = toArray(renderInline(
      "Click [here](javascript:alert(1)) to win!",
      [],
    ));
    const anchors = elementsOfType(nodes, "a");
    expect(anchors).toHaveLength(0);
    // The raw text is preserved so the user can see what the model
    // tried to put on the page.
    const text = nodes.filter((n): n is string => typeof n === "string").join("");
    expect(text).toContain("[here](javascript:alert(1))");
  });

  test("does NOT render data: URLs as anchors", () => {
    const nodes = toArray(renderInline(
      "[malicious](data:text/html,<script>alert(1)</script>)",
      [],
    ));
    expect(elementsOfType(nodes, "a")).toHaveLength(0);
  });

  test("renders citation refs like [S1] using the matching Citation.url", () => {
    const citations = [
      {
        title: "Process Document",
        url: "https://www.w3.org/policies/process/#charter-approval",
        source_type: "process" as const,
        heading_path: "Charter Approval",
      },
    ];
    const nodes = toArray(renderInline("As required by Process [S1].", citations));
    const anchors = elementsOfType(nodes, "a");
    expect(anchors).toHaveLength(1);
    expect(anchors[0].props.href).toBe(
      "https://www.w3.org/policies/process/#charter-approval",
    );
    // React renders ``S{index}`` as the array ``["S", 1]`` — concatenated
    // at render time. Normalise before asserting so the test doesn't
    // depend on JSX's internal representation.
    const text = (Array.isArray(anchors[0].props.children)
      ? anchors[0].props.children.join("")
      : String(anchors[0].props.children));
    expect(text).toBe("S1");
    expect(anchors[0].props.className).toContain("citation-ref");
  });

  test("renders both an action link AND a citation ref in the same line", () => {
    const citations = [{ title: "Guide", url: "https://www.w3.org/guide/", source_type: "guide" as const }];
    const nodes = toArray(renderInline(
      "File [an i18n request](https://github.com/w3c/i18n-request/issues/new/choose) per the guide [S1].",
      citations,
    ));
    const anchors = elementsOfType(nodes, "a");
    expect(anchors).toHaveLength(2);
    // Action link comes before the citation ref in source order.
    expect(anchors[0].props.className).toBe("action-link");
    expect(anchors[1].props.className).toContain("citation-ref");
  });

  test("renders **bold** + `code` alongside links without breaking either", () => {
    const nodes = toArray(renderInline(
      "Use the **AC mailing list** [here](https://lists.w3.org/) — the `ac-forum` archive is public.",
      [],
    ));
    const strongs = elementsOfType(nodes, "strong");
    const codes = elementsOfType(nodes, "code");
    const anchors = elementsOfType(nodes, "a");
    expect(strongs).toHaveLength(1);
    expect(codes).toHaveLength(1);
    expect(anchors).toHaveLength(1);
  });
});
