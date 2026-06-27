import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: {
    alias: {
      // Mirror tsconfig's ``@/*`` path alias so imports work identically
      // in tests and in the Next.js build.
      "@": fileURLToPath(new URL("./", import.meta.url)),
    },
  },
  test: {
    include: ["**/*.test.ts", "**/*.test.tsx"],
    exclude: ["node_modules", ".next", "dist"],
    // jsdom is heavy and not needed for the SSE-parser + fetch-mock
    // tests we're starting with. Switch to ``environment: "jsdom"``
    // when the suite grows to cover React components.
    environment: "node",
  },
});
