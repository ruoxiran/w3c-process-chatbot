# Multi-stage Next.js build. Stage 1 installs all deps and builds the
# static assets + .next directory; stage 2 ships only the runtime
# essentials. Cuts the final image from ~1.2 GB to ~300 MB.

FROM node:22-alpine AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json* /app/
COPY packages/ui /app/packages/ui
COPY apps/web /app/apps/web
RUN npm install --no-audit --no-fund
RUN npm --prefix apps/web run build


FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

# Copy only what's needed at runtime — drop dev tooling, source maps,
# and intermediate build artefacts.
COPY --from=builder /app/package.json /app/package-lock.json* /app/
COPY --from=builder /app/packages /app/packages
COPY --from=builder /app/apps/web/.next /app/apps/web/.next
COPY --from=builder /app/apps/web/public /app/apps/web/public
COPY --from=builder /app/apps/web/package.json /app/apps/web/package.json
COPY --from=builder /app/apps/web/next.config.ts /app/apps/web/next.config.ts
COPY --from=builder /app/node_modules /app/node_modules
COPY --from=builder /app/apps/web/node_modules /app/apps/web/node_modules

EXPOSE 3000
CMD ["npm", "--prefix", "apps/web", "run", "start"]
