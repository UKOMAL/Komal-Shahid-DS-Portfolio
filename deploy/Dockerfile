# Multi-stage Dockerfile for Komal Shahid AI Portfolio
# Stack: Next.js (Node 20) — production build
#
# Usage:
#   docker build -t komalshahid/portfolio:latest -f deploy/Dockerfile .
#   docker run -p 3000:3000 komalshahid/portfolio:latest
#
# For the FastAPI ML demo services, see projects/project13-dsc680/deploy/Dockerfile

# ── Stage 1: Dependencies ─────────────────────────────────────────────
FROM node:20-alpine AS deps
LABEL stage="deps"

RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install production + dev dependencies (dev needed for build)
RUN npm ci

# ── Stage 2: Build ────────────────────────────────────────────────────
FROM node:20-alpine AS builder
LABEL stage="builder"

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Disable telemetry during build
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# ── Stage 3: Production runner ─────────────────────────────────────────
FROM node:20-alpine AS runner
LABEL maintainer="Komal Shahid <komal.shahid@example.com>"
LABEL org.opencontainers.image.title="Komal Shahid AI Portfolio"
LABEL org.opencontainers.image.source="https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio"

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid 1001 nextjs

# Copy only production artifacts
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD wget -qO- http://localhost:3000/api/health || exit 1

CMD ["node", "server.js"]
