# Multi-stage build for AI/ML Engineer Portfolio
# Production-ready Docker container with optimization

# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (dev deps required for the build step)
RUN npm ci && npm cache clean --force

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Add non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Copy built application from builder stage
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Set ownership
RUN chown -R nextjs:nodejs /app

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1

# Start the application
CMD ["node", "server.js"]

# Build arguments for optimization
ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION=latest

# Labels for container metadata
LABEL maintainer="komal@komalshahid.dev" \
      version=${VERSION} \
      description="AI/ML Engineer Portfolio - Production Container" \
      build-date=${BUILD_DATE} \
      git-commit=${GIT_COMMIT}