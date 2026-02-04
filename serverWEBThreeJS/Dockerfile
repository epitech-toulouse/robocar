# Stage 1: Build React application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy client package files and install dependencies
COPY client/package*.json ./client/
WORKDIR /app/client
RUN npm install

# Copy client source and build
COPY client/ ./
RUN npm run build

# Stage 2: Production server
FROM node:20-alpine

# Install dependencies for serialport (native module compilation)
RUN apk add --no-cache python3 make g++ linux-headers eudev-dev

WORKDIR /app

# Copy server package files
COPY package*.json ./
RUN npm install --production

# Copy server code
COPY server.js ./

# Copy built React app from builder stage
COPY --from=builder /app/dist ./dist

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/api/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Start server
CMD ["npm", "start"]

