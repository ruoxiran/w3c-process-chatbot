FROM node:22-slim

WORKDIR /app
COPY package.json package-lock.json* /app/
COPY packages/ui /app/packages/ui
COPY apps/web /app/apps/web
RUN npm install
RUN npm --prefix apps/web run build

EXPOSE 3000
CMD ["npm", "--prefix", "apps/web", "run", "start"]

