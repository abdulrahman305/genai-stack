FROM node:23.11.1-alpine3.21

WORKDIR /app

COPY front-end/ .

RUN npm install

EXPOSE 8505

ENTRYPOINT [ "npm", "run", "dev" ]
