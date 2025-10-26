/**
 * @file This file configures and exports a singleton Axios instance for making
 * HTTP requests to the backend API.
 *
 * The base URL is configured to `/api`, which is handled by the Vite development
 * server's proxy to forward requests to the FastAPI backend. This setup ensures
 * that API calls are correctly routed in both development and production
 * environments.
 */
import axios from 'axios';

const client = axios.create({
  baseURL: '/api',
  timeout: 1000 * 60,
});

export default client;
