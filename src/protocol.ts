/**
 * QMD Daemon Protocol - Shared types for daemon communication
 *
 * Protocol: newline-delimited JSON (NDJSON) over Unix socket
 * Socket path: ~/.cache/qmd/qmd.sock
 * PID file: ~/.cache/qmd/daemon.pid
 */

import { homedir } from "./store.js";
import { join } from "node:path";
import type { OutputFormat } from "./formatter.js";

// =============================================================================
// Paths
// =============================================================================

const CACHE_BASE = Bun.env.XDG_CACHE_HOME || join(homedir(), ".cache");
export const CACHE_DIR = join(CACHE_BASE, "qmd");
export const SOCKET_PATH = join(CACHE_DIR, "qmd.sock");
export const PID_PATH = join(CACHE_DIR, "daemon.pid");

// =============================================================================
// Command Argument Types
// =============================================================================

/**
 * Arguments for search, vsearch, and query commands
 */
export type SearchArgs = {
  query: string;
  limit?: number;
  minScore?: number;
  all?: boolean;
  collection?: string;
  full?: boolean;
  format?: OutputFormat;
  dbPath?: string;
  useColor?: boolean;
  context?: string;
};

/**
 * Arguments for get command
 */
export type GetArgs = {
  path: string;
  fromLine?: number;
  maxLines?: number;
  dbPath?: string;
  cwd?: string;
};

/**
 * Arguments for multi-get command
 */
export type MultiGetArgs = {
  pattern: string;
  maxLines?: number;
  maxBytes?: number;
  dbPath?: string;
};

/**
 * Arguments for ls command
 */
export type LsArgs = {
  path?: string;
  dbPath?: string;
};

// =============================================================================
// Protocol Types
// =============================================================================

/**
 * Request sent to daemon - discriminated union for type safety
 */
export type DaemonRequest =
  | { cmd: "search"; args: SearchArgs }
  | { cmd: "vsearch"; args: SearchArgs }
  | { cmd: "query"; args: SearchArgs }
  | { cmd: "get"; args: GetArgs }
  | { cmd: "multi-get"; args: MultiGetArgs }
  | { cmd: "ls"; args: LsArgs }
  | { cmd: "status"; args: { dbPath?: string } }
  | { cmd: "ping"; args: Record<string, never> }
  | { cmd: "daemon-status"; args: Record<string, never> };

/**
 * Command names derived from DaemonRequest union
 */
export type DaemonCommandName = DaemonRequest["cmd"];

/**
 * Generic request type for dynamic command dispatch
 * Use DaemonRequest for type-safe construction, this for parsing
 */
export type DaemonRequestGeneric =
  | { cmd: "search"; args: Record<string, unknown> }
  | { cmd: "vsearch"; args: Record<string, unknown> }
  | { cmd: "query"; args: Record<string, unknown> }
  | { cmd: "get"; args: Record<string, unknown> }
  | { cmd: "multi-get"; args: Record<string, unknown> }
  | { cmd: "ls"; args: Record<string, unknown> }
  | { cmd: "status"; args: Record<string, unknown> }
  | { cmd: "ping"; args: Record<string, unknown> }
  | { cmd: "daemon-status"; args: Record<string, unknown> };

// =============================================================================
// Response Types
// =============================================================================

export type DaemonSearchResult = {
  file: string;
  displayPath: string;
  title: string;
  score: number;
  hash?: string;
  docid?: string;
  chunkPos?: number;
  body?: string;
  context?: string | null;
};

export type DaemonSearchResponse = {
  results: DaemonSearchResult[];
  stderr?: string[];
};

export type DaemonGetResponse = {
  file: string;
  title: string;
  body: string;
  context: string | null;
  hash: string;
  docid: string;
  startLine: number;
};

export type DaemonMultiGetItem = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  context: string | null;
  skipped: boolean;
  skipReason?: string;
};

export type DaemonMultiGetResponse = {
  results: DaemonMultiGetItem[];
  errors: string[];
};

export type DaemonLsCollectionsResponse = {
  mode: "collections";
  collections: { name: string; fileCount: number }[];
};

export type DaemonLsFilesResponse = {
  mode: "files";
  collectionName: string;
  pathPrefix: string | null;
  files: { path: string; title: string; modifiedAt: string; size: number }[];
};

export type DaemonLsResponse = DaemonLsCollectionsResponse | DaemonLsFilesResponse;

export type DaemonIndexStatus = {
  collections: { name: string; pattern: string; fileCount: number; lastModified: string }[];
  totalDocs: number;
  vectorCount: number;
  needsEmbedding: number;
  mostRecent: string | null;
  contexts: { collection: string; path: string; context: string }[];
};

export type DaemonPingResponse = {
  pong: true;
  pid: number;
};

export type DaemonResult =
  | DaemonSearchResponse
  | DaemonGetResponse
  | DaemonMultiGetResponse
  | DaemonLsResponse
  | DaemonIndexStatus
  | DaemonStatusRunning
  | DaemonPingResponse;

/**
 * Success response from daemon
 */
export type DaemonSuccessResponse<Result extends DaemonResult = DaemonResult> = {
  ok: true;
  result: Result;
};

/**
 * Error response from daemon
 */
export type DaemonErrorResponse = {
  ok: false;
  error: string;
};

/**
 * Response from daemon
 */
export type DaemonResponse = DaemonSuccessResponse | DaemonErrorResponse;

/**
 * Daemon status when running
 */
export type DaemonStatusRunning = {
  running: true;
  pid: number;
  uptime: number; // seconds
  loadedModels: string[];
  activeConnections: number;
};

/**
 * Daemon status when stopped
 */
export type DaemonStatus = DaemonStatusRunning | { running: false };

// =============================================================================
// Command Classification
// =============================================================================

/**
 * Commands that the daemon handles (immutable set)
 */
const DAEMON_COMMANDS_LIST = [
  "search",
  "vsearch",
  "query",
  "get",
  "multi-get",
  "ls",
  "status",
  "ping",
  "daemon-status",
] as const;

export const DAEMON_COMMANDS: ReadonlySet<string> = new Set(DAEMON_COMMANDS_LIST);

export function isDaemonRequestGeneric(value: unknown): value is DaemonRequestGeneric {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  if (typeof v.cmd !== "string" || !DAEMON_COMMANDS.has(v.cmd)) return false;
  if (typeof v.args !== "object" || v.args === null || Array.isArray(v.args)) return false;
  return true;
}

/**
 * Commands that should NOT use the daemon (mutate state or are long-running)
 */
const NON_DAEMON_COMMANDS_LIST = [
  "daemon",       // Meta command
  "collection",   // Mutates index
  "context",      // Mutates config
  "embed",        // Long-running batch
  "update",       // Long-running batch
  "mcp",          // Separate server
  "cleanup",      // Mutates DB
] as const;

export const NON_DAEMON_COMMANDS: ReadonlySet<string> = new Set(NON_DAEMON_COMMANDS_LIST);

/**
 * Decide whether a CLI command should be routed through the daemon.
 * Returns true for explicitly supported read-only commands.
 */
export function shouldUseDaemon(cmd: string): boolean {
  // Explicit allow list takes priority
  if (DAEMON_COMMANDS.has(cmd)) return true;
  // Explicit deny list
  if (NON_DAEMON_COMMANDS.has(cmd)) return false;
  // Unknown commands don't use daemon
  return false;
}
