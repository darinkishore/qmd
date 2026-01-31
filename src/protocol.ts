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
export type DaemonRequestGeneric = {
  cmd: string;
  args: Record<string, unknown>;
};

/**
 * Success response from daemon
 */
export type DaemonSuccessResponse = {
  ok: true;
  result: unknown;
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
export type DaemonStatusStopped = {
  running: false;
};

/**
 * Status info returned by daemon status command
 */
export type DaemonStatus = DaemonStatusRunning | DaemonStatusStopped;

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
 * Check if a command should use the daemon
 */
export function shouldUseDaemon(cmd: string): boolean {
  // Explicit allow list takes priority
  if (DAEMON_COMMANDS.has(cmd)) return true;
  // Explicit deny list
  if (NON_DAEMON_COMMANDS.has(cmd)) return false;
  // Unknown commands don't use daemon
  return false;
}
