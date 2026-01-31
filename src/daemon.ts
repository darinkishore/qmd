#!/usr/bin/env bun
/**
 * QMD Daemon - Unix socket server for fast queries
 *
 * Keeps models loaded in VRAM for sub-100ms response times.
 * Protocol: NDJSON over Unix socket
 */

import { existsSync, mkdirSync, unlinkSync, writeFileSync, readFileSync } from "node:fs";
import type { Database } from "bun:sqlite";
import type { TCPSocketListener } from "bun";
import {
  SOCKET_PATH,
  PID_PATH,
  CACHE_DIR,
  DAEMON_COMMANDS,
  isDaemonRequestGeneric,
  type DaemonRequest,
  type DaemonStatus,
  type DaemonStatusRunning,
  type DaemonRequestGeneric,
  type DaemonResponse,
  type DaemonResult,
  type DaemonSearchResult,
  type DaemonSearchResponse,
  type DaemonGetResponse,
  type DaemonMultiGetResponse,
  type DaemonLsResponse,
  type DaemonIndexStatus,
  type DaemonPingResponse,
  type SearchArgs,
  type GetArgs,
  type MultiGetArgs,
  type LsArgs,
} from "./protocol.js";
import {
  createStore,
  getDefaultDbPath,
  getDocid,
  searchFTS,
  searchVec,
  chunkDocument,
  getContextForPath,
  matchFilesByGlob,
  findDocumentByDocid,
  isDocid,
  parseVirtualPath,
  buildVirtualPath,
  isVirtualPath,
  getHashesNeedingEmbedding,
  getIndexHealth,
  listCollections,
  homedir,
  resolve,
  getRealPath,
  getPwd,
  DEFAULT_EMBED_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
  reciprocalRankFusion,
  enableProductionMode,
} from "./store.js";
import type { SearchResult } from "./store.js";
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp, type Queryable, type RerankDocument } from "./llm.js";
import {
  getCollection as getCollectionFromYaml,
  listCollections as yamlListCollections,
  listAllContexts,
} from "./collections.js";

// =============================================================================
// Daemon State
// =============================================================================

const stores = new Map<string, ReturnType<typeof createStore>>();
let server: TCPSocketListener<DaemonSocketData> | null = null;
let startTime: number = 0;
let activeConnections = 0;
let shuttingDown = false;
let signalHandlersRegistered = false;
const TRACE = Bun.env.QMD_DAEMON_TRACE === "1";

// Track which models have been loaded
const loadedModels = new Set<string>();

// Enable production mode - allows using default database path
enableProductionMode();

const DEFAULT_SEARCH_LIMIT = 20;
const DEFAULT_QUERY_LIMIT = 5;
const VSEARCH_PER_QUERY_LIMIT = 20;
const VSEARCH_ALL_PER_QUERY_LIMIT = 500;
const QUERY_FTS_LIMIT = 20;
const QUERY_VEC_LIMIT = 20;
const QUERY_RERANK_LIMIT = 40;
const BM25_STRONG_SCORE = 0.85;
const BM25_STRONG_GAP = 0.15;
const MIN_QUERY_TERM_LENGTH = 3;
const RERANK_TEXT_MAX_CHARS = 4000;
const ERROR_SNIPPET_LIMIT = 120;
const SOCKET_BUFFER_LIMIT = 1024 * 1024 * 4;
const SOCKET_ERROR_PARTIAL_LIMIT = 50;
const QUERY_EXPANSION_PREVIEW_LIMIT = 80;
const RRF_DEFAULT_RANK = 30;
const RRF_WEIGHTS = [
  { maxRank: 3, weight: 0.75 },
  { maxRank: 10, weight: 0.6 },
  { maxRank: Number.POSITIVE_INFINITY, weight: 0.4 },
] as const;
const RRF_PRIMARY_WEIGHT = 2;
const RRF_SECONDARY_WEIGHT = 1;

type LineFramer = {
  push: (chunk: string) => { lines: string[]; overflow: boolean };
  getBuffer: () => string;
};

type DaemonSocketData = {
  framer: LineFramer;
  cleaned?: boolean; // Track if connection was already cleaned up
};

function registerSignalHandlers(): void {
  if (signalHandlersRegistered) return;
  signalHandlersRegistered = true;

  process.on("SIGTERM", () => stopServer());
  process.on("SIGINT", () => stopServer());

  // Global error handlers to catch unhandled errors
  process.on("unhandledRejection", (reason) => {
    console.error("[FATAL] Unhandled promise rejection:", reason);
    stopServer();
  });

  process.on("uncaughtException", (err) => {
    console.error("[FATAL] Uncaught exception:", err);
    stopServer();
  });
}

// =============================================================================
// Store Management
// =============================================================================

function resolveDbPath(dbPath?: string): string {
  return dbPath || getDefaultDbPath();
}

function getStore(dbPath?: string): ReturnType<typeof createStore> {
  const resolved = resolveDbPath(dbPath);
  const existing = stores.get(resolved);
  if (existing) return existing;
  const created = createStore(resolved);
  stores.set(resolved, created);
  return created;
}

function getDb(dbPath?: string): Database {
  return getStore(dbPath).db;
}

// =============================================================================
// Command Handlers
// =============================================================================

type LogBuffer = {
  stderr: string[];
  write: (text: string) => void;
  c: {
    reset: string;
    dim: string;
    bold: string;
    cyan: string;
    yellow: string;
    green: string;
    magenta: string;
    blue: string;
  };
  progress: {
    set: (percent: number) => void;
    clear: () => void;
    indeterminate: () => void;
    error: () => void;
  };
};

function truncateLine(text: string, maxLength: number): string {
  return text.length > maxLength ? text.slice(0, maxLength) : text;
}

function formatLineSnippet(line: string): string {
  return truncateLine(line, ERROR_SNIPPET_LIMIT);
}

function formatParseError(prefix: string, message: string, line: string): string {
  return `${prefix}: ${message}. Line starts with: ${formatLineSnippet(line)}`;
}

function createErrorWithCause(message: string, cause: unknown): Error {
  return new Error(message, { cause });
}

function createLogColors(useColor: boolean): LogBuffer["c"] {
  return {
    reset: useColor ? "\x1b[0m" : "",
    dim: useColor ? "\x1b[2m" : "",
    bold: useColor ? "\x1b[1m" : "",
    cyan: useColor ? "\x1b[36m" : "",
    yellow: useColor ? "\x1b[33m" : "",
    green: useColor ? "\x1b[32m" : "",
    magenta: useColor ? "\x1b[35m" : "",
    blue: useColor ? "\x1b[34m" : "",
  };
}

function createProgressLogger(write: (text: string) => void): LogBuffer["progress"] {
  return {
    set(percent: number) {
      write(`\x1b]9;4;1;${Math.round(percent)}\x07`);
    },
    clear() {
      write(`\x1b]9;4;0\x07`);
    },
    indeterminate() {
      write(`\x1b]9;4;3\x07`);
    },
    error() {
      write(`\x1b]9;4;2\x07`);
    },
  };
}

function createLogBuffer(useColor: boolean | undefined): LogBuffer {
  const c = createLogColors(!!useColor);
  const stderr: string[] = [];
  const write = (text: string) => {
    stderr.push(text);
  };
  const progress = createProgressLogger(write);
  return { stderr, write, c, progress };
}

function checkIndexHealthWithLogs(db: Database, logger: LogBuffer): void {
  const { needsEmbedding, totalDocs, daysStale } = getIndexHealth(db);

  if (needsEmbedding > 0) {
    const pct = Math.round((needsEmbedding / totalDocs) * 100);
    if (pct >= 10) {
      logger.write(`${logger.c.yellow}Warning: ${needsEmbedding} documents (${pct}%) need embeddings. Run 'qmd embed' for better results.${logger.c.reset}\n`);
    } else {
      logger.write(`${logger.c.dim}Tip: ${needsEmbedding} documents need embeddings. Run 'qmd embed' to index them.${logger.c.reset}\n`);
    }
  }

  if (daysStale !== null && daysStale >= 14) {
    logger.write(`${logger.c.dim}Tip: Index last updated ${daysStale} days ago. Run 'qmd update' to refresh.${logger.c.reset}\n`);
  }
}

function previewQueryText(text: string): string {
  const flattened = text.replace(/\n/g, " ");
  if (flattened.length <= QUERY_EXPANSION_PREVIEW_LIMIT) return flattened;
  return flattened.slice(0, Math.max(0, QUERY_EXPANSION_PREVIEW_LIMIT - 3)) + "...";
}

function formatQueryExpansionLines(
  query: string,
  includeLexical: boolean,
  queryables: Queryable[],
  colors: LogBuffer["c"]
): string[] {
  const lines: string[] = [];
  const modeLabel = includeLexical ? " · (lexical+vector)" : " · (vector)";
  lines.push(`${colors.dim}├─ ${query}${modeLabel}${colors.reset}`);

  for (const q of queryables) {
    if (!q || q.text === query) continue;
    const label = q.type === "lex" ? "lexical" : (q.type === "hyde" ? "hyde" : "vector");
    lines.push(`${colors.dim}├─ ${previewQueryText(q.text)} · (${label})${colors.reset}`);
  }

  if (lines.length > 0) {
    lines[lines.length - 1] = lines[lines.length - 1]!.replace("├─", "└─");
  }
  return lines;
}

async function expandQueryStructuredWithLogs(
  query: string,
  includeLexical: boolean,
  context: string | undefined,
  logger: LogBuffer
): Promise<Queryable[]> {
  logger.write(`${logger.c.dim}Expanding query...${logger.c.reset}\n`);

  const llm = getDefaultLlamaCpp();
  const queryables = await llm.expandQuery(query, { includeLexical, context });

  const lines = formatQueryExpansionLines(query, includeLexical, queryables, logger.c);
  for (const line of lines) {
    logger.write(line + "\n");
  }

  return queryables;
}

async function rerankWithLogs(
  query: string,
  documents: { file: string; text: string }[],
  logger: LogBuffer
): Promise<{ file: string; score: number }[]> {
  if (documents.length === 0) return [];

  const total = documents.length;
  logger.write(`Reranking ${total} documents...\n`);
  logger.progress.indeterminate();

  const llm = getDefaultLlamaCpp();
  const rerankDocs: RerankDocument[] = documents.map((doc) => ({
    file: doc.file,
    text: doc.text.slice(0, RERANK_TEXT_MAX_CHARS),
  }));

  const result = await llm.rerank(query, rerankDocs, { model: DEFAULT_RERANK_MODEL });

  logger.progress.clear();
  logger.write("\n");

  return result.results.map((r) => ({ file: r.file, score: r.score }));
}

// =============================================================================
// Input Validation Helpers
// =============================================================================

function describeValue(value: unknown): string {
  if (value === undefined) return "undefined";
  if (value === null) return "null";
  if (typeof value === "string") return `"${value}"`;
  if (typeof value === "number" && Number.isNaN(value)) return "NaN";
  return String(value);
}

function requireNonEmptyString(value: unknown, name: string, hint: string): string {
  if (value === undefined || value === null) {
    throw new Error(`Missing required argument: ${name}. ${hint}`);
  }
  if (typeof value !== "string") {
    throw new Error(`Invalid argument: ${name} must be a non-empty string (received ${describeValue(value)}). ${hint}`);
  }
  if (!value.trim()) {
    throw new Error(`Missing required argument: ${name}. ${hint}`);
  }
  return value;
}

function optionalNonEmptyString(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") {
    throw new Error(`Invalid argument: ${name} must be a non-empty string (received ${describeValue(value)}).`);
  }
  if (!value.trim()) {
    throw new Error(`Invalid argument: ${name} must be a non-empty string (received ${describeValue(value)}).`);
  }
  return value;
}

function optionalBoolean(value: unknown, name: string): boolean | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "boolean") {
    throw new Error(`Invalid argument: ${name} must be a boolean (received ${describeValue(value)}).`);
  }
  return value;
}

function optionalNumber(
  value: unknown,
  name: string,
  opts: { min?: number; max?: number; integer?: boolean } = {}
): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid argument: ${name} must be a finite number (received ${describeValue(value)}).`);
  }
  if (opts.integer && !Number.isInteger(value)) {
    throw new Error(`Invalid argument: ${name} must be an integer (received ${describeValue(value)}).`);
  }
  if (opts.min !== undefined && value < opts.min) {
    throw new Error(`Invalid argument: ${name} must be >= ${opts.min} (received ${describeValue(value)}).`);
  }
  if (opts.max !== undefined && value > opts.max) {
    throw new Error(`Invalid argument: ${name} must be <= ${opts.max} (received ${describeValue(value)}).`);
  }
  return value;
}

/**
 * Validate and normalize arguments for search/vsearch/query commands.
 */
export function validateSearchArgs(args: unknown): SearchArgs {
  const a = (args && typeof args === "object") ? (args as Record<string, unknown>) : {};
  const query = requireNonEmptyString(a.query, "query", "Provide a non-empty search string.");
  return {
    query: query.trim(),
    limit: optionalNumber(a.limit, "limit", { min: 1, integer: true }),
    minScore: optionalNumber(a.minScore, "minScore", { min: 0, max: 1 }),
    all: optionalBoolean(a.all, "all"),
    collection: optionalNonEmptyString(a.collection, "collection"),
    full: optionalBoolean(a.full, "full"),
    dbPath: optionalNonEmptyString(a.dbPath, "dbPath"),
    useColor: optionalBoolean(a.useColor, "useColor"),
    context: optionalNonEmptyString(a.context, "context"),
  };
}

/**
 * Validate and normalize arguments for get command.
 */
export function validateGetArgs(args: unknown): GetArgs {
  const a = (args && typeof args === "object") ? (args as Record<string, unknown>) : {};
  const path = requireNonEmptyString(a.path, "path", "Provide a file path or qmd:// virtual path.");
  return {
    path,
    fromLine: optionalNumber(a.fromLine, "fromLine", { min: 1, integer: true }),
    maxLines: optionalNumber(a.maxLines, "maxLines", { min: 1, integer: true }),
    dbPath: optionalNonEmptyString(a.dbPath, "dbPath"),
    cwd: optionalNonEmptyString(a.cwd, "cwd"),
  };
}

/**
 * Validate and normalize arguments for multi-get command.
 */
export function validateMultiGetArgs(args: unknown): MultiGetArgs {
  const a = (args && typeof args === "object") ? (args as Record<string, unknown>) : {};
  const pattern = requireNonEmptyString(
    a.pattern,
    "pattern",
    "Provide a glob pattern or a comma-separated list of files."
  );
  return {
    pattern,
    maxLines: optionalNumber(a.maxLines, "maxLines", { min: 1, integer: true }),
    maxBytes: optionalNumber(a.maxBytes, "maxBytes", { min: 1, integer: true }),
    dbPath: optionalNonEmptyString(a.dbPath, "dbPath"),
  };
}

/**
 * Validate and normalize arguments for ls command.
 */
export function validateLsArgs(args: unknown): LsArgs {
  const a = (args && typeof args === "object") ? (args as Record<string, unknown>) : {};
  return {
    path: optionalNonEmptyString(a.path, "path"),
    dbPath: optionalNonEmptyString(a.dbPath, "dbPath"),
  };
}

/**
 * Validate and normalize arguments for status command.
 */
export function validateStatusArgs(args: unknown): { dbPath?: string } {
  const a = (args && typeof args === "object") ? (args as Record<string, unknown>) : {};
  return {
    dbPath: optionalNonEmptyString(a.dbPath, "dbPath"),
  };
}

function requireEmptyArgs(args: Record<string, unknown>, cmd: string): void {
  if (Object.keys(args).length > 0) {
    throw new Error(`Invalid argument: ${cmd} does not accept any arguments.`);
  }
}

function resolveCollectionName(collection?: string): string | undefined {
  if (!collection) return undefined;
  const coll = getCollectionFromYaml(collection);
  if (!coll) {
    throw new Error(`Collection not found: ${collection}. Run 'qmd collection list' to see available collections.`);
  }
  return collection;
}

function hasVectorIndex(db: Database): boolean {
  return !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
}

function getContextForFilePath(db: Database, filepath: string): string | null {
  const parsed = parseVirtualPath(filepath);
  if (!parsed) return null;
  return getContextForPath(db, parsed.collectionName, parsed.path);
}

function toDaemonSearchResult(result: SearchResult): DaemonSearchResult {
  return {
    file: result.filepath,
    displayPath: result.displayPath,
    title: result.title,
    score: result.score,
    hash: result.hash,
    docid: result.docid,
    chunkPos: result.chunkPos,
    body: result.body,
    context: result.context,
  };
}

function applySearchConstraints<T extends { score: number }>(
  results: T[],
  limit: number,
  minScore?: number
): T[] {
  const filtered = minScore !== undefined
    ? results.filter((result) => result.score >= minScore)
    : results;
  return filtered.slice(0, limit);
}

function shouldIncludeBody(full?: boolean): boolean {
  return full !== false;
}

function stripBodyIfNeeded(results: DaemonSearchResult[], includeBody: boolean): DaemonSearchResult[] {
  if (includeBody) return results;
  return results.map(({ body: _body, ...rest }) => rest);
}

// =============================================================================
// Path Resolution Helpers
// =============================================================================

function detectCollectionFromPath(fsPath: string): { collectionName: string; relativePath: string } | null {
  const realPath = getRealPath(fsPath);

  const allCollections = yamlListCollections();

  let bestMatch: { name: string; path: string } | null = null;
  for (const coll of allCollections) {
    if (realPath.startsWith(coll.path + "/") || realPath === coll.path) {
      if (!bestMatch || coll.path.length > bestMatch.path.length) {
        bestMatch = { name: coll.name, path: coll.path };
      }
    }
  }

  if (!bestMatch) return null;

  let relativePath = realPath;
  if (relativePath.startsWith(bestMatch.path + "/")) {
    relativePath = relativePath.slice(bestMatch.path.length + 1);
  } else if (relativePath === bestMatch.path) {
    relativePath = "";
  }

  return {
    collectionName: bestMatch.name,
    relativePath,
  };
}

type DocRow = {
  collectionName: string;
  path: string;
  body: string;
  title: string;
  hash: string;
};

type DocMetaRow = {
  virtual_path: string;
  body_length: number;
  collection: string;
  path: string;
};

type DocBodyRow = {
  body: string;
  title: string;
};

type FileCountRow = { file_count: number };
type LsFileRow = { path: string; title: string; modified_at: string; size: number };
type CountRow = { count: number };
type LatestRow = { latest: string | null };

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function readDbRow<T>(row: unknown, guard: (value: unknown) => value is T, label: string): T | null {
  if (row === null || row === undefined) return null;
  if (guard(row)) return row;
  throw new Error(`Unexpected ${label} row shape.`);
}

function readDbRows<T>(rows: unknown[], guard: (value: unknown) => value is T, label: string): T[] {
  const result: T[] = [];
  for (const row of rows) {
    if (!guard(row)) {
      throw new Error(`Unexpected ${label} row shape.`);
    }
    result.push(row);
  }
  return result;
}

function isDocRow(value: unknown): value is DocRow {
  if (!isRecord(value)) return false;
  return (
    typeof value.collectionName === "string" &&
    typeof value.path === "string" &&
    typeof value.body === "string" &&
    typeof value.title === "string" &&
    typeof value.hash === "string"
  );
}

function isDocMetaRow(value: unknown): value is DocMetaRow {
  if (!isRecord(value)) return false;
  return (
    typeof value.virtual_path === "string" &&
    typeof value.body_length === "number" &&
    typeof value.collection === "string" &&
    typeof value.path === "string"
  );
}

function isDocBodyRow(value: unknown): value is DocBodyRow {
  if (!isRecord(value)) return false;
  return typeof value.body === "string" && typeof value.title === "string";
}

function isFileCountRow(value: unknown): value is FileCountRow {
  return isRecord(value) && typeof value.file_count === "number";
}

function isLsFileRow(value: unknown): value is LsFileRow {
  if (!isRecord(value)) return false;
  return (
    typeof value.path === "string" &&
    typeof value.title === "string" &&
    typeof value.modified_at === "string" &&
    typeof value.size === "number"
  );
}

function isCountRow(value: unknown): value is CountRow {
  return isRecord(value) && typeof value.count === "number";
}

function isLatestRow(value: unknown): value is LatestRow {
  return isRecord(value) && (typeof value.latest === "string" || value.latest === null);
}

const DOC_SELECT_SQL = `
  SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
  FROM documents d
  JOIN content ON content.hash = d.hash
  WHERE d.collection = ? AND d.path = ? AND d.active = 1
`;

const DOC_META_BY_COLLECTION_SQL = `
  SELECT
    'qmd://' || d.collection || '/' || d.path as virtual_path,
    LENGTH(content.doc) as body_length,
    d.collection,
    d.path
  FROM documents d
  JOIN content ON content.hash = d.hash
  WHERE d.collection = ? AND d.path = ? AND d.active = 1
`;

const DOC_META_BY_PATH_SQL = `
  SELECT
    'qmd://' || d.collection || '/' || d.path as virtual_path,
    LENGTH(content.doc) as body_length,
    d.collection,
    d.path
  FROM documents d
  JOIN content ON content.hash = d.hash
  WHERE d.path = ? AND d.active = 1
  LIMIT 1
`;

const DOC_BODY_SQL = `
  SELECT content.doc as body, d.title
  FROM documents d
  JOIN content ON content.hash = d.hash
  WHERE d.collection = ? AND d.path = ? AND d.active = 1
`;

function getDocByCollectionPath(db: Database, collectionName: string, path: string): DocRow | null {
  const row = db.prepare(DOC_SELECT_SQL).get(collectionName, path);
  return readDbRow(row, isDocRow, "document");
}

function getDocMeta(db: Database, path: string, collectionName?: string): DocMetaRow | null {
  const row = collectionName
    ? db.prepare(DOC_META_BY_COLLECTION_SQL).get(collectionName, path)
    : db.prepare(DOC_META_BY_PATH_SQL).get(path);
  return readDbRow(row, isDocMetaRow, "document metadata");
}

function getDocBody(db: Database, collectionName: string, path: string): DocBodyRow | null {
  const row = db.prepare(DOC_BODY_SQL).get(collectionName, path);
  return readDbRow(row, isDocBodyRow, "document body");
}

function parsePathWithLineSuffix(path: string, fromLine?: number): { path: string; fromLine?: number } {
  if (fromLine !== undefined) return { path, fromLine };
  const match = path.match(/:(\d+)$/);
  if (!match) return { path, fromLine };
  const parsedLine = Number.parseInt(match[1] ?? "", 10);
  if (!Number.isFinite(parsedLine)) return { path, fromLine };
  return {
    path: path.slice(0, -match[0].length),
    fromLine: parsedLine,
  };
}

function resolveDocidPath(db: Database, inputPath: string, originalPath: string): string {
  if (!isDocid(inputPath)) return inputPath;
  const docidMatch = findDocumentByDocid(db, inputPath);
  if (!docidMatch) {
    throw new Error(`Document not found: ${originalPath}. Try a qmd:// path, a docid (prefix #), or 'qmd ls <collection>'.`);
  }
  return docidMatch.filepath;
}

function parseCollectionPath(inputPath: string): { collectionName: string; path: string } | null {
  if (inputPath.startsWith("/") || inputPath.startsWith("~")) return null;
  const parts = inputPath.split("/");
  if (parts.length < 2) return null;
  const candidate = parts[0];
  if (!candidate) return null;
  const coll = getCollectionFromYaml(candidate);
  if (!coll) return null;
  return { collectionName: candidate, path: parts.slice(1).join("/") };
}

function parseLsPath(inputPath: string): { collectionName: string; pathPrefix: string | null } {
  if (inputPath.startsWith("qmd://")) {
    const parsed = parseVirtualPath(inputPath);
    if (!parsed) {
      throw new Error(`Invalid virtual path: ${inputPath}. Expected format: qmd://<collection>/<path>.`);
    }
    return {
      collectionName: parsed.collectionName,
      pathPrefix: parsed.path || null,
    };
  }

  const parts = inputPath.split("/");
  const collectionName = parts[0] || "";
  const pathPrefix = parts.length > 1 ? parts.slice(1).join("/") : null;
  return { collectionName, pathPrefix };
}

type ResolvedDoc = { doc: DocRow; virtualPath: string };

function resolveDocFromVirtualPath(db: Database, inputPath: string): ResolvedDoc | null {
  if (!isVirtualPath(inputPath)) return null;
  const parsed = parseVirtualPath(inputPath);
  if (!parsed) {
    throw new Error(`Invalid virtual path: ${inputPath}. Expected format: qmd://<collection>/<path>.`);
  }
  const doc = getDocByCollectionPath(db, parsed.collectionName, parsed.path);
  if (!doc) return null;
  return { doc, virtualPath: buildVirtualPath(doc.collectionName, doc.path) };
}

function resolveDocFromCollectionPath(db: Database, inputPath: string): ResolvedDoc | null {
  const collectionPath = parseCollectionPath(inputPath);
  if (!collectionPath) return null;
  const doc = getDocByCollectionPath(db, collectionPath.collectionName, collectionPath.path);
  if (!doc) return null;
  return { doc, virtualPath: buildVirtualPath(doc.collectionName, doc.path) };
}

function resolveFilesystemPath(inputPath: string, cwd?: string): string {
  let fsPath = inputPath;
  if (fsPath.startsWith("~/")) {
    fsPath = homedir() + fsPath.slice(1);
  } else if (!fsPath.startsWith("/")) {
    fsPath = resolve(cwd || getPwd(), fsPath);
  }
  return getRealPath(fsPath);
}

function resolveDocFromFilesystemPath(db: Database, inputPath: string, cwd?: string): ResolvedDoc | null {
  const fsPath = resolveFilesystemPath(inputPath, cwd);
  const detected = detectCollectionFromPath(fsPath);
  if (!detected) return null;
  const doc = getDocByCollectionPath(db, detected.collectionName, detected.relativePath);
  if (!doc) return null;
  return { doc, virtualPath: buildVirtualPath(doc.collectionName, doc.path) };
}

function resolveDocFromInputPath(
  db: Database,
  inputPath: string,
  cwd?: string
): ResolvedDoc | null {
  return (
    resolveDocFromVirtualPath(db, inputPath)
    ?? resolveDocFromCollectionPath(db, inputPath)
    ?? resolveDocFromFilesystemPath(db, inputPath, cwd)
  );
}

function sliceDocumentBody(body: string, fromLine: number, maxLines?: number): string {
  const shouldSlice = fromLine > 1 || maxLines !== undefined;
  if (!shouldSlice) return body;
  const lines = body.split("\n");
  const start = Math.max(0, fromLine - 1);
  const end = maxLines !== undefined ? start + maxLines : lines.length;
  return lines.slice(start, end).join("\n");
}

function truncateBodyByLines(body: string, maxLines: number): string {
  const lines = body.split("\n");
  if (lines.length <= maxLines) return body;
  const truncated = lines.slice(0, maxLines).join("\n");
  return `${truncated}\n\n[... truncated ${lines.length - maxLines} more lines]`;
}

function parseMultiGetPattern(pattern: string): { kind: "list"; items: string[] } | { kind: "glob" } {
  const isList = pattern.includes(",") && !pattern.includes("*") && !pattern.includes("?");
  if (!isList) return { kind: "glob" };
  const items = pattern.split(",").map((item) => item.trim()).filter(Boolean);
  return { kind: "list", items };
}

type MultiGetTarget = {
  filepath: string;
  displayPath: string;
  bodyLength: number;
  collection?: string;
  path?: string;
};

function fallbackTitle(displayPath: string, title?: string): string {
  if (title && title.trim().length > 0) return title;
  return displayPath.split("/").pop() || displayPath;
}

function resolveMultiGetTargets(db: Database, patternText: string): { files: MultiGetTarget[]; errors: string[] } {
  const pattern = parseMultiGetPattern(patternText);
  const errors: string[] = [];

  if (pattern.kind === "list") {
    if (pattern.items.length === 0) {
      throw new Error("No files specified in pattern list. Provide a comma-separated list or a glob pattern.");
    }

    const files: MultiGetTarget[] = [];
    for (const name of pattern.items) {
      let doc: DocMetaRow | null = null;
      if (isVirtualPath(name)) {
        const parsed = parseVirtualPath(name);
        if (!parsed) {
          errors.push(`Invalid virtual path: ${name}`);
          continue;
        }
        doc = getDocMeta(db, parsed.path, parsed.collectionName);
      } else {
        doc = getDocMeta(db, name);
      }

      if (!doc) {
        errors.push(`File not found: ${name}`);
        continue;
      }

      files.push({
        filepath: doc.virtual_path,
        displayPath: doc.virtual_path,
        bodyLength: doc.body_length,
        collection: doc.collection,
        path: doc.path,
      });
    }

    return { files, errors };
  }

  const files = matchFilesByGlob(db, patternText).map((file) => ({
    ...file,
    collection: undefined,
    path: undefined,
  }));

  if (files.length === 0) {
    throw new Error(`No files matched pattern: ${patternText}. Try 'qmd ls <collection>' to browse files.`);
  }

  return { files, errors };
}

function resolveMultiGetLocation(file: MultiGetTarget): { collection: string; path: string } | null {
  if (file.collection && file.path) {
    return { collection: file.collection, path: file.path };
  }
  const parsed = parseVirtualPath(file.filepath);
  if (!parsed) return null;
  return { collection: parsed.collectionName, path: parsed.path };
}

type RankedDoc = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

function toRankedDocs(results: SearchResult[], hashMap: Map<string, string>): RankedDoc[] {
  for (const result of results) {
    hashMap.set(result.filepath, result.hash);
  }
  return results.map((result) => ({
    file: result.filepath,
    displayPath: result.displayPath,
    title: result.title,
    body: result.body || "",
    score: result.score,
  }));
}

function hasStrongBm25Signal(results: SearchResult[]): { strong: boolean; topScore: number; secondScore: number } {
  const topScore = results[0]?.score ?? 0;
  const secondScore = results[1]?.score ?? 0;
  const strong = results.length > 0 && topScore >= BM25_STRONG_SCORE && (topScore - secondScore) >= BM25_STRONG_GAP;
  return { strong, topScore, secondScore };
}

function buildRrfWeights(count: number): number[] {
  return Array.from({ length: count }, (_, i) => (i < 2 ? RRF_PRIMARY_WEIGHT : RRF_SECONDARY_WEIGHT));
}

function getRrfBlendWeight(rank: number): number {
  for (const entry of RRF_WEIGHTS) {
    if (rank <= entry.maxRank) return entry.weight;
  }
  return RRF_WEIGHTS[RRF_WEIGHTS.length - 1].weight;
}

function blendRrfAndRerank(rrfRank: number, rerankScore: number): number {
  const rrfWeight = getRrfBlendWeight(rrfRank);
  const rrfScore = 1 / rrfRank;
  return rrfWeight * rrfScore + (1 - rrfWeight) * rerankScore;
}

type ChunkSelection = { chunks: { text: string; pos: number }[]; bestIdx: number };

function scoreChunkTerms(text: string, terms: string[]): number {
  if (terms.length === 0) return 0;
  const lower = text.toLowerCase();
  return terms.reduce((acc, term) => acc + (lower.includes(term) ? 1 : 0), 0);
}

function pickBestChunkIndex(chunks: { text: string; pos: number }[], queryTerms: string[]): number {
  if (chunks.length === 0) return 0;
  let bestIdx = 0;
  let bestScore = -1;
  for (let i = 0; i < chunks.length; i++) {
    const score = scoreChunkTerms(chunks[i]!.text, queryTerms);
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

type RerankBlend = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  chunkPos: number;
  score: number;
  hash: string;
};

function buildQueryTerms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter((term) => term.length >= MIN_QUERY_TERM_LENGTH);
}

async function buildQueryPlan(
  args: SearchArgs,
  initialFts: SearchResult[],
  logger: LogBuffer
): Promise<{ ftsQueries: string[]; vectorQueries: string[] }> {
  const { strong, topScore } = hasStrongBm25Signal(initialFts);
  const ftsQueries = [args.query];
  const vectorQueries = [args.query];

  if (strong) {
    logger.write(`${logger.c.dim}Strong BM25 signal (${topScore.toFixed(2)}) - skipping expansion${logger.c.reset}\n`);
    const lines = formatQueryExpansionLines(args.query, true, [], logger.c);
    for (const line of lines) {
      logger.write(line + "\n");
    }
    return { ftsQueries, vectorQueries };
  }

  const queryables = await expandQueryStructuredWithLogs(args.query, true, args.context, logger);
  for (const q of queryables) {
    if (!q.text || q.text === args.query) continue;
    if (q.type === "lex") {
      ftsQueries.push(q.text);
    } else if (q.type === "vec" || q.type === "hyde") {
      vectorQueries.push(q.text);
    }
  }

  return { ftsQueries, vectorQueries };
}

async function runHybridSearches(
  db: Database,
  ftsQueries: string[],
  vectorQueries: string[],
  collectionName: string,
  hasVectors: boolean,
  hashMap: Map<string, string>
): Promise<RankedDoc[][]> {
  const ftsLists = ftsQueries
    .map((q) => searchFTS(db, q, QUERY_FTS_LIMIT, collectionName))
    .filter((results) => results.length > 0)
    .map((results) => toRankedDocs(results, hashMap));

  if (!hasVectors) return ftsLists;

  const vecResults = await Promise.all(
    vectorQueries.map((q) => searchVec(db, q, DEFAULT_EMBED_MODEL, QUERY_VEC_LIMIT, collectionName))
  );

  const vecLists = vecResults
    .filter((results) => results.length > 0)
    .map((results) => toRankedDocs(results, hashMap));

  return [...ftsLists, ...vecLists];
}

function fuseCandidates(rankedLists: RankedDoc[][]): RankedDoc[] {
  if (rankedLists.length === 0) return [];
  return reciprocalRankFusion(rankedLists, buildRrfWeights(rankedLists.length)).slice(0, QUERY_RERANK_LIMIT);
}

function selectRerankChunks(
  candidates: RankedDoc[],
  queryTerms: string[]
): { rerankDocs: { file: string; text: string }[]; docChunkMap: Map<string, ChunkSelection> } {
  const rerankDocs: { file: string; text: string }[] = [];
  const docChunkMap = new Map<string, ChunkSelection>();

  for (const candidate of candidates) {
    const chunks = chunkDocument(candidate.body);
    if (chunks.length === 0) continue;
    const bestIdx = pickBestChunkIndex(chunks, queryTerms);
    rerankDocs.push({ file: candidate.file, text: chunks[bestIdx]!.text });
    docChunkMap.set(candidate.file, { chunks, bestIdx });
  }

  return { rerankDocs, docChunkMap };
}

async function rerankCandidates(
  query: string,
  candidates: RankedDoc[],
  hashMap: Map<string, string>,
  logger: LogBuffer
): Promise<RerankBlend[]> {
  if (candidates.length === 0) return [];

  const queryTerms = buildQueryTerms(query);
  const { rerankDocs, docChunkMap } = selectRerankChunks(candidates, queryTerms);
  if (rerankDocs.length === 0) return [];

  const reranked = await rerankWithLogs(query, rerankDocs, logger);
  if (reranked.length === 0) return [];

  const candidateMap = new Map(candidates.map((c) => [c.file, c]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

  return reranked
    .map((r) => {
      const candidate = candidateMap.get(r.file);
      if (!candidate) return null;
      const rrfRank = rrfRankMap.get(r.file) ?? RRF_DEFAULT_RANK;
      const chunkInfo = docChunkMap.get(r.file);
      const chunk = chunkInfo?.chunks[chunkInfo.bestIdx] ?? { text: candidate.body, pos: 0 };
      return {
        file: r.file,
        displayPath: candidate.displayPath,
        title: candidate.title,
        body: chunk.text,
        chunkPos: chunk.pos,
        score: blendRrfAndRerank(rrfRank, r.score),
        hash: hashMap.get(r.file) ?? "",
      };
    })
    .filter((entry): entry is RerankBlend => entry !== null);
}

async function runHybridQueryPipeline(
  args: SearchArgs,
  db: Database,
  collectionName: string | undefined,
  logger: LogBuffer
): Promise<DaemonSearchResult[]> {
  const initialFts = searchFTS(db, args.query, QUERY_FTS_LIMIT, collectionName);
  const hasVectors = hasVectorIndex(db);
  const { ftsQueries, vectorQueries } = await buildQueryPlan(args, initialFts, logger);

  const vectorCount = hasVectors ? vectorQueries.length : 0;
  logger.write(`${logger.c.dim}Searching ${ftsQueries.length} lexical + ${vectorCount} vector queries...${logger.c.reset}\n`);

  const hashMap = new Map<string, string>();
  const rankedLists = await runHybridSearches(db, ftsQueries, vectorQueries, collectionName, hasVectors, hashMap);

  const candidates = fuseCandidates(rankedLists);
  if (candidates.length === 0) return [];

  const reranked = await rerankCandidates(args.query, candidates, hashMap, logger);
  if (reranked.length === 0) return [];

  return reranked
    .sort((a, b) => b.score - a.score)
    .map((result) => ({
      file: result.file,
      displayPath: result.displayPath,
      title: result.title,
      score: result.score,
      hash: result.hash,
      docid: result.hash ? getDocid(result.hash) : undefined,
      chunkPos: result.chunkPos,
      body: result.body,
      context: getContextForFilePath(db, result.file),
    }));
}

// =============================================================================
// Command Handlers
// =============================================================================

/**
 * Handle a search command
 */
async function handleSearch(args: SearchArgs): Promise<DaemonSearchResponse> {
  const limit = args.limit ?? DEFAULT_SEARCH_LIMIT;
  const collectionName = resolveCollectionName(args.collection);
  const db = getDb(args.dbPath);
  const results = searchFTS(db, args.query, limit, collectionName);
  const mapped = results.map(toDaemonSearchResult);
  const constrained = applySearchConstraints(mapped, limit, args.minScore);
  return { results: stripBodyIfNeeded(constrained, shouldIncludeBody(args.full)) };
}

/**
 * Handle a vector search command
 */
async function handleVsearch(args: SearchArgs): Promise<DaemonSearchResponse> {
  const limit = args.limit ?? DEFAULT_SEARCH_LIMIT;
  const logger = createLogBuffer(args.useColor);
  const collectionName = resolveCollectionName(args.collection);

  const db = getDb(args.dbPath);
  if (!hasVectorIndex(db)) {
    throw new Error("Vector index not found. Run 'qmd embed' to create embeddings before using vector search.");
  }

  checkIndexHealthWithLogs(db, logger);
  loadedModels.add(DEFAULT_EMBED_MODEL);

  const queryables = await expandQueryStructuredWithLogs(args.query, false, args.context, logger);
  const vectorQueries = [args.query, ...queryables
    .filter((q) => (q.type === "vec" || q.type === "hyde") && q.text && q.text !== args.query)
    .map((q) => q.text)];

  logger.write(`${logger.c.dim}Searching ${vectorQueries.length} vector queries...${logger.c.reset}\n`);

  const perQueryLimit = args.all ? VSEARCH_ALL_PER_QUERY_LIMIT : VSEARCH_PER_QUERY_LIMIT;
  const bestByFile = new Map<string, SearchResult>();

  for (const q of vectorQueries) {
    const vecResults = await searchVec(db, q, DEFAULT_EMBED_MODEL, perQueryLimit, collectionName);
    for (const result of vecResults) {
      const existing = bestByFile.get(result.filepath);
      if (!existing || result.score > existing.score) {
        bestByFile.set(result.filepath, result);
      }
    }
  }

  const results = Array.from(bestByFile.values())
    .sort((a, b) => b.score - a.score)
    .map(toDaemonSearchResult);
  const constrained = applySearchConstraints(results, limit, args.minScore);

  return {
    results: stripBodyIfNeeded(constrained, shouldIncludeBody(args.full)),
    stderr: logger.stderr,
  };
}

/**
 * Handle a hybrid query command (search + vsearch + rerank)
 */
async function handleQuery(args: SearchArgs): Promise<DaemonSearchResponse> {
  const limit = args.limit ?? DEFAULT_QUERY_LIMIT;
  const logger = createLogBuffer(args.useColor);
  const collectionName = resolveCollectionName(args.collection);

  const db = getDb(args.dbPath);
  loadedModels.add(DEFAULT_EMBED_MODEL);
  loadedModels.add(DEFAULT_RERANK_MODEL);
  loadedModels.add("query-expansion");

  checkIndexHealthWithLogs(db, logger);
  const results = await runHybridQueryPipeline(args, db, collectionName, logger);
  const constrained = applySearchConstraints(results, limit, args.minScore);

  return {
    results: stripBodyIfNeeded(constrained, shouldIncludeBody(args.full)),
    stderr: logger.stderr,
  };
}

/**
 * Handle a get document command
 */
async function handleGet(args: GetArgs): Promise<DaemonGetResponse> {
  const db = getDb(args.dbPath);
  const originalPath = args.path;
  const parsed = parsePathWithLineSuffix(args.path, args.fromLine);
  const inputPath = resolveDocidPath(db, parsed.path, originalPath);

  const resolved = resolveDocFromInputPath(db, inputPath, args.cwd);
  if (!resolved) {
    throw new Error(`Document not found: ${originalPath}. Try a qmd:// path, a docid (prefix #), or 'qmd ls <collection>'.`);
  }

  const { doc, virtualPath } = resolved;
  const context = getContextForPath(db, doc.collectionName, doc.path);
  const startLine = parsed.fromLine ?? 1;
  const body = sliceDocumentBody(doc.body, startLine, args.maxLines);

  return {
    file: virtualPath,
    title: doc.title,
    body,
    context,
    hash: doc.hash,
    docid: getDocid(doc.hash),
    startLine,
  };
}

/**
 * Handle a multi-get command
 */
async function handleMultiGet(args: MultiGetArgs): Promise<DaemonMultiGetResponse> {
  const db = getDb(args.dbPath);
  const maxBytes = args.maxBytes ?? DEFAULT_MULTI_GET_MAX_BYTES;
  const { files, errors } = resolveMultiGetTargets(db, args.pattern);

  const results: DaemonMultiGetResponse["results"] = [];

  for (const file of files) {
    const location = resolveMultiGetLocation(file);
    const context = location ? getContextForPath(db, location.collection, location.path) : null;

    if (file.bodyLength > maxBytes) {
      results.push({
        file: file.filepath,
        displayPath: file.displayPath,
        title: fallbackTitle(file.displayPath),
        body: "",
        context,
        skipped: true,
        skipReason: `File too large (${Math.round(file.bodyLength / 1024)}KB > ${Math.round(maxBytes / 1024)}KB). Use 'qmd get ${file.filepath}' to retrieve.`,
      });
      continue;
    }

    if (!location) continue;
    const doc = getDocBody(db, location.collection, location.path);
    if (!doc) continue;

    const body = args.maxLines !== undefined ? truncateBodyByLines(doc.body, args.maxLines) : doc.body;

    results.push({
      file: file.filepath,
      displayPath: file.displayPath,
      title: fallbackTitle(file.displayPath, doc.title),
      body,
      context,
      skipped: false,
    });
  }

  return { results, errors };
}

/**
 * Handle ls command
 */
async function handleLs(args: LsArgs): Promise<DaemonLsResponse> {
  const db = getDb(args.dbPath);

  if (!args.path) {
    const yamlCollections = yamlListCollections();
    if (yamlCollections.length === 0) {
      return { mode: "collections", collections: [] };
    }

    const collections = yamlCollections.map(coll => {
      const statsRow = db.prepare(`
        SELECT COUNT(*) as file_count
        FROM documents d
        WHERE d.collection = ? AND d.active = 1
      `).get(coll.name);
      const stats = readDbRow(statsRow, isFileCountRow, "collection file count");

      return {
        name: coll.name,
        fileCount: stats?.file_count ?? 0,
      };
    });

    return { mode: "collections", collections };
  }

  const { collectionName, pathPrefix } = parseLsPath(args.path);
  resolveCollectionName(collectionName);

  const baseQuery = `
    SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) as size
    FROM documents d
    JOIN content ct ON d.hash = ct.hash
    WHERE d.collection = ? AND d.active = 1
  `;
  const params: string[] = [collectionName];
  const query = pathPrefix
    ? `${baseQuery} AND d.path LIKE ? ORDER BY d.path`
    : `${baseQuery} ORDER BY d.path`;

  if (pathPrefix) params.push(`${pathPrefix}%`);

  const rows = db.prepare(query).all(...params) as unknown[];
  const files = readDbRows(rows, isLsFileRow, "ls file");

  return {
    mode: "files",
    collectionName,
    pathPrefix,
    files: files.map((file) => ({
      path: file.path,
      title: file.title,
      modifiedAt: file.modified_at,
      size: file.size,
    })),
  };
}

/**
 * Handle status command
 */
async function handleStatus(args: { dbPath?: string }): Promise<DaemonIndexStatus> {
  const db = getDb(args.dbPath);

  const collections = listCollections(db);
  const totalDocsRow = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get();
  const vectorCountRow = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get();
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const mostRecentRow = db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get();
  const totalDocs = readDbRow(totalDocsRow, isCountRow, "document count");
  const vectorCount = readDbRow(vectorCountRow, isCountRow, "vector count");
  const mostRecent = readDbRow(mostRecentRow, isLatestRow, "latest modified");
  const contexts = listAllContexts();

  return {
    collections: collections.map(c => ({
      name: c.name,
      pattern: c.glob_pattern,
      fileCount: c.active_count,
      lastModified: c.last_modified,
    })),
    totalDocs: totalDocs?.count ?? 0,
    vectorCount: vectorCount?.count ?? 0,
    needsEmbedding,
    mostRecent: mostRecent?.latest ?? null,
    contexts,
  };
}

/**
 * Handle daemon-status command (internal)
 */
async function handleDaemonStatus(): Promise<DaemonStatusRunning> {
  return {
    running: true,
    pid: process.pid,
    uptime: Math.floor((Date.now() - startTime) / 1000),
    loadedModels: Array.from(loadedModels),
    activeConnections,
  };
}

/**
 * Dispatch a command to the appropriate handler
 */
async function handleCommand(req: DaemonRequestGeneric): Promise<DaemonResponse> {
  const startedAt = TRACE ? Date.now() : 0;
  try {
    let result: DaemonResult;

    switch (req.cmd) {
      case "search":
        result = await handleSearch(validateSearchArgs(req.args));
        break;
      case "vsearch":
        result = await handleVsearch(validateSearchArgs(req.args));
        break;
      case "query":
        result = await handleQuery(validateSearchArgs(req.args));
        break;
      case "get":
        result = await handleGet(validateGetArgs(req.args));
        break;
      case "multi-get":
        result = await handleMultiGet(validateMultiGetArgs(req.args));
        break;
      case "ls":
        result = await handleLs(validateLsArgs(req.args));
        break;
      case "status":
        result = await handleStatus(validateStatusArgs(req.args));
        break;
      case "daemon-status":
        requireEmptyArgs(req.args, req.cmd);
        result = await handleDaemonStatus();
        break;
      case "ping":
        requireEmptyArgs(req.args, req.cmd);
        result = { pong: true, pid: process.pid } satisfies DaemonPingResponse;
        break;
      default:
        throw new Error(`Unknown command: ${req.cmd}. Expected one of: ${Array.from(DAEMON_COMMANDS).join(", ")}.`);
    }

    if (TRACE) {
      const elapsed = Date.now() - startedAt;
      console.log(`[daemon] ${req.cmd} ok (${elapsed}ms)`);
    }
    return { ok: true, result };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const stack = err instanceof Error ? err.stack : undefined;
    // Log server-side for debugging with request context
    const argsSummary = JSON.stringify(req.args).slice(0, 200);
    console.error(`Command '${req.cmd}' failed: ${message}`);
    console.error(`  args: ${argsSummary}`);
    if (stack) console.error(stack);
    if (TRACE) {
      const elapsed = Date.now() - startedAt;
      console.log(`[daemon] ${req.cmd} error (${elapsed}ms)`);
    }
    return { ok: false, error: message };
  }
}

// =============================================================================
// Socket Server
// =============================================================================

function createLineFramer(maxBytes: number): LineFramer {
  let buffer = "";
  return {
    push(chunk: string) {
      buffer += chunk;
      if (buffer.length > maxBytes) {
        return { lines: [], overflow: true };
      }
      const lines = buffer.split('\n');
      buffer = lines.pop() || "";
      return { lines, overflow: false };
    },
    getBuffer() {
      return buffer;
    },
  };
}

async function handleRequestLine(socket: { write: (text: string) => void }, line: string): Promise<void> {
  if (!line.trim()) return;

  try {
    const parsed = JSON.parse(line) as unknown;
    if (!isDaemonRequestGeneric(parsed)) {
      throw new Error("Invalid request: expected { cmd, args } with a supported command.");
    }
    const res = await handleCommand(parsed);
    socket.write(JSON.stringify(res) + '\n');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const res: DaemonResponse = { ok: false, error: formatParseError("Parse error", message, line) };
    socket.write(JSON.stringify(res) + '\n');
  }
}

function startServer(): void {
  registerSignalHandlers();

  // Ensure cache directory exists
  if (!existsSync(CACHE_DIR)) {
    mkdirSync(CACHE_DIR, { recursive: true });
  }

  // Remove stale socket if it exists
  if (existsSync(SOCKET_PATH)) {
    try {
      unlinkSync(SOCKET_PATH);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`Warning: Could not remove stale socket ${SOCKET_PATH}: ${msg}`);
      console.error("You may need to manually delete it or check file permissions.");
    }
  }

  startTime = Date.now();

  server = Bun.listen<DaemonSocketData>({
    unix: SOCKET_PATH,
    socket: {
      open(socket) {
        socket.data = { framer: createLineFramer(SOCKET_BUFFER_LIMIT) };
        activeConnections++;
      },

      async data(socket, data) {
        const { lines, overflow } = socket.data.framer.push(data.toString());
        if (overflow) {
          const res: DaemonResponse = {
            ok: false,
            error: `Request too large (>${SOCKET_BUFFER_LIMIT} bytes).`,
          };
          socket.write(JSON.stringify(res) + '\n');
          socket.end();
          return;
        }

        for (const line of lines) {
          await handleRequestLine(socket, line);
        }
      },

      close(socket) {
        if (socket.data && !socket.data.cleaned) {
          socket.data.cleaned = true;
          activeConnections--;
        }
      },

      error(socket, error) {
        const bufferInfo = socket.data?.framer.getBuffer()
          ? ` (partial: ${socket.data.framer.getBuffer().slice(0, SOCKET_ERROR_PARTIAL_LIMIT)}...)`
          : '';
        console.error(`Socket error${bufferInfo}:`, error);
        if (socket.data && !socket.data.cleaned) {
          socket.data.cleaned = true;
          activeConnections--;
        }
      },
    },
  });

  // Write PID file
  writeFileSync(PID_PATH, String(process.pid));

  console.log(`QMD daemon started (PID ${process.pid})`);
  console.log(`Socket: ${SOCKET_PATH}`);
}

async function stopServer(): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;

  console.log("\nShutting down...");
  let cleanupErrors = 0;

  // Close server
  if (server) {
    server.stop();
    server = null;
  }

  // Clean up socket file
  if (existsSync(SOCKET_PATH)) {
    try {
      unlinkSync(SOCKET_PATH);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`Warning: Could not clean up socket file: ${msg}`);
      cleanupErrors++;
    }
  }

  // Clean up PID file
  if (existsSync(PID_PATH)) {
    try {
      unlinkSync(PID_PATH);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`Warning: Could not clean up PID file: ${msg}`);
      cleanupErrors++;
    }
  }

  // Close databases
  for (const entry of stores.values()) {
    entry.close();
  }
  stores.clear();

  // Dispose LLM resources
  try {
    await disposeDefaultLlamaCpp();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Warning: Error disposing LLM resources: ${msg}`);
    cleanupErrors++;
  }

  if (cleanupErrors > 0) {
    console.log(`Daemon stopped with ${cleanupErrors} cleanup warning(s).`);
    process.exit(1);
  }
  console.log("Daemon stopped.");
  process.exit(0);
}

/**
 * Start the daemon in the foreground (blocking).
 */
export function runDaemonForeground(): void {
  startServer();
}

// =============================================================================
// Client Functions (for use by CLI)
// =============================================================================

/**
 * Send a single request to the daemon over the Unix socket.
 * Throws on connection errors, timeouts, or invalid JSON responses.
 */
export async function sendToDaemon(req: DaemonRequest, timeoutMs = 30000): Promise<DaemonResponse> {
  if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error(`Invalid timeoutMs: expected a positive number of milliseconds (received ${describeValue(timeoutMs)}).`);
  }
  return new Promise((resolve, reject) => {
    let buffer = "";
    let resolved = false;
    let client: { end: () => void } | null = null;

    const closeClient = () => {
      if (!client) return;
      try {
        client.end();
      } catch {
        // best-effort close
      }
      client = null;
    };

    const finalize = (closeSocket: boolean) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      if (closeSocket) closeClient();
    };

    const timeout = setTimeout(() => {
      if (!resolved) {
        finalize(true);
        reject(new Error(`Daemon request timed out after ${timeoutMs}ms for '${req.cmd}'.`));
      }
    }, timeoutMs);

    Bun.connect({
      unix: SOCKET_PATH,
      socket: {
        open(socket) {
          client = socket;
          socket.write(JSON.stringify(req) + '\n');
        },
        data(socket, data) {
          if (resolved) return;

          // Buffer data until we get a complete line
          buffer += data.toString();
          const newlineIdx = buffer.indexOf('\n');
          if (newlineIdx === -1) return; // Wait for more data

          const line = buffer.slice(0, newlineIdx);
          try {
            const res = JSON.parse(line) as DaemonResponse;
            finalize(true);
            resolve(res);
          } catch (err) {
            finalize(true);
            const message = err instanceof Error ? err.message : String(err);
            const wrapped = createErrorWithCause(
              formatParseError(`Failed to parse daemon response for '${req.cmd}'`, message, line),
              err
            );
            reject(wrapped);
          }
        },
        error(socket, error) {
          if (!resolved) {
            finalize(false);
            reject(createErrorWithCause(`Daemon socket error during '${req.cmd}': ${error.message || error}`, error));
          }
        },
        close() {
          if (!resolved) {
            finalize(false);
            reject(new Error(`Connection closed by daemon before response for '${req.cmd}'.`));
          }
        },
        connectError(socket, error) {
          if (!resolved) {
            finalize(false);
            reject(createErrorWithCause(
              `Failed to connect to daemon for '${req.cmd}': ${error.message || error}. Is the daemon running?`,
              error
            ));
          }
        },
      },
    });
  });
}

/**
 * Remove stale socket and PID files if they exist.
 */
export function cleanupStaleFiles(): void {
  try {
    if (existsSync(SOCKET_PATH)) unlinkSync(SOCKET_PATH);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Warning: Could not clean up socket file: ${msg}`);
  }
  try {
    if (existsSync(PID_PATH)) unlinkSync(PID_PATH);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Warning: Could not clean up PID file: ${msg}`);
  }
}

/**
 * Returns true if the daemon appears to be running and its PID is alive.
 */
export function isDaemonRunning(): boolean {
  if (!existsSync(SOCKET_PATH)) return false;
  if (!existsSync(PID_PATH)) return false;

  // Check if PID is still alive
  try {
    const pidContent = readFileSync(PID_PATH, 'utf-8').trim();
    const pid = parseInt(pidContent, 10);
    if (isNaN(pid)) {
      // Invalid PID file content
      cleanupStaleFiles();
      return false;
    }
    process.kill(pid, 0); // Doesn't actually kill, just checks
    return true;
  } catch (err) {
    // ESRCH means process doesn't exist - clean up stale files
    if (err instanceof Error && 'code' in err && err.code === 'ESRCH') {
      cleanupStaleFiles();
    }
    // EPERM means process exists but we can't signal it - still running
    if (err instanceof Error && 'code' in err && err.code === 'EPERM') {
      return true;
    }
    return false;
  }
}

/**
 * Fetch daemon status over the socket. Returns { running: false } on failure.
 */
export async function getDaemonStatus(): Promise<DaemonStatus> {
  if (!isDaemonRunning()) {
    return { running: false };
  }

  try {
    const res = await sendToDaemon({ cmd: "daemon-status", args: {} }, 5000);
    if (res.ok) {
      const result = res.result;
      if (result && typeof result === "object" && "running" in result) {
        return result as DaemonStatus;
      }
      console.error("Warning: Daemon status response missing expected fields.");
      return { running: false };
    } else {
      // Daemon responded with error - treat as not running since we can't get full status
      console.error(`Warning: Daemon status error: ${res.error}`);
      return { running: false };
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Warning: Could not get daemon status: ${msg}`);
    return { running: false };
  }
}

/**
 * Stop the daemon by sending SIGTERM to the PID file process.
 */
export function stopDaemon(): { stopped: boolean; reason?: string } {
  if (!existsSync(PID_PATH)) {
    return { stopped: false, reason: "No PID file found (daemon not running?)" };
  }

  try {
    const pidContent = readFileSync(PID_PATH, 'utf-8').trim();
    const pid = parseInt(pidContent, 10);
    if (isNaN(pid)) {
      cleanupStaleFiles();
      return { stopped: false, reason: "Invalid PID file content (stale files removed)" };
    }
    process.kill(pid, 'SIGTERM');
    return { stopped: true };
  } catch (err) {
    if (err instanceof Error && 'code' in err) {
      if (err.code === 'ESRCH') {
        // Process already gone
        cleanupStaleFiles();
        return { stopped: true };
      }
      if (err.code === 'EPERM') {
        return { stopped: false, reason: "Permission denied to stop daemon" };
      }
    }
    const msg = err instanceof Error ? err.message : String(err);
    return { stopped: false, reason: msg };
  }
}

// =============================================================================
// Main Entry Point
// =============================================================================

if (import.meta.main) {
  const arg = process.argv[2];

  if (arg === "run" || arg === undefined) {
    // Run in foreground
    runDaemonForeground();
  } else {
    console.error("Usage: bun src/daemon.ts [run]");
    console.error("  run: Start daemon in foreground");
    process.exit(1);
  }
}
