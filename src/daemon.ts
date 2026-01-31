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

type DaemonSocketData = {
  buffer: string;
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
  const error = new Error(message);
  (error as Error & { cause?: unknown }).cause = cause;
  return error;
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
  return db.prepare(DOC_SELECT_SQL).get(collectionName, path) as DocRow | null;
}

function getDocMetaByCollectionPath(db: Database, collectionName: string, path: string): DocMetaRow | null {
  return db.prepare(DOC_META_BY_COLLECTION_SQL).get(collectionName, path) as DocMetaRow | null;
}

function getDocMetaByPath(db: Database, path: string): DocMetaRow | null {
  return db.prepare(DOC_META_BY_PATH_SQL).get(path) as DocMetaRow | null;
}

function getDocBody(db: Database, collectionName: string, path: string): DocBodyRow | null {
  return db.prepare(DOC_BODY_SQL).get(collectionName, path) as DocBodyRow | null;
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

function resolveDocFromInputPath(
  db: Database,
  inputPath: string,
  cwd?: string
): { doc: DocRow; virtualPath: string } | null {
  if (isVirtualPath(inputPath)) {
    const parsed = parseVirtualPath(inputPath);
    if (!parsed) {
      throw new Error(`Invalid virtual path: ${inputPath}. Expected format: qmd://<collection>/<path>.`);
    }
    const doc = getDocByCollectionPath(db, parsed.collectionName, parsed.path);
    if (!doc) return null;
    return { doc, virtualPath: buildVirtualPath(doc.collectionName, doc.path) };
  }

  const collectionPath = parseCollectionPath(inputPath);
  if (collectionPath) {
    const doc = getDocByCollectionPath(db, collectionPath.collectionName, collectionPath.path);
    if (!doc) return null;
    return { doc, virtualPath: buildVirtualPath(doc.collectionName, doc.path) };
  }

  let fsPath = inputPath;
  if (fsPath.startsWith("~/")) {
    fsPath = homedir() + fsPath.slice(1);
  } else if (!fsPath.startsWith("/")) {
    fsPath = resolve(cwd || getPwd(), fsPath);
  }
  fsPath = getRealPath(fsPath);

  const detected = detectCollectionFromPath(fsPath);
  if (!detected) return null;
  const doc = getDocByCollectionPath(db, detected.collectionName, detected.relativePath);
  if (!doc) return null;
  return { doc, virtualPath: buildVirtualPath(doc.collectionName, doc.path) };
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

function logQueryExpansionLines(logger: LogBuffer, query: string, includeLexical: boolean, queryables: Queryable[]): void {
  const lines = formatQueryExpansionLines(query, includeLexical, queryables, logger.c);
  for (const line of lines) {
    logger.write(line + "\n");
  }
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
  return { results: results.map(toDaemonSearchResult) };
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
    .slice(0, limit)
    .map(toDaemonSearchResult);

  return { results, stderr: logger.stderr };
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

  const initialFts = searchFTS(db, args.query, QUERY_FTS_LIMIT, collectionName);
  const hasVectors = hasVectorIndex(db);
  const { strong, topScore } = hasStrongBm25Signal(initialFts);

  const ftsQueries = [args.query];
  const vectorQueries = [args.query];

  if (strong) {
    logger.write(`${logger.c.dim}Strong BM25 signal (${topScore.toFixed(2)}) - skipping expansion${logger.c.reset}\n`);
    logQueryExpansionLines(logger, args.query, true, []);
  } else {
    const queryables = await expandQueryStructuredWithLogs(args.query, true, args.context, logger);
    for (const q of queryables) {
      if (!q.text || q.text === args.query) continue;
      if (q.type === "lex") {
        ftsQueries.push(q.text);
      } else if (q.type === "vec" || q.type === "hyde") {
        vectorQueries.push(q.text);
      }
    }
  }

  logger.write(`${logger.c.dim}Searching ${ftsQueries.length} lexical + ${vectorQueries.length} vector queries...${logger.c.reset}\n`);

  const rankedLists: RankedDoc[][] = [];
  const hashMap = new Map<string, string>();
  const searches: Promise<void>[] = [];

  for (const q of ftsQueries) {
    searches.push((async () => {
      const ftsResults = searchFTS(db, q, QUERY_FTS_LIMIT, collectionName);
      if (ftsResults.length > 0) {
        rankedLists.push(toRankedDocs(ftsResults, hashMap));
      }
    })());
  }

  if (hasVectors) {
    for (const q of vectorQueries) {
      searches.push((async () => {
        const vecResults = await searchVec(db, q, DEFAULT_EMBED_MODEL, QUERY_VEC_LIMIT, collectionName);
        if (vecResults.length > 0) {
          rankedLists.push(toRankedDocs(vecResults, hashMap));
        }
      })());
    }
  }

  await Promise.all(searches);

  if (rankedLists.length === 0) {
    return { results: [], stderr: logger.stderr };
  }

  const fused = reciprocalRankFusion(rankedLists, buildRrfWeights(rankedLists.length));
  const candidates = fused.slice(0, QUERY_RERANK_LIMIT);
  if (candidates.length === 0) {
    return { results: [], stderr: logger.stderr };
  }

  const queryTerms = args.query
    .toLowerCase()
    .split(/\s+/)
    .filter((term) => term.length >= MIN_QUERY_TERM_LENGTH);

  const chunksToRerank: { file: string; text: string }[] = [];
  const docChunkMap = new Map<string, ChunkSelection>();

  for (const candidate of candidates) {
    const chunks = chunkDocument(candidate.body);
    if (chunks.length === 0) continue;
    const bestIdx = pickBestChunkIndex(chunks, queryTerms);
    chunksToRerank.push({ file: candidate.file, text: chunks[bestIdx]!.text });
    docChunkMap.set(candidate.file, { chunks, bestIdx });
  }

  const reranked = await rerankWithLogs(
    args.query,
    chunksToRerank.map((c) => ({ file: c.file, text: c.text })),
    logger
  );

  if (reranked.length === 0) {
    return { results: [], stderr: logger.stderr };
  }

  const candidateMap = new Map(candidates.map((c) => [c.file, c]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

  const scored = new Map<string, number>();
  for (const r of reranked) {
    scored.set(r.file, r.score);
  }

  const blendedResults = Array.from(scored.entries())
    .map(([file, rerankScore]) => {
      const candidate = candidateMap.get(file);
      const rrfRank = rrfRankMap.get(file) ?? RRF_DEFAULT_RANK;
      const chunkInfo = docChunkMap.get(file);
      const chunk = chunkInfo?.chunks[chunkInfo.bestIdx] ?? { text: candidate?.body ?? "", pos: 0 };
      return {
        file,
        displayPath: candidate?.displayPath ?? "",
        title: candidate?.title ?? "",
        body: chunk.text,
        chunkPos: chunk.pos,
        score: blendRrfAndRerank(rrfRank, rerankScore),
        hash: hashMap.get(file) ?? "",
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  const results: DaemonSearchResult[] = blendedResults.map((result) => ({
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

  return { results, stderr: logger.stderr };
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
  const pattern = parseMultiGetPattern(args.pattern);

  let files: { filepath: string; displayPath: string; bodyLength: number; collection?: string; path?: string }[] = [];
  const errors: string[] = [];

  if (pattern.kind === "list") {
    if (pattern.items.length === 0) {
      throw new Error("No files specified in pattern list. Provide a comma-separated list or a glob pattern.");
    }

    for (const name of pattern.items) {
      let doc: DocMetaRow | null = null;
      if (isVirtualPath(name)) {
        const parsed = parseVirtualPath(name);
        if (!parsed) {
          errors.push(`Invalid virtual path: ${name}`);
          continue;
        }
        doc = getDocMetaByCollectionPath(db, parsed.collectionName, parsed.path);
      } else {
        doc = getDocMetaByPath(db, name);
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
  } else {
    files = matchFilesByGlob(db, args.pattern).map((file) => ({
      ...file,
      collection: undefined,
      path: undefined,
    }));
    if (files.length === 0) {
      throw new Error(`No files matched pattern: ${args.pattern}. Try 'qmd ls <collection>' to browse files.`);
    }
  }

  const results: DaemonMultiGetResponse["results"] = [];

  for (const file of files) {
    let collection = file.collection;
    let path = file.path;

    if (!collection || !path) {
      const parsed = parseVirtualPath(file.filepath);
      if (parsed) {
        collection = parsed.collectionName;
        path = parsed.path;
      }
    }

    const context = collection && path ? getContextForPath(db, collection, path) : null;

    if (file.bodyLength > maxBytes) {
      results.push({
        file: file.filepath,
        displayPath: file.displayPath,
        title: file.displayPath.split("/").pop() || file.displayPath,
        body: "",
        context,
        skipped: true,
        skipReason: `File too large (${Math.round(file.bodyLength / 1024)}KB > ${Math.round(maxBytes / 1024)}KB). Use 'qmd get ${file.displayPath}' to retrieve.`,
      });
      continue;
    }

    if (!collection || !path) continue;
    const doc = getDocBody(db, collection, path);
    if (!doc) continue;

    const body = args.maxLines !== undefined ? truncateBodyByLines(doc.body, args.maxLines) : doc.body;

    results.push({
      file: file.filepath,
      displayPath: file.displayPath,
      title: doc.title || file.displayPath.split("/").pop() || file.displayPath,
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
      const stats = db.prepare(`
        SELECT COUNT(*) as file_count
        FROM documents d
        WHERE d.collection = ? AND d.active = 1
      `).get(coll.name) as { file_count: number } | null;

      return {
        name: coll.name,
        fileCount: stats?.file_count || 0,
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

  const files = db.prepare(query).all(...params) as {
    path: string;
    title: string;
    modified_at: string;
    size: number;
  }[];

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
  const totalDocs = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };
  const vectorCount = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const mostRecent = db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get() as { latest: string | null };
  const contexts = listAllContexts();

  return {
    collections: collections.map(c => ({
      name: c.name,
      pattern: c.glob_pattern,
      fileCount: c.active_count,
      lastModified: c.last_modified,
    })),
    totalDocs: totalDocs.count,
    vectorCount: vectorCount.count,
    needsEmbedding,
    mostRecent: mostRecent.latest,
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
        result = await handleDaemonStatus();
        break;
      case "ping":
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
        socket.data = { buffer: "" };
        activeConnections++;
      },

      async data(socket, data) {
        // Accumulate data in buffer
        socket.data.buffer += data.toString();
        if (socket.data.buffer.length > SOCKET_BUFFER_LIMIT) {
          const res: DaemonResponse = {
            ok: false,
            error: `Request too large (>${SOCKET_BUFFER_LIMIT} bytes).`,
          };
          socket.write(JSON.stringify(res) + '\n');
          socket.end();
          return;
        }

        // Process complete lines (NDJSON)
        const lines = socket.data.buffer.split('\n');
        socket.data.buffer = lines.pop() || ""; // Keep incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;

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
      },

      close(socket) {
        if (socket.data && !socket.data.cleaned) {
          socket.data.cleaned = true;
          activeConnections--;
        }
      },

      error(socket, error) {
        const bufferInfo = socket.data?.buffer
          ? ` (partial: ${socket.data.buffer.slice(0, SOCKET_ERROR_PARTIAL_LIMIT)}...)`
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

    const cleanup = () => {
      resolved = true;
      clearTimeout(timeout);
    };

    const timeout = setTimeout(() => {
      if (!resolved) {
        cleanup();
        reject(new Error(`Daemon request timed out after ${timeoutMs}ms for '${req.cmd}'.`));
      }
    }, timeoutMs);

    Bun.connect({
      unix: SOCKET_PATH,
      socket: {
        open(socket) {
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
            cleanup();
            socket.end();
            resolve(res);
          } catch (err) {
            cleanup();
            socket.end();
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
            cleanup();
            reject(createErrorWithCause(`Daemon socket error during '${req.cmd}': ${error.message || error}`, error));
          }
        },
        close() {
          if (!resolved) {
            cleanup();
            reject(new Error(`Connection closed by daemon before response for '${req.cmd}'.`));
          }
        },
        connectError(socket, error) {
          if (!resolved) {
            cleanup();
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
