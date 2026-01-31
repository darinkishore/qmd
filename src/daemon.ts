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
  type DaemonRequest,
  type DaemonResponse,
  type DaemonStatus,
  type DaemonStatusRunning,
  type DaemonRequestGeneric,
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
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp, type Queryable, type RerankDocument } from "./llm.js";

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

function getDb(dbPath?: string) {
  return getStore(dbPath).db;
}

// =============================================================================
// Command Handlers
// =============================================================================

type DaemonSearchResult = {
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

type DaemonSearchResponse = {
  results: DaemonSearchResult[];
  stderr?: string[];
};

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

function createLogBuffer(useColor: boolean | undefined): LogBuffer {
  const enableColor = !!useColor;
  const c = {
    reset: enableColor ? "\x1b[0m" : "",
    dim: enableColor ? "\x1b[2m" : "",
    bold: enableColor ? "\x1b[1m" : "",
    cyan: enableColor ? "\x1b[36m" : "",
    yellow: enableColor ? "\x1b[33m" : "",
    green: enableColor ? "\x1b[32m" : "",
    magenta: enableColor ? "\x1b[35m" : "",
    blue: enableColor ? "\x1b[34m" : "",
  };
  const stderr: string[] = [];
  const write = (text: string) => {
    stderr.push(text);
  };
  const progress = {
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

async function expandQueryStructuredWithLogs(
  query: string,
  includeLexical: boolean,
  context: string | undefined,
  logger: LogBuffer
): Promise<Queryable[]> {
  logger.write(`${logger.c.dim}Expanding query...${logger.c.reset}\n`);

  const llm = getDefaultLlamaCpp();
  const queryables = await llm.expandQuery(query, { includeLexical, context });

  const lines: string[] = [];
  const bothLabel = includeLexical ? " · (lexical+vector)" : " · (vector)";
  lines.push(`${logger.c.dim}├─ ${query}${bothLabel}${logger.c.reset}`);

  for (let i = 0; i < queryables.length; i++) {
    const q = queryables[i];
    if (!q || q.text === query) continue;

    let textPreview = q.text.replace(/\n/g, " ");
    if (textPreview.length > 80) {
      textPreview = textPreview.substring(0, 77) + "...";
    }

    const label = q.type === "lex" ? "lexical" : (q.type === "hyde" ? "hyde" : "vector");
    lines.push(`${logger.c.dim}├─ ${textPreview} · (${label})${logger.c.reset}`);
  }

  if (lines.length > 0) {
    lines[lines.length - 1] = lines[lines.length - 1]!.replace("├─", "└─");
  }

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
    text: doc.text.slice(0, 4000),
  }));

  const result = await llm.rerank(query, rerankDocs, { model: DEFAULT_RERANK_MODEL });

  logger.progress.clear();
  logger.write("\n");

  return result.results.map((r) => ({ file: r.file, score: r.score }));
}

// Enable production mode - allows using default database path
enableProductionMode();
import {
  getCollection as getCollectionFromYaml,
  listCollections as yamlListCollections,
  listAllContexts,
} from "./collections.js";
// =============================================================================
// Input Validation Helpers
// =============================================================================

export function validateSearchArgs(args: unknown): SearchArgs {
  const a = args as Record<string, unknown>;
  if (typeof a.query !== 'string' || !a.query.trim()) {
    throw new Error("Missing required argument: query");
  }
  return {
    query: a.query,
    limit: typeof a.limit === 'number' ? a.limit : undefined,
    minScore: typeof a.minScore === 'number' ? a.minScore : undefined,
    all: typeof a.all === 'boolean' ? a.all : undefined,
    collection: typeof a.collection === 'string' ? a.collection : undefined,
    full: typeof a.full === 'boolean' ? a.full : undefined,
    dbPath: typeof a.dbPath === 'string' ? a.dbPath : undefined,
    useColor: typeof a.useColor === 'boolean' ? a.useColor : undefined,
    context: typeof a.context === 'string' ? a.context : undefined,
  };
}

export function validateGetArgs(args: unknown): GetArgs {
  const a = args as Record<string, unknown>;
  if (typeof a.path !== 'string' || !a.path.trim()) {
    throw new Error("Missing required argument: path");
  }
  return {
    path: a.path,
    fromLine: typeof a.fromLine === 'number' ? a.fromLine : undefined,
    maxLines: typeof a.maxLines === 'number' ? a.maxLines : undefined,
    dbPath: typeof a.dbPath === 'string' ? a.dbPath : undefined,
    cwd: typeof a.cwd === 'string' ? a.cwd : undefined,
  };
}

export function validateMultiGetArgs(args: unknown): MultiGetArgs {
  const a = args as Record<string, unknown>;
  if (typeof a.pattern !== 'string' || !a.pattern.trim()) {
    throw new Error("Missing required argument: pattern");
  }
  return {
    pattern: a.pattern,
    maxLines: typeof a.maxLines === 'number' ? a.maxLines : undefined,
    maxBytes: typeof a.maxBytes === 'number' ? a.maxBytes : undefined,
    dbPath: typeof a.dbPath === 'string' ? a.dbPath : undefined,
  };
}

export function validateLsArgs(args: unknown): LsArgs {
  const a = args as Record<string, unknown>;
  return {
    path: typeof a.path === 'string' ? a.path : undefined,
    dbPath: typeof a.dbPath === 'string' ? a.dbPath : undefined,
  };
}

export function validateStatusArgs(args: unknown): { dbPath?: string } {
  const a = args as Record<string, unknown>;
  return {
    dbPath: typeof a.dbPath === "string" ? a.dbPath : undefined,
  };
}

// =============================================================================
// Path Resolution Helpers
// =============================================================================

function detectCollectionFromPath(db: Database, fsPath: string): { collectionName: string; relativePath: string } | null {
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

// =============================================================================
// Command Handlers
// =============================================================================

/**
 * Handle a search command
 */
async function handleSearch(args: SearchArgs): Promise<unknown> {
  const limit = args.limit ?? 20;

  let collectionName: string | undefined;
  if (args.collection) {
    const coll = getCollectionFromYaml(args.collection);
    if (!coll) {
      throw new Error(`Collection not found: ${args.collection}`);
    }
    collectionName = args.collection;
  }

  const db = getDb(args.dbPath);
  const results = searchFTS(db, args.query, limit, collectionName as any);
  const mapped: DaemonSearchResult[] = results.map(r => ({
    file: r.filepath,
    displayPath: r.displayPath,
    title: r.title,
    score: r.score,
    hash: r.hash,
    docid: r.docid,
    body: r.body,
    context: r.context,
  }));

  return { results: mapped } satisfies DaemonSearchResponse;
}

/**
 * Handle a vector search command
 */
async function handleVsearch(args: SearchArgs): Promise<unknown> {
  const limit = args.limit ?? 20;
  const useColor = args.useColor;
  const logger = createLogBuffer(useColor);

  let collectionName: string | undefined;
  if (args.collection) {
    const coll = getCollectionFromYaml(args.collection);
    if (!coll) {
      throw new Error(`Collection not found: ${args.collection}`);
    }
    collectionName = args.collection;
  }

  const db = getDb(args.dbPath);
  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) {
    throw new Error("Vector index not found. Run 'qmd embed' first to create embeddings.");
  }

  checkIndexHealthWithLogs(db, logger);

  loadedModels.add(DEFAULT_EMBED_MODEL);

  const queryables = await expandQueryStructuredWithLogs(args.query, false, args.context, logger);

  const vectorQueries: string[] = [args.query];
  for (const q of queryables) {
    if (q.type === "vec" || q.type === "hyde") {
      if (q.text && q.text !== args.query) {
        vectorQueries.push(q.text);
      }
    }
  }

  logger.write(`${logger.c.dim}Searching ${vectorQueries.length} vector queries...${logger.c.reset}\n`);

  const perQueryLimit = args.all ? 500 : 20;
  const allResults = new Map<string, { file: string; displayPath: string; title: string; score: number; hash: string; docid: string; chunkPos?: number; body: string; context: string | null }>();

  for (const q of vectorQueries) {
    const vecResults = await searchVec(db, q, DEFAULT_EMBED_MODEL, perQueryLimit, collectionName as any);
    for (const r of vecResults) {
      const existing = allResults.get(r.filepath);
      if (!existing || r.score > existing.score) {
        allResults.set(r.filepath, {
          file: r.filepath,
          displayPath: r.displayPath,
          title: r.title,
          score: r.score,
          hash: r.hash,
          docid: r.docid,
          chunkPos: r.chunkPos,
          body: r.body || "",
          context: r.context ?? null,
        });
      }
    }
  }

  const results = Array.from(allResults.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(r => ({
      file: r.file,
      displayPath: r.displayPath,
      title: r.title,
      score: r.score,
      hash: r.hash,
      docid: r.docid,
      chunkPos: r.chunkPos,
      body: r.body,
      context: r.context,
    }));

  return { results, stderr: logger.stderr } satisfies DaemonSearchResponse;
}

/**
 * Handle a hybrid query command (search + vsearch + rerank)
 */
async function handleQuery(args: SearchArgs): Promise<unknown> {
  const limit = args.limit ?? 5;
  const useColor = args.useColor;
  const logger = createLogBuffer(useColor);

  let collectionName: string | undefined;
  if (args.collection) {
    const coll = getCollectionFromYaml(args.collection);
    if (!coll) {
      throw new Error(`Collection not found: ${args.collection}`);
    }
    collectionName = args.collection;
  }

  const db = getDb(args.dbPath);
  loadedModels.add(DEFAULT_EMBED_MODEL);
  loadedModels.add(DEFAULT_RERANK_MODEL);
  loadedModels.add("query-expansion");

  checkIndexHealthWithLogs(db, logger);

  const initialFts = searchFTS(db, args.query, 20, collectionName as any);
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  const topScore = initialFts[0]?.score ?? 0;
  const secondScore = initialFts[1]?.score ?? 0;
  const hasStrongSignal = initialFts.length > 0 && topScore >= 0.85 && (topScore - secondScore) >= 0.15;

  let ftsQueries: string[] = [args.query];
  let vectorQueries: string[] = [args.query];

  if (hasStrongSignal) {
    logger.write(`${logger.c.dim}Strong BM25 signal (${topScore.toFixed(2)}) - skipping expansion${logger.c.reset}\n`);
    const lines: string[] = [];
    lines.push(`${logger.c.dim}├─ ${args.query} · (lexical+vector)${logger.c.reset}`);
    lines[lines.length - 1] = lines[lines.length - 1]!.replace("├─", "└─");
    for (const line of lines) logger.write(line + "\n");
  } else {
    const queryables = await expandQueryStructuredWithLogs(args.query, true, args.context, logger);
    for (const q of queryables) {
      if (q.type === "lex") {
        if (q.text && q.text !== args.query) ftsQueries.push(q.text);
      } else if (q.type === "vec" || q.type === "hyde") {
        if (q.text && q.text !== args.query) vectorQueries.push(q.text);
      }
    }
  }

  logger.write(`${logger.c.dim}Searching ${ftsQueries.length} lexical + ${vectorQueries.length} vector queries...${logger.c.reset}\n`);

  const rankedLists: { file: string; displayPath: string; title: string; body: string; score: number }[][] = [];
  const hashMap = new Map<string, string>();
  const searchPromises: Promise<void>[] = [];

  for (const q of ftsQueries) {
    if (!q) continue;
    searchPromises.push((async () => {
      const ftsResults = searchFTS(db, q, 20, (collectionName || "") as any);
      if (ftsResults.length > 0) {
        for (const r of ftsResults) {
          hashMap.set(r.filepath, r.hash);
        }
        rankedLists.push(ftsResults.map(r => ({
          file: r.filepath,
          displayPath: r.displayPath,
          title: r.title,
          body: r.body || "",
          score: r.score,
        })));
      }
    })());
  }

  if (hasVectors) {
    for (const q of vectorQueries) {
      if (!q) continue;
      searchPromises.push((async () => {
        const vecResults = await searchVec(db, q, DEFAULT_EMBED_MODEL, 20, (collectionName || "") as any);
        if (vecResults.length > 0) {
          for (const r of vecResults) hashMap.set(r.filepath, r.hash);
          rankedLists.push(vecResults.map(r => ({
            file: r.filepath,
            displayPath: r.displayPath,
            title: r.title,
            body: r.body || "",
            score: r.score,
          })));
        }
      })());
    }
  }

  await Promise.all(searchPromises);

  const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
  const fused = reciprocalRankFusion(rankedLists, weights);
  const RERANK_DOC_LIMIT = 40;
  const candidates = fused.slice(0, RERANK_DOC_LIMIT);

  if (candidates.length === 0) {
    return { results: [], stderr: logger.stderr } satisfies DaemonSearchResponse;
  }

  const chunksToRerank: { file: string; text: string; chunkIdx: number }[] = [];
  const docChunkMap = new Map<string, { chunks: { text: string; pos: number }[]; bestIdx: number }>();

  const queryTerms = args.query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  for (const c of candidates) {
    const chunks = chunkDocument(c.body);
    if (chunks.length === 0) continue;

    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < chunks.length; i++) {
      const chunkLower = chunks[i]!.text.toLowerCase();
      const score = queryTerms.reduce((acc, term) => acc + (chunkLower.includes(term) ? 1 : 0), 0);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    chunksToRerank.push({ file: c.file, text: chunks[bestIdx]!.text, chunkIdx: bestIdx });
    docChunkMap.set(c.file, { chunks, bestIdx });
  }

  const reranked = await rerankWithLogs(
    args.query,
    chunksToRerank.map(c => ({ file: c.file, text: c.text })),
    logger
  );

  const aggregatedScores = new Map<string, { score: number; bestChunkIdx: number }>();
  for (const r of reranked) {
    const chunkInfo = docChunkMap.get(r.file);
    aggregatedScores.set(r.file, { score: r.score, bestChunkIdx: chunkInfo?.bestIdx ?? 0 });
  }

  const candidateMap = new Map(candidates.map(c => [c.file, { displayPath: c.displayPath, title: c.title, body: c.body }]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

  const finalResults = Array.from(aggregatedScores.entries()).map(([file, { score: rerankScore, bestChunkIdx }]) => {
    const rrfRank = rrfRankMap.get(file) || 30;
    let rrfWeight: number;
    if (rrfRank <= 3) {
      rrfWeight = 0.75;
    } else if (rrfRank <= 10) {
      rrfWeight = 0.60;
    } else {
      rrfWeight = 0.40;
    }
    const rrfScore = 1 / rrfRank;
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * rerankScore;
    const candidate = candidateMap.get(file);
    const chunkInfo = docChunkMap.get(file);
    const chunkBody = chunkInfo ? (chunkInfo.chunks[bestChunkIdx]?.text || chunkInfo.chunks[0]!.text) : candidate?.body || "";
    const chunkPos = chunkInfo ? (chunkInfo.chunks[bestChunkIdx]?.pos || 0) : 0;
    return {
      file,
      displayPath: candidate?.displayPath || "",
      title: candidate?.title || "",
      body: chunkBody,
      chunkPos,
      score: blendedScore,
      hash: hashMap.get(file) || "",
    };
  }).sort((a, b) => b.score - a.score);

  const seenFiles = new Set<string>();
  const deduped = finalResults.filter(r => {
    if (seenFiles.has(r.file)) return false;
    seenFiles.add(r.file);
    return true;
  });

  const mapped: DaemonSearchResult[] = deduped.map(r => ({
    file: r.file,
    displayPath: r.displayPath,
    title: r.title,
    score: r.score,
    hash: r.hash,
    docid: r.hash ? getDocid(r.hash) : undefined,
    chunkPos: r.chunkPos,
    body: r.body,
    context: (() => {
      const parsed = parseVirtualPath(r.file);
      if (!parsed) return null;
      return getContextForPath(db, parsed.collectionName, parsed.path);
    })(),
  }));

  return { results: mapped, stderr: logger.stderr } satisfies DaemonSearchResponse;
}

/**
 * Handle a get document command
 */
async function handleGet(args: GetArgs): Promise<unknown> {
  const db = getDb(args.dbPath);
  const originalPath = args.path;
  let inputPath = args.path;
  let fromLine = args.fromLine;

  const colonMatch = inputPath.match(/:(\d+)$/);
  if (colonMatch && fromLine === undefined) {
    const matched = colonMatch[1];
    if (matched) {
      fromLine = parseInt(matched, 10);
      inputPath = inputPath.slice(0, -colonMatch[0].length);
    }
  }

  if (isDocid(inputPath)) {
    const docidMatch = findDocumentByDocid(db, inputPath);
    if (docidMatch) {
      inputPath = docidMatch.filepath;
    } else {
      throw new Error(`Document not found: ${originalPath}`);
    }
  }

  let doc: { collectionName: string; path: string; body: string; title: string; hash: string } | null = null;
  let virtualPath: string;

  if (isVirtualPath(inputPath)) {
    const parsed = parseVirtualPath(inputPath);
    if (!parsed) {
      throw new Error(`Invalid virtual path: ${inputPath}`);
    }

    doc = db.prepare(`
      SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE d.collection = ? AND d.path = ? AND d.active = 1
    `).get(parsed.collectionName, parsed.path) as typeof doc;

    if (!doc) {
      doc = db.prepare(`
        SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
        LIMIT 1
      `).get(parsed.collectionName, `%${parsed.path}`) as typeof doc;
    }

    virtualPath = inputPath;
  } else {
    if (!inputPath.startsWith("/") && !inputPath.startsWith("~")) {
      const parts = inputPath.split("/");
      if (parts.length >= 2) {
        const possibleCollection = parts[0];
        const possiblePath = parts.slice(1).join("/");
        const collExists = possibleCollection ? db.prepare(`
          SELECT 1 FROM documents WHERE collection = ? AND active = 1 LIMIT 1
        `).get(possibleCollection) : null;

        if (collExists) {
          doc = db.prepare(`
            SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
            FROM documents d
            JOIN content ON content.hash = d.hash
            WHERE d.collection = ? AND d.path = ? AND d.active = 1
          `).get(possibleCollection || "", possiblePath || "") as typeof doc;

          if (!doc) {
            doc = db.prepare(`
              SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
              FROM documents d
              JOIN content ON content.hash = d.hash
              WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
              LIMIT 1
            `).get(possibleCollection || "", `%${possiblePath}`) as typeof doc;
          }

          if (doc) {
            virtualPath = buildVirtualPath(doc.collectionName, doc.path);
          }
        }
      }
    }

    if (!doc) {
      let fsPath = inputPath;

      if (fsPath.startsWith("~/")) {
        fsPath = homedir() + fsPath.slice(1);
      } else if (!fsPath.startsWith("/")) {
        const cwd = args.cwd || getPwd();
        fsPath = resolve(cwd, fsPath);
      }
      fsPath = getRealPath(fsPath);

      const detected = detectCollectionFromPath(db, fsPath);
      if (detected) {
        doc = db.prepare(`
          SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.collection = ? AND d.path = ? AND d.active = 1
        `).get(detected.collectionName, detected.relativePath) as typeof doc;
      }

      if (!doc) {
        const filename = inputPath.split("/").pop() || inputPath;
        doc = db.prepare(`
          SELECT d.collection as collectionName, d.path, content.doc as body, d.title, d.hash
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${filename}`) as typeof doc;
      }

      if (doc) {
        virtualPath = buildVirtualPath(doc.collectionName, doc.path);
      } else {
        virtualPath = inputPath;
      }
    }
  }

  if (!doc) {
    throw new Error(`Document not found: ${originalPath}`);
  }

  const context = getContextForPath(db, doc.collectionName, doc.path);

  let output = doc.body;
  const startLine = fromLine || 1;
  if (fromLine !== undefined || args.maxLines !== undefined) {
    const lines = output.split("\n");
    const start = startLine - 1;
    const end = args.maxLines !== undefined ? start + args.maxLines : lines.length;
    output = lines.slice(start, end).join("\n");
  }

  return {
    file: virtualPath!,
    title: doc.title,
    body: output,
    context,
    hash: doc.hash,
    docid: doc.hash.slice(0, 6),
    startLine,
  };
}

/**
 * Handle a multi-get command
 */
async function handleMultiGet(args: MultiGetArgs): Promise<unknown> {
  const db = getDb(args.dbPath);
  const maxBytes = args.maxBytes ?? DEFAULT_MULTI_GET_MAX_BYTES;

  const isCommaSeparated = args.pattern.includes(",") && !args.pattern.includes("*") && !args.pattern.includes("?");

  let files: { filepath: string; displayPath: string; bodyLength: number; collection?: string; path?: string }[] = [];
  const errors: string[] = [];

  if (isCommaSeparated) {
    const names = args.pattern.split(",").map(s => s.trim()).filter(Boolean);
    for (const name of names) {
      let doc: { virtual_path: string; body_length: number; collection: string; path: string } | null = null;

      if (isVirtualPath(name)) {
        const parsed = parseVirtualPath(name);
        if (parsed) {
          doc = db.prepare(`
            SELECT
              'qmd://' || d.collection || '/' || d.path as virtual_path,
              LENGTH(content.doc) as body_length,
              d.collection,
              d.path
            FROM documents d
            JOIN content ON content.hash = d.hash
            WHERE d.collection = ? AND d.path = ? AND d.active = 1
          `).get(parsed.collectionName, parsed.path) as typeof doc;
        }
      } else {
        doc = db.prepare(`
          SELECT
            'qmd://' || d.collection || '/' || d.path as virtual_path,
            LENGTH(content.doc) as body_length,
            d.collection,
            d.path
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.path = ? AND d.active = 1
          LIMIT 1
        `).get(name) as typeof doc;

        if (!doc) {
          doc = db.prepare(`
            SELECT
              'qmd://' || d.collection || '/' || d.path as virtual_path,
              LENGTH(content.doc) as body_length,
              d.collection,
              d.path
            FROM documents d
            JOIN content ON content.hash = d.hash
            WHERE d.path LIKE ? AND d.active = 1
            LIMIT 1
          `).get(`%${name}`) as typeof doc;
        }
      }

      if (doc) {
        files.push({
          filepath: doc.virtual_path,
          displayPath: doc.virtual_path,
          bodyLength: doc.body_length,
          collection: doc.collection,
          path: doc.path,
        });
      } else {
        errors.push(`File not found: ${name}`);
      }
    }
  } else {
    files = matchFilesByGlob(db, args.pattern).map(f => ({
      ...f,
      collection: undefined,
      path: undefined,
    }));
    if (files.length === 0) {
      throw new Error(`No files matched pattern: ${args.pattern}`);
    }
  }

  const results: { file: string; displayPath: string; title: string; body: string; context: string | null; skipped: boolean; skipReason?: string }[] = [];

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

    const doc = db.prepare(`
      SELECT content.doc as body, d.title
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE d.collection = ? AND d.path = ? AND d.active = 1
    `).get(collection, path) as { body: string; title: string } | null;

    if (!doc) continue;

    let body = doc.body;
    if (args.maxLines !== undefined) {
      const lines = body.split("\n");
      body = lines.slice(0, args.maxLines).join("\n");
      if (lines.length > args.maxLines) {
        body += `\n\n[... truncated ${lines.length - args.maxLines} more lines]`;
      }
    }

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
async function handleLs(args: LsArgs): Promise<unknown> {
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

  // Parse path
  let collectionName: string;
  let pathPrefix: string | null = null;

  if (args.path.startsWith('qmd://')) {
    const parsed = parseVirtualPath(args.path);
    if (!parsed) {
      throw new Error(`Invalid virtual path: ${args.path}`);
    }
    collectionName = parsed.collectionName;
    pathPrefix = parsed.path;
  } else {
    const parts = args.path.split('/');
    collectionName = parts[0] || '';
    if (parts.length > 1) {
      pathPrefix = parts.slice(1).join('/');
    }
  }

  const coll = getCollectionFromYaml(collectionName);
  if (!coll) {
    throw new Error(`Collection not found: ${collectionName}`);
  }

  let query: string;
  let params: any[];

  if (pathPrefix) {
    query = `
      SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) as size
      FROM documents d
      JOIN content ct ON d.hash = ct.hash
      WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.name, `${pathPrefix}%`];
  } else {
    query = `
      SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) as size
      FROM documents d
      JOIN content ct ON d.hash = ct.hash
      WHERE d.collection = ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.name];
  }

  const files = db.prepare(query).all(...params) as { path: string; title: string; modified_at: string; size: number }[];

  return {
    mode: "files",
    collectionName,
    pathPrefix,
    files: files.map(f => ({
      path: f.path,
      title: f.title,
      modifiedAt: f.modified_at,
      size: f.size,
    })),
  };
}

/**
 * Handle status command
 */
async function handleStatus(args: { dbPath?: string }): Promise<unknown> {
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
    let result: unknown;

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
        result = { pong: true, pid: process.pid };
        break;
      default:
        throw new Error(`Unknown command: ${req.cmd}`);
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

        // Process complete lines (NDJSON)
        const lines = socket.data.buffer.split('\n');
        socket.data.buffer = lines.pop() || ""; // Keep incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const req = JSON.parse(line) as DaemonRequest;
            const res = await handleCommand(req);
            socket.write(JSON.stringify(res) + '\n');
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            const res: DaemonResponse = { ok: false, error: `Parse error: ${message}` };
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
        const bufferInfo = socket.data?.buffer ? ` (partial: ${socket.data.buffer.slice(0, 50)}...)` : '';
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

export function runDaemonForeground(): void {
  startServer();
}

// =============================================================================
// Client Functions (for use by CLI)
// =============================================================================

/**
 * Send a request to the daemon and get a response
 */
export async function sendToDaemon(req: DaemonRequest, timeoutMs = 30000): Promise<DaemonResponse> {
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
        reject(new Error(`Daemon request timed out for '${req.cmd}'`));
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
            reject(err);
          }
        },
        error(socket, error) {
          if (!resolved) {
            cleanup();
            const wrapped = new Error(`Daemon socket error during '${req.cmd}': ${error.message || error}`);
            (wrapped as any).cause = error;
            reject(wrapped);
          }
        },
        close() {
          if (!resolved) {
            cleanup();
            reject(new Error(`Connection closed by daemon before response for '${req.cmd}'`));
          }
        },
        connectError(socket, error) {
          if (!resolved) {
            cleanup();
            const wrapped = new Error(`Failed to connect to daemon for '${req.cmd}': ${error.message || error}`);
            (wrapped as any).cause = error;
            reject(wrapped);
          }
        },
      },
    });
  });
}

/**
 * Clean up stale socket and PID files
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
 * Check if daemon is running
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
 * Get daemon status
 */
export async function getDaemonStatus(): Promise<DaemonStatus> {
  if (!isDaemonRunning()) {
    return { running: false };
  }

  try {
    const res = await sendToDaemon({ cmd: "daemon-status", args: {} }, 5000);
    if (res.ok) {
      return res.result as DaemonStatus;
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
 * Stop the daemon. Returns { stopped: true } on success, { stopped: false, reason: string } on failure.
 */
export function stopDaemon(): { stopped: boolean; reason?: string } {
  if (!existsSync(PID_PATH)) {
    return { stopped: false, reason: "No PID file found" };
  }

  try {
    const pidContent = readFileSync(PID_PATH, 'utf-8').trim();
    const pid = parseInt(pidContent, 10);
    if (isNaN(pid)) {
      return { stopped: false, reason: "Invalid PID file content" };
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
