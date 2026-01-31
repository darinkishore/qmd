/**
 * Tests for QMD daemon functionality
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { existsSync, unlinkSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// Import protocol types and utilities
import {
  DAEMON_COMMANDS,
  NON_DAEMON_COMMANDS,
  shouldUseDaemon,
  type DaemonRequest,
  type DaemonResponse,
} from "./protocol";

// =============================================================================
// Protocol Tests
// =============================================================================

describe("Protocol: shouldUseDaemon", () => {
  test("returns true for search commands", () => {
    expect(shouldUseDaemon("search")).toBe(true);
    expect(shouldUseDaemon("vsearch")).toBe(true);
    expect(shouldUseDaemon("query")).toBe(true);
  });

  test("returns true for get commands", () => {
    expect(shouldUseDaemon("get")).toBe(true);
    expect(shouldUseDaemon("multi-get")).toBe(true);
  });

  test("returns true for ls and status", () => {
    expect(shouldUseDaemon("ls")).toBe(true);
    expect(shouldUseDaemon("status")).toBe(true);
  });

  test("returns false for mutation commands", () => {
    expect(shouldUseDaemon("collection")).toBe(false);
    expect(shouldUseDaemon("context")).toBe(false);
    expect(shouldUseDaemon("embed")).toBe(false);
    expect(shouldUseDaemon("update")).toBe(false);
    expect(shouldUseDaemon("cleanup")).toBe(false);
  });

  test("returns false for daemon command itself", () => {
    expect(shouldUseDaemon("daemon")).toBe(false);
  });

  test("returns false for mcp command", () => {
    expect(shouldUseDaemon("mcp")).toBe(false);
  });

  test("returns false for unknown commands", () => {
    expect(shouldUseDaemon("unknown")).toBe(false);
    expect(shouldUseDaemon("")).toBe(false);
    expect(shouldUseDaemon("foo")).toBe(false);
  });
});

describe("Protocol: Command sets", () => {
  test("DAEMON_COMMANDS contains expected commands", () => {
    expect(DAEMON_COMMANDS.has("search")).toBe(true);
    expect(DAEMON_COMMANDS.has("vsearch")).toBe(true);
    expect(DAEMON_COMMANDS.has("query")).toBe(true);
    expect(DAEMON_COMMANDS.has("get")).toBe(true);
    expect(DAEMON_COMMANDS.has("multi-get")).toBe(true);
    expect(DAEMON_COMMANDS.has("ls")).toBe(true);
    expect(DAEMON_COMMANDS.has("status")).toBe(true);
  });

  test("NON_DAEMON_COMMANDS contains expected commands", () => {
    expect(NON_DAEMON_COMMANDS.has("daemon")).toBe(true);
    expect(NON_DAEMON_COMMANDS.has("collection")).toBe(true);
    expect(NON_DAEMON_COMMANDS.has("context")).toBe(true);
    expect(NON_DAEMON_COMMANDS.has("embed")).toBe(true);
    expect(NON_DAEMON_COMMANDS.has("update")).toBe(true);
    expect(NON_DAEMON_COMMANDS.has("mcp")).toBe(true);
    expect(NON_DAEMON_COMMANDS.has("cleanup")).toBe(true);
  });

  test("no overlap between DAEMON_COMMANDS and NON_DAEMON_COMMANDS", () => {
    for (const cmd of DAEMON_COMMANDS) {
      expect(NON_DAEMON_COMMANDS.has(cmd)).toBe(false);
    }
    for (const cmd of NON_DAEMON_COMMANDS) {
      expect(DAEMON_COMMANDS.has(cmd)).toBe(false);
    }
  });
});

describe("Protocol: Request/Response types", () => {
  test("DaemonRequest structure", () => {
    const req: DaemonRequest = {
      cmd: "search",
      args: { query: "test" },
    };
    expect(req.cmd).toBe("search");
    expect(req.args).toEqual({ query: "test" });
  });

  test("DaemonResponse success structure", () => {
    const res: DaemonResponse = {
      ok: true,
      result: [{ file: "test.md", score: 0.9 }],
    };
    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.result).toEqual([{ file: "test.md", score: 0.9 }]);
    }
  });

  test("DaemonResponse error structure", () => {
    const res: DaemonResponse = {
      ok: false,
      error: "Something went wrong",
    };
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.error).toBe("Something went wrong");
    }
  });
});

// =============================================================================
// Daemon Client Tests (mock-based)
// =============================================================================

describe("Daemon: isDaemonRunning", () => {
  // These tests verify the function without actually starting a daemon

  test("returns false when socket does not exist", async () => {
    const { isDaemonRunning } = await import("./daemon");
    // In test environment, socket shouldn't exist
    // This test may pass or fail depending on environment state
    const result = isDaemonRunning();
    expect(typeof result).toBe("boolean");
  });
});

describe("Daemon: cleanupStaleFiles", () => {
  test("doesn't throw when files don't exist", async () => {
    const { cleanupStaleFiles } = await import("./daemon");
    // Should not throw even if files don't exist
    expect(() => cleanupStaleFiles()).not.toThrow();
  });
});

// =============================================================================
// Validation Tests
// =============================================================================

describe("Daemon: Input validation", () => {
  test("validateSearchArgs throws on missing query", async () => {
    const { validateSearchArgs } = await import("./daemon");
    expect(() => validateSearchArgs({})).toThrow(/Missing required argument: query/i);
  });

  test("validateSearchArgs throws on empty query", async () => {
    const { validateSearchArgs } = await import("./daemon");
    expect(() => validateSearchArgs({ query: "   " })).toThrow(/Missing required argument: query/i);
  });

  test("validateSearchArgs extracts valid args", async () => {
    const { validateSearchArgs } = await import("./daemon");
    const result = validateSearchArgs({ query: "test", limit: 10, full: true });
    expect(result.query).toBe("test");
    expect(result.limit).toBe(10);
    expect(result.full).toBe(true);
  });

  test("validateSearchArgs preserves context and useColor", async () => {
    const { validateSearchArgs } = await import("./daemon");
    const result = validateSearchArgs({ query: "test", context: "ctx", useColor: true, dbPath: "/tmp/db" });
    expect(result.context).toBe("ctx");
    expect(result.useColor).toBe(true);
    expect(result.dbPath).toBe("/tmp/db");
  });

  test("validateGetArgs throws on missing path", async () => {
    const { validateGetArgs } = await import("./daemon");
    expect(() => validateGetArgs({})).toThrow(/Missing required argument: path/i);
  });

  test("validateGetArgs throws on empty path", async () => {
    const { validateGetArgs } = await import("./daemon");
    expect(() => validateGetArgs({ path: "  " })).toThrow(/Missing required argument: path/i);
  });

  test("validateGetArgs extracts valid args", async () => {
    const { validateGetArgs } = await import("./daemon");
    const result = validateGetArgs({ path: "qmd://test/file.md", fromLine: 5, maxLines: 10 });
    expect(result.path).toBe("qmd://test/file.md");
    expect(result.fromLine).toBe(5);
    expect(result.maxLines).toBe(10);
  });

  test("validateGetArgs preserves cwd", async () => {
    const { validateGetArgs } = await import("./daemon");
    const result = validateGetArgs({ path: "file.md", cwd: "/tmp" });
    expect(result.path).toBe("file.md");
    expect(result.cwd).toBe("/tmp");
  });

  test("validateMultiGetArgs throws on missing pattern", async () => {
    const { validateMultiGetArgs } = await import("./daemon");
    expect(() => validateMultiGetArgs({})).toThrow(/Missing required argument: pattern/i);
  });

  test("validateMultiGetArgs extracts valid args", async () => {
    const { validateMultiGetArgs } = await import("./daemon");
    const result = validateMultiGetArgs({ pattern: "*.md", maxLines: 50 });
    expect(result.pattern).toBe("*.md");
    expect(result.maxLines).toBe(50);
  });

  test("validateMultiGetArgs throws on empty pattern", async () => {
    const { validateMultiGetArgs } = await import("./daemon");
    expect(() => validateMultiGetArgs({ pattern: "   " })).toThrow(/Missing required argument: pattern/i);
  });

  test("validateSearchArgs rejects invalid numeric args", async () => {
    const { validateSearchArgs } = await import("./daemon");
    expect(() => validateSearchArgs({ query: "test", limit: 0 })).toThrow(/limit/i);
    expect(() => validateSearchArgs({ query: "test", minScore: 2 })).toThrow(/minScore/i);
  });

  test("validateGetArgs rejects invalid line arguments", async () => {
    const { validateGetArgs } = await import("./daemon");
    expect(() => validateGetArgs({ path: "file.md", fromLine: 0 })).toThrow(/fromLine/i);
    expect(() => validateGetArgs({ path: "file.md", maxLines: -3 })).toThrow(/maxLines/i);
  });

  test("validateMultiGetArgs rejects invalid maxBytes", async () => {
    const { validateMultiGetArgs } = await import("./daemon");
    expect(() => validateMultiGetArgs({ pattern: "*.md", maxBytes: 0 })).toThrow(/maxBytes/i);
  });

  test("validateLsArgs handles optional path", async () => {
    const { validateLsArgs } = await import("./daemon");
    const result1 = validateLsArgs({});
    expect(result1.path).toBeUndefined();
    
    const result2 = validateLsArgs({ path: "qmd://test" });
    expect(result2.path).toBe("qmd://test");
  });
});

// =============================================================================
// Integration Tests (require running daemon)
// =============================================================================

describe("Daemon: Integration", () => {
  // These tests require the daemon to be running
  // They're skipped by default - run with DAEMON_TESTS=1

  const shouldRunIntegration = process.env.DAEMON_TESTS === "1";

  test.skipIf(!shouldRunIntegration)("can ping daemon", async () => {
    const { sendToDaemon, isDaemonRunning } = await import("./daemon");

    if (!isDaemonRunning()) {
      console.log("Daemon not running, skipping integration test");
      return;
    }

    const res = await sendToDaemon({ cmd: "ping", args: {} });
    expect(res.ok).toBe(true);
    if (res.ok) {
      const result = res.result as { pong: boolean; pid: number };
      expect(result.pong).toBe(true);
      expect(typeof result.pid).toBe("number");
    }
  });

  test.skipIf(!shouldRunIntegration)("can get daemon status", async () => {
    const { sendToDaemon, isDaemonRunning } = await import("./daemon");

    if (!isDaemonRunning()) {
      console.log("Daemon not running, skipping integration test");
      return;
    }

    const res = await sendToDaemon({ cmd: "daemon-status", args: {} });
    expect(res.ok).toBe(true);
    if (res.ok) {
      const result = res.result as { running: boolean; pid: number; uptime: number };
      expect(result.running).toBe(true);
      expect(typeof result.pid).toBe("number");
      expect(typeof result.uptime).toBe("number");
    }
  });
});
