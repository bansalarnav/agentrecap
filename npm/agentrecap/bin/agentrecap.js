#!/usr/bin/env node

const childProcess = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const platform = os.platform();
const arch = os.arch();

function isMusl() {
  if (platform !== "linux") {
    return false;
  }

  const report = process.report?.getReport();
  if (report?.header?.glibcVersionRuntime) {
    return false;
  }

  try {
    const result = childProcess.spawnSync("ldd", ["--version"], {
      encoding: "utf8",
    });
    return `${result.stdout || ""}${result.stderr || ""}`
      .toLowerCase()
      .includes("musl");
  } catch {
    return true;
  }
}

const suffix = platform === "linux" && isMusl() ? "-musl" : "";
const packagePlatform = platform === "win32" ? "windows" : platform;
const packageName = `agentrecap-${packagePlatform}-${arch}${suffix}`;
const binaryName = platform === "win32" ? "agentrecap.exe" : "agentrecap";

let binary;
try {
  const manifest = require.resolve(`${packageName}/package.json`);
  binary = path.join(path.dirname(manifest), "bin", binaryName);
} catch {
  console.error(
    `agentrecap does not have an installed binary for ${platform}-${arch}${suffix}.`,
  );
  console.error(`Try installing ${packageName} manually.`);
  process.exit(1);
}

if (!fs.existsSync(binary)) {
  console.error(`The agentrecap binary is missing from ${packageName}.`);
  process.exit(1);
}

const child = childProcess.spawn(binary, process.argv.slice(2), {
  stdio: "inherit",
});

const signalHandlers = {};
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  signalHandlers[signal] = () => {
    try {
      child.kill(signal);
    } catch {
      // The child may already have exited.
    }
  };
  process.on(signal, signalHandlers[signal]);
}

child.on("error", (error) => {
  console.error(error.message);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  for (const [forwardedSignal, handler] of Object.entries(signalHandlers)) {
    process.removeListener(forwardedSignal, handler);
  }

  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
