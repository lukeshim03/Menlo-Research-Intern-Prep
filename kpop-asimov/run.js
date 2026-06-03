// Node wrapper for `npm run dev`.
// Locates a working Python 3.12 + the venv's installed packages, then runs run.py.
// Falls back to the uv base interpreter + PYTHONPATH if the venv launcher is broken.

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

const HERE = __dirname;
const venvPy = path.join(HERE, ".venv", "Scripts", "python.exe");
const sitePackages = path.join(HERE, ".venv", "Lib", "site-packages");

// uv-managed base interpreter (read home= from pyvenv.cfg)
function basePythonFromCfg() {
  try {
    const cfg = fs.readFileSync(path.join(HERE, ".venv", "pyvenv.cfg"), "utf8");
    const m = cfg.match(/home\s*=\s*(.+)/);
    if (m) {
      const home = m[1].replace(/[\r\n]+$/, "").trim();
      const exe = path.join(home, "python.exe");
      if (fs.existsSync(exe)) return exe;
    }
  } catch (_) {}
  // last resort: scan uv's python dir for any cpython-3.12 build
  try {
    const uvPy = path.join(os.homedir(), "AppData", "Roaming", "uv", "python");
    for (const d of fs.readdirSync(uvPy)) {
      if (d.startsWith("cpython-3.12")) {
        const exe = path.join(uvPy, d, "python.exe");
        if (fs.existsSync(exe)) return exe;
      }
    }
  } catch (_) {}
  return null;
}

function works(exe, env) {
  const r = spawnSync(exe, ["-c", "import mujoco"], { env });
  return r.status === 0;
}

const args = ["run.py", ...process.argv.slice(2)];
const baseEnv = { ...process.env };

// 1) try the venv launcher as-is
if (fs.existsSync(venvPy) && works(venvPy, baseEnv)) {
  run(venvPy, baseEnv);
} else {
  // 2) fall back: base interpreter + PYTHONPATH to venv packages
  const base = basePythonFromCfg();
  if (!base) {
    console.error("[error] No working Python found. Recreate the venv:\n" +
      "  uv venv .venv --python 3.12\n" +
      "  uv pip install -r requirements.txt --python .venv\\Scripts\\python.exe");
    process.exit(1);
  }
  const env = { ...baseEnv, PYTHONPATH: sitePackages };
  if (!works(base, env)) {
    console.error("[error] venv packages not importable. Reinstall:\n" +
      "  uv pip install -r requirements.txt --python .venv\\Scripts\\python.exe");
    process.exit(1);
  }
  console.log("[info] using base interpreter + venv packages (venv launcher bypassed)");
  run(base, env);
}

function run(exe, env) {
  const r = spawnSync(exe, args, { cwd: HERE, env, stdio: "inherit" });
  process.exit(r.status === null ? 1 : r.status);
}
