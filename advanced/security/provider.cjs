// Fixed local Python entry point; no shell, arbitrary commands or remote provider.
const {spawnSync} = require('node:child_process');
const path = require('node:path');
module.exports = class DemoShopProvider {
  id() { return 'demoshop-local-tools'; }
  async callApi(prompt, context) {
    const root = path.resolve(__dirname, '../..');
    const python = process.platform === 'win32'
      ? path.join(root, '.venv/Scripts/python.exe')
      : path.join(root, '.venv/bin/python');
    const env = {...process.env};
    for (const key of Object.keys(env)) {
      if (/API_KEY|SECRET|LANGSMITH|LANGCHAIN|OTEL_/i.test(key)) delete env[key];
    }
    const result = spawnSync(python, ['-m', 'ai_quality_evals_advanced.security_target'], {
      cwd: root, env, input: JSON.stringify(context.vars), encoding: 'utf8',
      timeout: 10000, windowsHide: true,
    });
    if (result.error || result.status !== 0) return {error: 'Local target execution failed'};
    return {output: result.stdout.trim()};
  }
};
