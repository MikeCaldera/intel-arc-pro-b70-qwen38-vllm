// One-shot PR comment. Never mentions APPLY_SERGIO (internal overlay flag).
// Never closes anything.
module.exports = async function prPrecheck({ github, context, core }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const num = context.payload.pull_request && context.payload.pull_request.number;
  if (!num) return;
  if (context.payload.action !== "opened" && context.payload.action !== "reopened") {
    core.info(`skip comment on ${context.payload.action}`);
    return;
  }

  const { data: files } = await github.rest.pulls.listFiles({
    owner, repo, pull_number: num, per_page: 100,
  });
  const paths = files.map((f) => f.filename);
  const patchPy = paths.filter((p) => p.startsWith("patches/") && p.endsWith(".py"));
  const hasVerifier = paths.some((p) => p.includes("verify-") && p.endsWith(".sh"));
  const body = (context.payload.pull_request.body || "").toLowerCase();
  const hasImage =
    /sha256:[0-9a-f]{8,}/i.test(body) ||
    body.includes("f01e24f") ||
    body.includes("2c427ef");
  const hasRan =
    body.includes("all checks passed") ||
    body.includes("verify-") ||
    body.includes("gpu-free") ||
    body.includes("docker run");

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: num,
  });
  if (comments.some((c) => (c.body || "").includes("cookbook-triage:pr-precheck"))) {
    return;
  }

  const lines = [];
  if (patchPy.length) {
    lines.push(
      `Patches in this PR: ${patchPy.map((p) => "`" + p + "`").join(", ")}.`
    );
    if (hasVerifier) {
      lines.push("Includes a verify script. GPU-free `docker run` without `--device` is enough for text transforms.");
    } else {
      lines.push(
        "No verify script. For text patches, a GPU-free apply + `py_compile` + idempotent re-apply on the pinned image is the bar."
      );
    }
    if (!hasImage) {
      lines.push("PR body has no image digest. Name the pinned sha (`f01e24f6` / `2c427ef` / full sha256).");
    }
    if (!hasRan) {
      lines.push("PR body does not say what was run. Paste the command and the last line of output.");
    }
  } else {
    lines.push("No `patches/*.py` in this PR. Docs/CI-only: say which page or workflow you changed.");
  }
  lines.push(
    "Do not claim a live corruption rate unless you measured one. This lab will not treat GPU-free apply as a soak."
  );

  await github.rest.issues.createComment({
    owner, repo, issue_number: num,
    body: `<!-- cookbook-triage:pr-precheck -->\n${lines.join("\n\n")}\n`,
  });
};
