// Issue triage only. Never run on pull requests. Never close issues.
// Do not put GitHub closing keywords (fix/close/resolve #N) in commits that
// only touch this file.
module.exports = async function triage({ github, context, core, issueNumber }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const num = Number(issueNumber);
  if (!num) {
    core.warning("no issue number");
    return;
  }

  const { data: issue } = await github.rest.issues.get({
    owner, repo, issue_number: num,
  });
  if (issue.pull_request) {
    core.info(`#${num} is a PR — skip issue triage`);
    return;
  }

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: num,
  });
  const allText = [issue.body, issue.title, ...comments.map((c) => c.body)]
    .map((s) => s || "")
    .join("\n")
    .toLowerCase();

  const labels = new Set((issue.labels || []).map((l) => l.name));
  const add = new Set();
  const remove = new Set();

  const hasTP = (n) =>
    new RegExp(
      String.raw`\btp\s*[:=]?\s*${n}\b|tensor-parallel-size\s+${n}\b`,
      "i"
    ).test(allText);
  const hasB70 = (n) =>
    new RegExp(String.raw`\b${n}\s*[×x]\s*b70\b|\b${n}\s*x\s*b70\b`, "i").test(
      allText
    );

  if (hasB70(4) || hasTP(4)) add.add("b70:4");
  else if (hasB70(2) || hasTP(2)) add.add("b70:2");
  else if (hasB70(1) || hasTP(1)) add.add("b70:1");

  if (hasTP(4)) add.add("tp:4");
  else if (hasTP(2)) add.add("tp:2");
  else if (hasTP(1)) add.add("tp:1");
  if (/\bpp\s*2\b/i.test(allText)) add.add("pp:2");

  if (
    allText.includes("xe coredump") ||
    allText.includes("timedout job") ||
    allText.includes("guc_id") ||
    allText.includes("driver hang")
  ) {
    add.add("driver-hang");
  }
  if (
    allText.includes("tool-call") ||
    allText.includes("tool_call") ||
    allText.includes("enable-auto-tool-choice") ||
    allText.includes("qwen3_xml")
  ) {
    add.add("tool-calling");
  }
  if (allText.includes("draft") && allText.includes("int4")) add.add("draft-int4");
  if (
    allText.includes("enable-prefix-caching") &&
    !allText.includes("no-enable-prefix-caching")
  ) {
    add.add("prefix-cache");
  }

  if (
    (labels.has("tp:2") || add.has("tp:2")) &&
    (labels.has("draft-int4") || add.has("draft-int4"))
  ) {
    add.add("tp2-draft-block");
  }
  if (allText.includes("patch_draft_lmhead") && /\btp\s*2\b/.test(allText)) {
    add.add("tp2-draft-block");
  }
  if (allText.includes("patch_draft_mtp_int4") && /\btp\s*2\b/.test(allText)) {
    add.add("tp2-draft-block");
  }

  const hasImage =
    /sha256:[0-9a-f]{8,}/i.test(allText) ||
    allText.includes("f01e24f") ||
    allText.includes("2c427ef") ||
    allText.includes("1da0a954");
  const hasCmd =
    allText.includes("docker run") ||
    allText.includes("vllm serve") ||
    allText.includes("--quantization") ||
    allText.includes("--gpu-memory-utilization");
  const hasLog =
    allText.includes("xe ") ||
    allText.includes("seqno") ||
    allText.includes("tok=") ||
    allText.includes("finish_reason") ||
    allText.includes("coredump");

  const title = (issue.title || "").trim();
  const looksLikeHowTo =
    /^(how |what |can i |why |is there )/i.test(title) ||
    /\?\s*$/.test(title);
  const isQuestion =
    looksLikeHowTo && !hasImage && !hasCmd && !add.has("driver-hang");

  const hangNeedsLog = add.has("driver-hang") && !hasLog;
  const needsInfo = !isQuestion && (!hasImage || !hasCmd || hangNeedsLog);

  if (isQuestion) {
    add.add("question");
    remove.add("ready");
  } else if (needsInfo) {
    add.add("needs-info");
    remove.add("ready");
  } else {
    add.add("ready");
    remove.add("needs-info");
  }

  const toAdd = [...add].filter((l) => !labels.has(l));
  const toRemove = [...remove].filter((l) => labels.has(l));
  for (const l of toAdd) {
    try {
      await github.rest.issues.addLabels({
        owner, repo, issue_number: num, labels: [l],
      });
    } catch (e) {
      core.info(`add ${l}: ${e.message}`);
    }
  }
  for (const l of toRemove) {
    try {
      await github.rest.issues.removeLabel({
        owner, repo, issue_number: num, name: l,
      });
    } catch (e) {
      /* already gone */
    }
  }

  const botComments = comments.filter(
    (c) => c.user && c.user.type === "Bot" && (c.body || "").includes("cookbook-triage")
  );
  const hasBot = (needle) => botComments.some((c) => (c.body || "").includes(needle));

  async function once(marker, body) {
    if (hasBot(marker)) return;
    await github.rest.issues.createComment({
      owner, repo, issue_number: num,
      body: `<!-- cookbook-triage:${marker} -->\n${body}\n`,
    });
  }

  if (add.has("tp2-draft-block")) {
    await once(
      "tp2-draft-block",
      "TP2 + draft-INT4 is single-card only (QWEN38-VLLM-XPU.md §12). Drop `patch_draft_lmhead_int4.py` and `patch_draft_mtp_int4.py` for TP>1; keep `mtp_nightly`, `mtp_boundary`, `gdn_mixed_split_v5`.\n\nLeave this issue open if you have a new fact (new image, missing runtime skip, numbers)."
    );
  }
  if (hangNeedsLog) {
    await once(
      "needs-dmesg",
      "Driver hang: paste ~30 lines of dmesg around `Xe device coredump` / `Timedout job` (seqno, guc_id) and the full docker/serve line."
    );
  }
  if (needsInfo && !add.has("driver-hang")) {
    await once(
      "needs-info",
      "Need the image sha256 and the full docker/serve line. Then this moves to `ready`."
    );
  }
  if (isQuestion) {
    await once(
      "question",
      `Questions go in Discussions: https://github.com/${owner}/${repo}/discussions\n\nIssues stay for a repro (image sha + command + what broke).`
    );
  }

  core.summary.addHeading("triage");
  core.summary.addRaw(
    `#${num} add=[${[...add].join(",")}] remove=[${[...remove].join(",")}] question=${isQuestion} needsInfo=${needsInfo}`
  );
  await core.summary.write();
};
