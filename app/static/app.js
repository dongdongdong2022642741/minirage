"use strict";

const $ = (sel) => document.querySelector(sel);

const els = {
  chat: $("#chat"), chatEmpty: $("#chat-empty"), input: $("#input"),
  userSelect: $("#user-select"), kbSwitch: $("#kb-switch"), curKb: $("#cur-kb"),
  composer: $("#composer"), sendBtn: $("#btn-send"),
  fileInput: $("#file-input"), dirInput: $("#dir-input"),
  btnImport: $("#btn-import"), btnRebuild: $("#btn-rebuild"), btnEval: $("#btn-eval"),
  docUl: $("#doc-ul"), docEmpty: $("#doc-empty"),
  statDocs: $("#stat-docs"), statChunks: $("#stat-chunks"), statTerms: $("#stat-terms"),
  builtBadge: $("#built-badge"), evalPanel: $("#eval-panel"),
  qInfo: $("#q-info"), questionsFile: $("#questions-file"),
  drawer: $("#drawer"), drawerLabel: $("#drawer-label"),
  drawerMeta: $("#drawer-meta"), drawerText: $("#drawer-text"),
  drawerClose: $("#btn-drawer-close"), toast: $("#toast"),
};

let currentKb = localStorage.getItem("mr.kb") || "main";
let activeEv = null;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `请求失败 (${res.status})`);
  return data;
}

function toast(text, ms = 2600) {
  els.toast.textContent = text;
  els.toast.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.add("hidden"), ms);
}

function esc(text) {
  return String(text ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderCitations(text) {
  return esc(text).replace(/\[(\d{1,2})\]/g,
    '<button class="cite" data-cite="$1">$1</button>');
}

function setBusy(busy) {
  els.sendBtn.disabled = busy;
  els.btnImport.disabled = busy;
  els.btnRebuild.disabled = busy;
  els.btnEval.disabled = busy;
}

/* ---------- 知识库切换 ---------- */

async function refreshKbs() {
  const { kbs } = await api("/api/kbs");
  if (!kbs.some((k) => k.kb_id === currentKb)) currentKb = kbs[0]?.kb_id || "main";
  els.kbSwitch.innerHTML = kbs.map((k) => `
    <button class="kb-pill ${k.kb_id === currentKb ? "active" : ""}"
            data-kb="${esc(k.kb_id)}" title="${esc(k.description)}">
      ${esc(k.name)}
    </button>`).join("");
  els.kbSwitch.querySelectorAll(".kb-pill").forEach((pill) => {
    pill.addEventListener("click", () => setKb(pill.dataset.kb));
  });
  els.curKb.textContent =
    kbs.find((k) => k.kb_id === currentKb)?.name || currentKb;
}

function setKb(kbId) {
  if (kbId === currentKb) return;
  currentKb = kbId;
  localStorage.setItem("mr.kb", kbId);
  refreshKbs().catch(() => {});
  els.chatEmpty.classList.remove("hidden");
  addMessage("assistant",
    `<span class="sys-note">已切换到「${esc(els.curKb.textContent)}」——只能检索该库内授权给你的内容</span>`);
  refreshStatus().catch(() => {});
  refreshQuestions().catch(() => {});
  refreshUsers().catch(() => {});
}

/* ---------- 状态 / 文档 ---------- */

function badgeHtml(status) {
  if (status.built && status.chunks > 0) {
    return '<span class="badge badge-ok" id="built-badge">索引就绪</span>';
  }
  if (status.docs.length > 0) {
    return '<span class="badge badge-wait" id="built-badge">待重建索引</span>';
  }
  return '<span class="badge badge-bad" id="built-badge">空知识库</span>';
}

async function refreshStatus() {
  const status = await api(`/api/status?kb_id=${encodeURIComponent(currentKb)}`);
  els.statDocs.textContent = status.docs.length;
  els.statChunks.textContent = status.chunks ?? "—";
  els.statTerms.textContent = status.bm25_terms ?? "—";
  els.builtBadge.outerHTML = badgeHtml(status);

  els.docUl.innerHTML = "";
  for (const doc of status.docs) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <span class="doc-name" title="${esc(doc.name)}">${esc(doc.name)}</span>
      <span class="ev-score">v${doc.version} · ${doc.size} B</span>
      <button class="doc-del" data-id="${esc(doc.id)}" title="删除">×</button>`;
    li.querySelector(".doc-del").addEventListener("click", async () => {
      await api("/api/delete", { method: "POST",
        body: JSON.stringify({ document_id: doc.id, kb_id: currentKb }) });
      toast(`已删除 ${doc.name}`);
      await refreshStatus();
    });
    els.docUl.appendChild(li);
  }
  els.docEmpty.classList.toggle("hidden", status.docs.length > 0);
  await refreshQuestions();
  return status;
}

async function refreshQuestions() {
  const info = await api(`/api/questions?kb_id=${encodeURIComponent(currentKb)}`);
  if (info.exists) {
    els.qInfo.textContent =
      `${info.name} · ${info.count} 题（可答 ${info.answer_count} / 拒答 ${info.refuse_count}）`;
  } else {
    els.qInfo.textContent = "本库暂无题目集，可上传替换";
  }
}

async function refreshUsers() {
  const { users } = await api(`/api/users?kb_id=${encodeURIComponent(currentKb)}`);
  els.userSelect.innerHTML = users
    .map((u) => `<option value="${esc(u.user_id)}">${esc(u.display_name)}</option>`)
    .join("");
}

/* ---------- 聊天 ---------- */

function addThinking() {
  const msg = document.createElement("div");
  msg.className = "chat-msg assistant";
  msg.id = "thinking";
  msg.innerHTML = '<div class="bubble"><span class="thinking">检索中<span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div>';
  els.chat.appendChild(msg);
  els.chat.scrollTop = els.chat.scrollHeight;
  return msg;
}

function openDrawer(ev) {
  els.drawerLabel.textContent = ev.label;
  els.drawerMeta.textContent =
    `来源 #${ev.rank} · 相关度 ${ev.score.toFixed(4)} · 命中引用 [${ev.rank}]`;
  els.drawerText.textContent = ev.text;
  els.drawer.classList.remove("hidden");
}

function addMessage(role, html) {
  const msg = document.createElement("div");
  msg.className = `chat-msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  msg.appendChild(bubble);
  els.chat.appendChild(msg);
  els.chat.scrollTop = els.chat.scrollHeight;
  return msg;
}

function renderAnswer(data) {
  const msg = addMessage("assistant", renderCitations(data.answer));
  if (data.refusal) msg.querySelector(".bubble").classList.add("refusal");

  if (data.retried) {
    const note = document.createElement("div");
    note.className = "retry-note";
    note.textContent = "首轮回答疑似误拒，已按检索提示重试一次";
    msg.appendChild(note);
  }
  if (data.acl && data.acl.filtered > 0) {
    const note = document.createElement("div");
    note.className = "retry-note";
    note.textContent =
      `已按身份「${data.acl.user_id}」过滤 ${data.acl.filtered} 条无权限候选`;
    msg.appendChild(note);
  }

  const strip = document.createElement("div");
  strip.className = "ev-strip";
  const evCards = [];
  data.evidence.forEach((ev, i) => {
    const card = document.createElement("button");
    card.className = "ev-card";
    card.innerHTML =
      `<span class="ev-rank">#${ev.rank}</span>` +
      `<span class="ev-label">${esc(ev.label)}</span>` +
      `<span class="ev-score">${ev.score.toFixed(3)}</span>`;
    card.addEventListener("click", () => {
      evCards.forEach((c, j) => c.classList.toggle("active", j === i));
      openDrawer(ev);
    });
    evCards.push(card);
    strip.appendChild(card);
  });
  msg.appendChild(strip);

  msg.querySelectorAll(".cite").forEach((chip) => {
    chip.addEventListener("click", () => {
      const n = parseInt(chip.dataset.cite, 10);
      if (!data.evidence[n - 1]) return;
      evCards.forEach((c, j) => c.classList.toggle("active", j === n - 1));
      evCards[n - 1].scrollIntoView({ behavior: "smooth", block: "nearest" });
      openDrawer(data.evidence[n - 1]);
    });
  });
}

async function ask() {
  const query = els.input.value.trim();
  if (!query) return;
  els.input.value = "";
  els.input.style.height = "auto";
  els.chatEmpty.classList.add("hidden");
  addMessage("user", esc(query));
  const thinking = addThinking();
  setBusy(true);
  try {
    const data = await api("/api/ask", { method: "POST",
      body: JSON.stringify({ query, user_id: els.userSelect.value,
                             kb_id: currentKb }) });
    thinking.remove();
    renderAnswer(data);
  } catch (error) {
    thinking.remove();
    toast(`出错了：${error.message}`, 4000);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

/* ---------- 评测 ---------- */

async function waitBuildDone(jobId, timeoutMs = 300000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 800));
    try {
      const cur = await api(`/api/builds/current?kb_id=${encodeURIComponent(currentKb)}`);
      if (cur.job_id === jobId && (cur.state === "done" || cur.state === "failed")) {
        return cur;
      }
    } catch (_) { /* 任务未注册或已清理 */ }
  }
  return null;
}

async function renderEval() {
  els.btnEval.disabled = true;
  els.evalPanel.classList.remove("hidden");
  els.evalPanel.innerHTML = '<span class="thinking">自测中<span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
  try {
    const result = await api(`/api/eval?kb_id=${encodeURIComponent(currentKb)}`,
                             { method: "POST", body: "{}" });
    const s = result.summary;
    const rows = result.rows.map((r) => {
      const cls = r.ok ? "pass" : "fail";
      const mark = r.ok ? "PASS" : r.verdict;
      return `<div class="eval-row">
        <span class="eval-qid">Q${r.qid}</span>
        <span class="eval-verdict ${cls}">${mark}</span>
        <span class="eval-ans">${esc(r.answer).replace(/\n/g, " ").slice(0, 60)}</span>
      </div>`;
    }).join("");
    els.evalPanel.innerHTML = `
      <div class="eval-summary">
        <span class="chip">能答 ${s.answer_rate}</span>
        <span class="chip">拒答 ${s.refuse_rate}</span>
        <span class="chip">引用 ${s.cited}</span>
      </div>
      ${rows}`;
    toast(`自测完成：能答率 ${s.answer_rate}`, 4000);
  } catch (error) {
    els.evalPanel.innerHTML = `<div class="empty">自测失败：${esc(error.message)}</div>`;
  } finally {
    els.btnEval.disabled = false;
  }
}

/* ---------- 事件绑定 ---------- */

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  ask();
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
  event.target.style.height = "auto";
  event.target.style.height = Math.min(event.target.scrollHeight, 160) + "px";
});

els.fileInput.addEventListener("change", async () => {
  const files = [...els.fileInput.files];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("kb_id", currentKb);
  els.fileInput.value = "";
  try {
    const { saved } = await api("/api/upload", { method: "POST", body: form });
    toast(saved.length ? `已上传 ${saved.length} 个文档` : "没有可上传的文档");
    await refreshStatus();
  } catch (error) {
    toast(`上传失败：${error.message}`, 4000);
  }
});

els.btnImport.addEventListener("click", async () => {
  setBusy(true);
  try {
    const { imported } = await api("/api/import_dir", {
      method: "POST",
      body: JSON.stringify({ directory: els.dirInput.value.trim(), kb_id: currentKb }),
    });
    toast(imported.length ? `已导入 ${imported.length} 个文档` : "目录没有新文档");
    await refreshStatus();
  } catch (error) {
    toast(`导入失败：${error.message}`, 4000);
  } finally {
    setBusy(false);
  }
});

els.btnRebuild.addEventListener("click", async () => {
  setBusy(true);
  els.btnRebuild.textContent = "提交中…";
  try {
    const job = await api(`/api/rebuild?kb_id=${encodeURIComponent(currentKb)}`,
                          { method: "POST", body: "{}" });
    if (job.state === "done") {
      toast("索引已重建");
    } else {
      toast(`构建已提交后台（${String(job.job_id).slice(0, 8)}…），可继续提问`);
      const final = await waitBuildDone(job.job_id);
      if (final && final.state === "done") {
        toast(`索引就绪：${final.chunks} 块，复用 ${final.reused}，新算 ${final.embedded}`);
      } else if (final && final.state === "failed") {
        toast(`构建失败：${final.error || "未知错误"}`, 6000);
      } else {
        toast("构建仍在进行，可稍后刷新查看", 4000);
      }
    }
  } catch (error) {
    toast(`重建失败：${error.message}`, 4000);
  } finally {
    els.btnRebuild.textContent = "重建索引";
    setBusy(false);
    await refreshStatus();
  }
});

els.btnEval.addEventListener("click", renderEval);

els.questionsFile.addEventListener("change", async () => {
  const file = els.questionsFile.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("kb_id", currentKb);
  els.questionsFile.value = "";
  try {
    const info = await api("/api/questions_upload", { method: "POST", body: form });
    toast(`题目集已替换：${info.count} 题`);
    await refreshQuestions();
  } catch (error) {
    toast(`上传失败：${error.message}`, 4000);
  }
});

els.drawerClose.addEventListener("click", () => els.drawer.classList.add("hidden"));

refreshKbs()
  .then(() => Promise.allSettled([refreshStatus(), refreshUsers()]))
  .catch((error) => toast(`初始化失败：${error.message}`, 5000));
els.input.focus();
