import { UI, CONSENT_TEXT, LANG_LABEL, t } from "./i18n.js";
import { Sensors } from "./sensors.js";
import { Voice } from "./voice.js";

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : document.createTextNode(k));
  return n;
};
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

const S = {
  st: null, convLang: null, ws: null, offset: 0, names: {}, tlFilter: "all",
  camOn: false, micOn: false, paused: false, safety: null, hints: {}, lastCaption: "", mute: false,
};

// ------------------------------------------------------------ language
function navLang() {
  const n = (navigator.language || "en").toLowerCase();
  if (n.startsWith("hi")) return "hi";
  for (const l of ["es", "fr", "de"]) if (n.startsWith(l)) return l;
  return "en";
}
function uiLang() {
  const pref = S.st?.profile?.preferred_lang;
  if (pref && pref !== "auto") return pref;
  return S.convLang || S.st?.session_lang || navLang();
}
function applyI18n() {
  const L = uiLang();
  document.documentElement.lang = L === "hi-Latn" ? "hi" : L;
  $$("[data-i18n]").forEach(n => { n.textContent = t(L, n.dataset.i18n); });
  $$("[data-i18n-ph]").forEach(n => { n.placeholder = t(L, n.dataset.i18nPh); });
  $("#micBtn").title = t(L, "mic"); $("#camBtn").title = t(L, "cam");
  $("#pauseBtn").textContent = t(L, S.paused ? "privacy_resume" : "privacy_pause");
  renderState(); fillSttModes();
  api(`/api/names/${L}`).then(n => { S.names = n; }).catch(() => {});
}
function fillLangSelect(sel, withAuto = true) {
  sel.innerHTML = "";
  if (withAuto) sel.append(el("option", { value: "auto" }, t(uiLang(), "auto")));
  for (const [k, v] of Object.entries(LANG_LABEL)) sel.append(el("option", { value: k }, v));
}

// --------------------------------------------------------------- init
async function init() {
  S.st = await api("/api/state");
  fillLangSelect($("#langSelect")); $("#langSelect").value = S.st.profile.preferred_lang;
  fillLangSelect($("#prefLang")); fillLangSelect($("#obLang"));
  applyI18n();
  bind();
  connect();
  if (!S.st.onboarded) showOnboarding();
  renderState();
}

function bind() {
  $$("#tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#textForm").addEventListener("submit", e => {
    e.preventDefault();
    const v = $("#textInput").value.trim(); if (!v) return;
    $("#textInput").value = "";
    renderUser(v, "text"); send({ type: "utterance", text: v, modality: "text" });
  });
  $("#qOk").onclick = () => send({ type: "ui", action: "im_ok" });
  $("#qHelp").onclick = () => send({ type: "ui", action: "help_now" });
  $("#micBtn").onclick = () => S.micOn ? stopMic() : startMic();
  $("#camBtn").onclick = () => S.camOn ? stopCam() : startCam();
  $("#pauseBtn").onclick = togglePause;
  $("#showVideo").onchange = e => $("#videoBox").classList.toggle("hide-video", !e.target.checked);
  $("#langSelect").onchange = async e => { await saveProfile({ preferred_lang: e.target.value }); };
  $("#sttMode").onchange = async () => { if (S.micOn) { stopMic(); await startMic(); } };
  $("#drawerClose").onclick = () => { $("#drawer").hidden = true; };
  $("#ovOk").onclick = () => overlayAction("ok");
  $("#ovHelp").onclick = () => overlayAction("help");
  // settings
  $("#prefLang").onchange = e => saveProfile({ preferred_lang: e.target.value });
  $("#requireWake").onchange = e => saveProfile({ require_wake_word: e.target.checked });
  $("#llmEnabled").onchange = e => saveProfile({ llm_enabled: e.target.checked });
  $("#deleteAllBtn").onclick = deleteAll;
  $("#addContact").onclick = () => addContactRow({ name: "", channel: "webhook", address: "", lang: uiLang() });
  $("#saveEsc").onclick = saveEscalation;
  $("#testEsc").onclick = testEscalation;
  $("#savePolicy").onclick = savePolicy;
  $("#verifyAudit").onclick = verifyAudit;
  setInterval(tickOverlay, 250);
}

async function saveProfile(p) {
  S.st.profile = await api("/api/profile", { method: "POST", body: p });
  $("#langSelect").value = S.st.profile.preferred_lang; $("#prefLang").value = S.st.profile.preferred_lang;
  applyI18n(); S.voice?.setLang(uiLang());
}

function showTab(name) {
  $$("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "timeline") renderTimeline();
  if (name === "privacy") renderPrivacy();
  if (name === "emergency") renderEmergency();
  if (name === "system") renderSystem();
}

// ------------------------------------------------------------ realtime
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  S.ws = ws;
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === "response") onResponse(m);
    else if (m.type === "status") onStatus(m);
    else if (m.type === "observation") onObservation(m.observation);
    else if (m.type === "escalation") onEscalation(m.result);
  };
  ws.onclose = () => setTimeout(connect, 1500);
}
function send(obj) { if (S.ws?.readyState === 1) S.ws.send(JSON.stringify(obj)); }

// --------------------------------------------------------- conversation
function renderUser(text, modality) {
  const box = $("#convo");
  box.append(el("div", { class: "msg user" }, text, el("div", { class: "meta" }, modality === "speech" ? "🎙" : "⌨", time())));
  box.scrollTop = box.scrollHeight;
}
function time(ts) { return new Date(ts ? ts * 1000 : Date.now()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

function onResponse(m) {
  const r = m.response;
  if (r.lang && (S.st.profile.preferred_lang === "auto") && r.lang !== S.convLang) { S.convLang = r.lang; applyI18n(); S.voice?.setLang(r.lang); }
  const L = r.lang || uiLang();
  const wrap = el("div", { class: "msg baymax" });
  for (const s of r.statements) {
    const chipLabel = t(L, `prov_${s.provenance}`);
    const conf = s.confidence != null ? ` · ${Math.round(s.confidence * 100)}%` : "";
    const node = el("div", { class: `stmt p-${s.provenance}` }, s.text);
    if (chipLabel) {
      node.append(el("div", {},
        el("button", { class: `chip ${s.provenance}`, onclick: () => openRef(s, L) }, chipLabel + conf)));
    }
    wrap.append(node);
  }
  if (r.statements.length) {
    wrap.append(el("div", { class: "meta" }, time(r.ts),
      m.trace_id ? el("button", { class: "why", onclick: () => openTrace(m.trace_id, L) }, t(L, "why")) : null));
    $("#convo").append(wrap); $("#convo").scrollTop = $("#convo").scrollHeight;
    const spoken = r.statements.map(s => s.text).join(" ");
    S.lastCaption = spoken; $("#caption").textContent = spoken;
    if (!S.mute) S.voice?.speak(spoken, L) ?? speakFallback(spoken, L);
  }
  const h = r.ui_hints || {};
  if (h.alarm) (S.voice || alarmOnly()).alarm(h.alarm);
  if (h.privacy) handlePrivacyHint(h.privacy);
  S.hints = h;
  setSafety(r.safety, h);
}
let _alarmVoice = null;
function alarmOnly() { return (_alarmVoice ||= new Voice({})); }
function speakFallback(text, L) { alarmOnly().speak(text, L); }

async function openRef(s, L) {
  const body = $("#drawerBody"); body.innerHTML = "";
  body.append(el("h3", {}, t(L, `prov_${s.provenance}`) || s.provenance));
  body.append(el("p", {}, s.text));
  const kv = el("dl", { class: "kv" }, el("dt", {}, "ref"), el("dd", {}, s.ref || "—"));
  if (s.model_id) kv.append(el("dt", {}, "model"), el("dd", {}, s.model_id));
  if (s.confidence != null) kv.append(el("dt", {}, t(L, "confidence")), el("dd", {}, `${Math.round(s.confidence * 100)}%`));
  body.append(kv);
  $("#drawer").hidden = false;
  if (!s.ref) return;
  try {
    const r = await api(`/api/ref?ref=${encodeURIComponent(s.ref)}`);
    const d = el("dl", { class: "kv" });
    if (r.type === "knowledge") {
      d.append(el("dt", {}, "id"), el("dd", {}, r.id), el("dt", {}, t(L, "source")), el("dd", {}, r.source),
        el("dt", {}, t(L, "review_status")), el("dd", {}, r.review_status));
    } else if (r.type === "observation") {
      d.append(el("dt", {}, "model"), el("dd", {}, r.model_id), el("dt", {}, t(L, "confidence")), el("dd", {}, `${Math.round(r.confidence * 100)}%`),
        el("dt", {}, "flags"), el("dd", {}, (r.failure_flags || []).join(", ") || "—"));
      body.append(el("pre", {}, JSON.stringify(r.evidence, null, 2)));
    } else if (r.type === "event") {
      d.append(el("dt", {}, "kind"), el("dd", {}, r.kind), el("dt", {}, "provenance"), el("dd", {}, r.provenance), el("dt", {}, "summary"), el("dd", {}, r.summary));
    } else {
      body.append(el("pre", {}, JSON.stringify(r, null, 2)));
    }
    body.insertBefore(d, body.children[3] || null);
  } catch (e) { body.append(el("p", { class: "status-text" }, String(e.message))); }
}

async function openTrace(id, L) {
  const tr = await api(`/api/trace/${id}`);
  const body = $("#drawerBody"); body.innerHTML = "";
  body.append(el("h3", {}, t(L, "why")));
  const kv = el("dl", { class: "kv" });
  if (tr.nlu) {
    kv.append(el("dt", {}, "intent"), el("dd", {}, `${tr.nlu.intent} (${Math.round(tr.nlu.intent_confidence * 100)}%)`),
      el("dt", {}, "language"), el("dd", {}, `${tr.nlu.language.lang} (${Math.round(tr.nlu.language.confidence * 100)}%)`),
      el("dt", {}, "concepts"), el("dd", {}, tr.nlu.concepts.map(c => `${c.concept}${c.negated ? " ¬" : ""}${c.hypothetical ? " ?" : ""}${c.past ? " (past)" : ""}${c.subject === "other" ? " (other)" : ""}`).join(", ") || "—"));
    if (tr.nlu.pain_score != null) kv.append(el("dt", {}, "pain"), el("dd", {}, `${tr.nlu.pain_score}/10`));
    if (tr.nlu.notes.length) kv.append(el("dt", {}, "notes"), el("dd", {}, tr.nlu.notes.join(", ")));
  }
  kv.append(el("dt", {}, "safety rules"), el("dd", {}, tr.transitions.map(x => `${x.rule}: ${x.before}→${x.after}`).join("; ") || "—"));
  kv.append(el("dt", {}, "knowledge"), el("dd", {}, tr.knowledge.map(k => k.entry_id).join(", ") || "—"));
  if (tr.llm) kv.append(el("dt", {}, "LLM"), el("dd", {}, `${tr.llm.backend}: ${tr.llm.accepted ? "accepted" : "rejected"} ${tr.llm.violations.map(v => v.code).join(",")}`));
  body.append(kv);
  body.append(el("pre", {}, JSON.stringify({ input: tr.input, safety_inputs: tr.safety_inputs }, null, 2)));
  $("#drawer").hidden = false;
}

// -------------------------------------------------------------- safety
function renderState() {
  const st = S.safety?.state || "monitoring", L = uiLang();
  $("#statePill").dataset.state = st; $("#stateText").textContent = t(L, `state_${st}`);
  const mood = st === "escalation_countdown" || st === "escalated" ? "alert" : (st === "check_in" || st === "assisting") ? "concerned" : null;
  if (mood) $("#faceWrap").dataset.mood = mood;
  else if (["alert", "concerned"].includes($("#faceWrap").dataset.mood)) $("#faceWrap").dataset.mood = S.micOn ? "listening" : "idle";
}

function setSafety(ctx, hints = {}) {
  if (!ctx) return;
  S.safety = ctx; renderState();
  const ov = $("#safetyOverlay"), L = uiLang(), st = ctx.state;
  if (!["check_in", "escalation_countdown", "escalated"].includes(st)) { ov.hidden = true; return; }
  ov.hidden = false; ov.dataset.state = st;
  $("#ovTitle").textContent = t(L, `state_${st}`);
  $("#ovText").textContent = S.lastCaption;
  const num = hints.emergency_number || S.st.escalation.emergency_number;
  $("#ovNumber").textContent = st === "escalated" ? num : "";
  $("#ovOk").textContent = t(L, st === "check_in" ? "im_ok" : st === "escalated" ? "resolved" : "cancel_safe");
  $("#ovHelp").textContent = st === "check_in" ? t(L, "need_help") : st === "escalated" ? `${t(L, "call_now")} ${num}` : t(L, "get_help_now");
  $("#ovTest").textContent = (hints.dry_run ?? S.st.escalation.dry_run) ? t(L, "test_mode") : "";
  $("#countdown").style.display = st === "escalated" ? "none" : "block";
}

function tickOverlay() {
  const c = S.safety; if (!c || !c.timer_deadline || $("#safetyOverlay").hidden) return;
  const now = Date.now() / 1000 + S.offset;
  const total = c.timer_deadline - c.since, left = Math.max(0, c.timer_deadline - now);
  $("#cdNum").textContent = Math.ceil(left);
  $("#cdProg").style.strokeDashoffset = String(327 * (1 - left / Math.max(total, 1)));
}

function overlayAction(which) {
  const st = S.safety?.state;
  if (st === "check_in") send({ type: "ui", action: which === "ok" ? "im_ok" : "help_now" });
  else if (st === "escalation_countdown") send({ type: "ui", action: which === "ok" ? "cancel" : "confirm" });
  else if (st === "escalated") {
    if (which === "ok") send({ type: "ui", action: "resolve" });
    else window.location.href = `tel:${S.hints.emergency_number || S.st.escalation.emergency_number}`;
  }
}

function onStatus(m) {
  S.offset = m.server_time - Date.now() / 1000;
  const c = m.safety;
  if (!S.safety || c.state !== S.safety.state || c.timer_deadline !== S.safety.timer_deadline) setSafety(c, S.hints);
  const p = m.perception, L = uiLang();
  if (S.camOn) {
    $("#rActivity").textContent = t(L, `act_${p.activity}`) + (p.activity_confidence ? ` · ${Math.round(p.activity_confidence * 100)}%` : "");
    $("#rVis").textContent = p.visibility != null ? `${Math.round(p.visibility * 100)}%` : "—";
    $("#rPersons").textContent = p.persons ?? "—";
  }
}
function onObservation(o) {
  if (o.kind === "fall") $("#rFall").textContent = `${Math.round(o.confidence * 100)}%${o.failure_flags.length ? " ⚠ " + o.failure_flags.join(", ") : ""}`;
}
function onEscalation(r) {
  const note = el("div", { class: "msg alert-note" }, `⚑ ${r.level} → ${r.status}${r.results.length ? ` (${r.results.filter(x => x.ok).length}/${r.results.length})` : ""}`);
  $("#convo").append(note); $("#convo").scrollTop = $("#convo").scrollHeight;
}

// -------------------------------------------------------------- sensors
async function ensureConsent(scope) {
  if (S.st.consents[scope]) return true;
  const [title, desc] = CONSENT_TEXT[uiLang()][scope];
  if (!confirm(`${title}\n\n${desc}`)) return false;
  S.st.consents = await api("/api/consent", { method: "POST", body: { scope, granted: true } });
  return true;
}

async function startCam() {
  if (!(await ensureConsent("camera_processing"))) return;
  S.sensors ||= new Sensors({ video: $("#video"), canvas: $("#overlay"),
    onSamples: samples => send({ type: "body", samples }),
    onStatus: s => { $("#modeBadge").textContent = s.mode === "pose" ? t(uiLang(), "pose_mode") : s.mode === "motion" ? t(uiLang(), "motion_mode") : s.mode === "loading" ? "…" : ""; } });
  try { await S.sensors.start(); S.camOn = true; $("#videoBox").classList.add("live"); $("#camBtn").setAttribute("aria-pressed", "true"); }
  catch (e) { alert(e.message); }
}
function stopCam() { S.sensors?.stop(); S.camOn = false; $("#videoBox").classList.remove("live"); $("#camBtn").setAttribute("aria-pressed", "false"); $("#rActivity").textContent = "—"; }

function fillSttModes() {
  const sel = $("#sttMode"), L = uiLang(), cur = sel.value;
  sel.innerHTML = "";
  if (S.st?.stt?.local_available) sel.append(el("option", { value: "local" }, t(L, "stt_local")));
  if (window.SpeechRecognition || window.webkitSpeechRecognition) sel.append(el("option", { value: "browser" }, t(L, "stt_browser")));
  sel.append(el("option", { value: "off" }, t(L, "stt_none")));
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
}

async function startMic() {
  if (!(await ensureConsent("microphone_processing"))) return;
  const mode = $("#sttMode").value;
  if (mode === "browser" && !(await ensureConsent("cloud_speech_recognition"))) return;
  S.voice?.stop();
  S.voice = new Voice({
    onTranscript: tr => { renderUser(tr.text, "speech"); send({ type: "utterance", modality: "speech", ...tr }); },
    onLevels: levels => { send({ type: "audio", levels }); const db = levels[levels.length - 1].rms_db; $("#audioBar").style.width = `${Math.max(0, Math.min(100, (db + 70) * 1.6))}%`; },
    onSpeechStart: () => { $("#faceWrap").dataset.mood = "listening"; },
    onTTS: on => { if (!["alert", "concerned"].includes($("#faceWrap").dataset.mood)) $("#faceWrap").dataset.mood = on ? "speaking" : (S.micOn ? "listening" : "idle"); },
  });
  try {
    await S.voice.start(mode, uiLang());
    S.micOn = true; $("#micBtn").setAttribute("aria-pressed", "true"); $("#micBtn").classList.add("live");
    $("#faceWrap").dataset.mood = "listening";
  } catch (e) { alert(e.message); }
}
function stopMic() { S.voice?.stop(); S.micOn = false; $("#micBtn").setAttribute("aria-pressed", "false"); $("#micBtn").classList.remove("live"); $("#faceWrap").dataset.mood = "idle"; $("#audioBar").style.width = "0"; }

let _resume = { cam: false, mic: false };
function togglePause() {
  if (!S.paused) { _resume = { cam: S.camOn, mic: S.micOn }; stopCam(); stopMic(); S.paused = true; $("#pauseBtn").classList.add("on"); }
  else { S.paused = false; $("#pauseBtn").classList.remove("on"); if (_resume.cam) startCam(); if (_resume.mic) startMic(); }
  $("#pauseBtn").textContent = t(uiLang(), S.paused ? "privacy_resume" : "privacy_pause");
}
function handlePrivacyHint(action) {
  if (action === "pause_camera") stopCam();
  else if (action === "pause_mic") stopMic();
  else if (action === "privacy_mode" && !S.paused) togglePause();
  else if (action === "resume" && S.paused) togglePause();
}

// -------------------------------------------------------------- timeline
const KIND_ICON = { pain: "◉", symptom: "◌", fall_reported: "↘", fall_detected: "↘", measurement: "▤", medication_taken: "✚", emergency: "⚑", posture_alert: "∿", inactivity: "…", check_in: "?", escalation: "⚑", note: "✎" };

async function renderTimeline() {
  const L = uiLang();
  const { events } = await api("/api/timeline?days=365");
  const kinds = ["all", ...new Set(events.map(e => e.kind))];
  const nm = k => k === "all" ? t(L, "filter_all") : (S.names[k] || k);
  $("#tlFilters").replaceChildren(...kinds.map(k => el("button", { class: S.tlFilter === k ? "active" : "", onclick: () => { S.tlFilter = k; renderTimeline(); } }, nm(k))));
  // charts
  const series = {};
  for (const e of events) {
    if (e.kind === "measurement" && e.measurement) (series[e.measurement.type] ||= []).push([e.ts, e.measurement.value, e.measurement]);
    if (e.kind === "pain" && e.pain_score != null) (series.pain_score ||= []).push([e.ts, e.pain_score, { unit: "/10", value: e.pain_score }]);
  }
  $("#tlCharts").replaceChildren(...Object.entries(series).map(([k, pts]) => chart(S.names[k] || k, pts.reverse())));
  const list = events.filter(e => S.tlFilter === "all" || e.kind === S.tlFilter);
  const ol = $("#timeline"); ol.innerHTML = "";
  if (!list.length) { ol.append(el("li", { class: "empty" }, t(L, "timeline_empty"))); return; }
  let day = "";
  for (const e of list) {
    const d = new Date(e.ts * 1000), ds = d.toLocaleDateString(L === "hi-Latn" ? "en-IN" : L, { weekday: "long", day: "numeric", month: "long" });
    if (ds !== day) { day = ds; ol.append(el("li", {}, el("div", { class: "day" }, ds))); }
    ol.append(el("li", { class: `sev-${e.severity}` },
      el("div", { class: "time" }, d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })),
      el("div", { class: "node" }),
      el("div", { class: "body" }, el("strong", {}, `${KIND_ICON[e.kind] || "•"} ${describe(e, L)}`),
        el("small", {}, `${e.provenance === "system" ? t(L, "prov_system_record") : t(L, `prov_${e.provenance}`)}${e.model_id ? " · " + e.model_id : ""}${e.confidence != null && e.provenance === "model_inference" ? ` · ${Math.round(e.confidence * 100)}%` : ""}`)),
      el("button", { class: "why", onclick: async () => { await api(`/api/events/${e.id}`, { method: "DELETE" }); renderTimeline(); } }, t(L, "delete"))));
  }
}
function describe(e, L) {
  const n = k => S.names[k] || k;
  if (e.kind === "measurement" && e.measurement) {
    const m = e.measurement; return `${n(m.type)}: ${m.value2 != null ? `${m.value}/${m.value2}` : m.value} ${m.unit}`;
  }
  if (e.kind === "pain") return `${n("pain")}${e.pain_score != null ? ` ${e.pain_score}/10` : ""}${e.body_regions.length ? " · " + e.body_regions.join(", ") : ""}`;
  if (e.kind === "symptom") return e.concepts.map(c => n("c:" + c)).join(", ");
  if (e.kind === "medication_taken") return `${n("medication_taken")}: ${e.medication}`;
  return n(e.kind) + (e.concepts.length ? ": " + e.concepts.map(c => n("c:" + c)).join(", ") : "");
}
function chart(title, pts) {
  const W = 220, H = 48, vals = pts.map(p => p[1]);
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const x = i => pts.length === 1 ? W / 2 : (i / (pts.length - 1)) * (W - 8) + 4, y = v => H - 6 - ((v - lo) / span) * (H - 12);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1][2];
  const svg = `<svg viewBox="0 0 ${W} ${H}"><path d="${path}" fill="none" stroke="#3570c9" stroke-width="2.5" stroke-linejoin="round"/>${pts.map((p, i) => `<circle cx="${x(i)}" cy="${y(p[1])}" r="2.5" fill="#3570c9"/>`).join("")}</svg>`;
  const c = el("div", { class: "chart" }, el("h4", {}, title), el("div", { class: "last" }, last.value2 != null ? `${last.value}/${last.value2} ${last.unit}` : `${last.value}${last.unit?.startsWith("/") ? "" : " "}${last.unit || ""}`));
  c.insertAdjacentHTML("beforeend", svg);
  return c;
}

// -------------------------------------------------------------- privacy
async function renderPrivacy() {
  S.st = await api("/api/state");
  const L = uiLang();
  $("#consentList").replaceChildren(...Object.keys(S.st.consents).map(scope => consentRow(scope, L)));
  $("#prefLang").value = S.st.profile.preferred_lang;
  $("#requireWake").checked = S.st.profile.require_wake_word;
  $("#llmEnabled").checked = S.st.profile.llm_enabled;
  $("#llmInfo").textContent = S.st.llm.backend ? `(${S.st.llm.backend}${S.st.llm.cloud ? ", cloud" : ", local"})` : "(none configured)";
  const { memories } = await api("/api/memories");
  $("#memoryList").replaceChildren(...memories.map(m => el("li", {}, el("span", {}, m.value, " ", el("small", {}, `(${m.kind})`)),
    el("button", { class: "why", onclick: async () => { await api(`/api/memories/${m.id}`, { method: "DELETE" }); renderPrivacy(); } }, t(L, "forget")))));
}
function consentRow(scope, L) {
  const [title, desc] = (CONSENT_TEXT[L] || CONSENT_TEXT.en)[scope] || [scope, ""];
  const box = el("input", { type: "checkbox", class: "toggle" });
  box.checked = !!S.st.consents[scope];
  box.onchange = async () => { S.st.consents = await api("/api/consent", { method: "POST", body: { scope, granted: box.checked } }); };
  return el("label", { class: "consent-item" }, el("div", {}, el("b", {}, title), el("small", {}, desc)), box);
}
async function deleteAll() {
  const v = prompt(t(uiLang(), "delete_confirm"));
  if (v !== "DELETE") return;
  await api("/api/delete_all", { method: "POST", body: { confirm: "DELETE" } });
  $("#convo").innerHTML = ""; renderPrivacy();
}

// ------------------------------------------------------------ emergency
async function renderEmergency() {
  S.st = await api("/api/state");
  const e = S.st.escalation, p = S.st.policy;
  $("#contactList").innerHTML = "";
  e.contacts.forEach(addContactRow);
  $("#userName").value = e.user_display_name; $("#emNumber").value = e.emergency_number; $("#dryRun").checked = e.dry_run;
  $("#pCheckin").value = p.checkin_timeout_s; $("#pCountdown").value = p.countdown_s; $("#pFallCountdown").value = p.fall_countdown_s;
  $("#pHighPain").value = p.high_pain_threshold; $("#pInactivity").value = S.st.profile.inactivity_minutes;
}
function addContactRow(c) {
  const L = uiLang();
  const langSel = el("select", { class: "c-lang" }); for (const [k, v] of Object.entries(LANG_LABEL)) langSel.append(el("option", { value: k }, v)); langSel.value = c.lang || "en";
  const ch = el("select", { class: "c-channel" }, ...["webhook", "email", "console"].map(x => el("option", { value: x }, x))); ch.value = c.channel;
  const row = el("div", { class: "contact-row", "data-id": c.id || "" },
    el("input", { class: "c-name", placeholder: t(L, "name"), value: c.name || "" }), ch,
    el("input", { class: "c-address", placeholder: t(L, "address"), value: c.address || "" }), langSel,
    el("button", { onclick: () => row.remove(), title: t(L, "delete") }, "×"));
  $("#contactList").append(row);
}
async function saveEscalation() {
  const contacts = $$(".contact-row").map(r => ({ ...(r.dataset.id ? { id: r.dataset.id } : {}), name: r.querySelector(".c-name").value.trim(),
    channel: r.querySelector(".c-channel").value, address: r.querySelector(".c-address").value.trim(), lang: r.querySelector(".c-lang").value })).filter(c => c.name && c.address);
  try {
    S.st.escalation = await api("/api/escalation", { method: "POST", body: { contacts, emergency_number: $("#emNumber").value.trim() || "112",
      dry_run: $("#dryRun").checked, user_display_name: $("#userName").value.trim() || "Baymax user" } });
    $("#escStatus").textContent = t(uiLang(), "saved"); renderEmergency();
  } catch (e) { $("#escStatus").textContent = e.message; }
}
async function testEscalation() {
  try { const r = await api("/api/escalation/test", { method: "POST" }); $("#escStatus").textContent = `${r.status}: ${r.results.map(x => `${x.channel} ${x.ok ? "✓" : "✗ " + x.detail}`).join(", ")}`; }
  catch (e) { $("#escStatus").textContent = e.message; }
}
async function savePolicy() {
  await api("/api/policy", { method: "POST", body: { checkin_timeout_s: +$("#pCheckin").value, countdown_s: +$("#pCountdown").value,
    fall_countdown_s: +$("#pFallCountdown").value, high_pain_threshold: +$("#pHighPain").value } });
  await saveProfile({ inactivity_minutes: +$("#pInactivity").value });
  $("#policyStatus").textContent = t(uiLang(), "saved");
}

// --------------------------------------------------------------- system
async function renderSystem() {
  const L = uiLang();
  const { models } = await api("/api/models");
  $("#modelList").replaceChildren(...models.map(m => el("details", { class: "model" },
    el("summary", {}, m.name, el("span", {}, m.model_id, m.may_trigger_safety ? " · " : "", m.may_trigger_safety ? el("b", { class: "tag" }, "safety input") : null)),
    el("p", {}, el("b", {}, "Input: "), m.input), el("p", {}, el("b", {}, "Output: "), m.output), el("p", {}, el("b", {}, "Confidence: "), m.confidence),
    el("p", {}, el("b", {}, "Failure modes:")), el("ul", {}, ...m.failure_modes.map(f => el("li", {}, f))),
    el("p", {}, el("b", {}, "Mitigations:")), el("ul", {}, ...m.mitigations.map(f => el("li", {}, f))))));
  try {
    const rep = await api("/api/eval/latest");
    const g = el("div", { class: "gates" });
    for (const gate of rep.gates) g.append(el("span", {}, gate.name), el("span", {}, `${gate.value} ${gate.op} ${gate.threshold}`), el("span", { class: gate.passed ? "pass" : "fail" }, gate.passed ? "PASS" : "FAIL"));
    $("#evalBox").replaceChildren(el("p", { class: "status-text" }, `${new Date(rep.generated_at * 1000).toLocaleString()} · ${rep.summary.cases} cases · ${rep.passed ? "ALL GATES PASS" : "GATES FAILING"}`), g);
  } catch { $("#evalBox").replaceChildren(el("p", { class: "status-text" }, t(L, "run_eval_hint"))); }
  const { entries } = await api("/api/audit?limit=150");
  $("#auditRows").replaceChildren(...entries.map(a => el("tr", {}, el("td", {}, new Date(a.ts * 1000).toLocaleTimeString()), el("td", { class: "a" }, `${a.actor}:${a.action}`),
    el("td", { class: "d" }, JSON.stringify(a.detail)))));
}
async function verifyAudit() {
  const r = await api("/api/audit/verify"), L = uiLang();
  $("#auditStatus").textContent = r.ok ? `✓ ${t(L, "audit_ok")} (${r.entries})` : `✗ ${t(L, "audit_broken")} ${r.broken_at}`;
}

// ------------------------------------------------------------ onboarding
function showOnboarding() {
  const m = $("#onboarding"); m.hidden = false;
  $("#obLang").value = S.st.profile.preferred_lang;
  const render = () => {
    const L = $("#obLang").value === "auto" ? navLang() : $("#obLang").value;
    S.convLang = L; applyI18n();
    $("#obConsents").replaceChildren(...["camera_processing", "microphone_processing", "store_health_events", "long_term_memory", "emergency_contact_sharing"].map(s => consentRow(s, L)));
  };
  $("#obLang").onchange = render; render();
  $("#obStart").onclick = async () => {
    await saveProfile({ preferred_lang: $("#obLang").value });
    await api("/api/onboarded", { method: "POST" });
    S.st = await api("/api/state"); m.hidden = true;
  };
}

init();
