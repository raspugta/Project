// Microphone, speech recognition, speech synthesis and alarm sounds.
//
// STT modes
//   local   : energy-VAD segments -> 16 kHz WAV -> POST /api/stt (faster-whisper on this machine)
//   browser : Web Speech API (may use the browser vendor's cloud; needs explicit consent)
//   off     : keyboard only
// Sound levels (RMS dBFS, no audio) are streamed for the impact detector.

import { SPEECH_TAG } from "./i18n.js";

export class Voice {
  constructor({ onTranscript, onLevels, onSpeechStart, onTTS }) {
    this.onTranscript = onTranscript; this.onLevels = onLevels; this.onSpeechStart = onSpeechStart; this.onTTS = onTTS;
    this.mode = "off"; this.lang = "en"; this.active = false;
    this.ctx = null; this.stream = null; this.levels = []; this.t0 = performance.now();
    this.noiseDb = -60; this.inSpeech = false; this.silenceMs = 0; this.speechMs = 0;
    this.pre = []; this.seg = []; this.sampleRate = 48000;
    this.rec = null; this.speaking = false; this.lastSpoken = { text: "", end: 0 };
    this.alarmTimer = null; this.voices = [];
    if ("speechSynthesis" in window) {
      const load = () => { this.voices = speechSynthesis.getVoices(); };
      load(); speechSynthesis.onvoiceschanged = load;
    }
  }

  async start(mode, lang) {
    this.mode = mode; this.lang = lang;
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this.ctx.sampleRate;
    const src = this.ctx.createMediaStreamSource(this.stream);
    const proc = this.ctx.createScriptProcessor(2048, 1, 1);
    src.connect(proc); proc.connect(this.ctx.destination);
    proc.onaudioprocess = e => this._audio(e.inputBuffer.getChannelData(0));
    this.proc = proc; this.active = true;
    if (mode === "browser") this._startBrowserSTT();
  }

  stop() {
    this.active = false;
    this.proc?.disconnect(); this.ctx?.close(); this.ctx = null;
    this.stream?.getTracks().forEach(t => t.stop()); this.stream = null;
    if (this.rec) { this.rec.onend = null; this.rec.stop(); this.rec = null; }
  }

  setLang(lang) {
    if (lang === this.lang) return;
    this.lang = lang;
    if (this.rec) { this.rec.lang = SPEECH_TAG[lang] || "en-US"; try { this.rec.stop(); } catch {} }
  }

  // ------------------------------------------------------------ audio
  _audio(buf) {
    let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    const db = 10 * Math.log10(s / buf.length + 1e-10);
    const dur = buf.length / this.sampleRate * 1000;
    const t = (performance.now() - this.t0) / 1000;
    this.levels.push({ t, rms_db: Math.max(-100, db) });
    if (this.levels.length >= 5) { this.onLevels?.(this.levels); this.levels = []; }
    if (this.mode !== "local") {
      if (db > this.noiseDb + 15 && db > -45) this._bargeIn();
      return;
    }
    // energy VAD with adaptive noise floor
    if (!this.inSpeech) this.noiseDb = this.noiseDb * 0.98 + Math.min(db, -20) * 0.02;
    const threshold = Math.max(this.noiseDb + (this.speaking ? 20 : 12), -50);
    const copy = new Float32Array(buf);
    if (!this.inSpeech) {
      this.pre.push(copy); if (this.pre.length > 8) this.pre.shift();
      if (db > threshold) { this.inSpeech = true; this.seg = [...this.pre]; this.silenceMs = 0; this.speechMs = 0; this._bargeIn(); }
    } else {
      this.seg.push(copy); this.speechMs += dur;
      this.silenceMs = db < threshold - 6 ? this.silenceMs + dur : 0;
      if (this.silenceMs > 700 || this.speechMs > 12000) {
        this.inSpeech = false;
        const segment = this.seg; this.seg = []; this.pre = [];
        if (this.speechMs > 300) this._sendSegment(segment);
      }
    }
  }

  _bargeIn() {
    if (this.speaking) { speechSynthesis.cancel(); this.speaking = false; this.onTTS?.(false); }
    this.onSpeechStart?.();
  }

  async _sendSegment(chunks) {
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const pcm = new Float32Array(n); let o = 0; for (const c of chunks) { pcm.set(c, o); o += c.length; }
    const wav = encodeWav(downsample(pcm, this.sampleRate, 16000), 16000);
    try {
      const r = await fetch(`/api/stt?lang=${encodeURIComponent(this.lang)}`, { method: "POST", body: wav, headers: { "Content-Type": "audio/wav" } });
      if (!r.ok) return;
      const j = await r.json();
      if (j.text && !this._isEcho(j.text)) this.onTranscript?.({ text: j.text, stt_model_id: j.model_id, stt_confidence: j.confidence, stt_language: j.language });
    } catch (e) { console.warn("stt failed", e); }
  }

  _startBrowserSTT() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) throw new Error("no browser speech recognition");
    const rec = new SR();
    rec.continuous = true; rec.interimResults = false; rec.lang = SPEECH_TAG[this.lang] || "en-US";
    rec.onresult = e => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) {
          const text = r[0].transcript.trim();
          if (text && !this._isEcho(text)) this.onTranscript?.({ text, stt_model_id: "browser_webspeech", stt_confidence: r[0].confidence || null, stt_language: rec.lang });
        }
      }
    };
    rec.onend = () => { if (this.active && this.mode === "browser") setTimeout(() => { try { rec.start(); } catch {} }, 250); };
    rec.onerror = () => {};
    rec.start(); this.rec = rec;
  }

  _isEcho(text) {
    if (!this.lastSpoken.text) return false;
    const recent = this.speaking || performance.now() - this.lastSpoken.end < 1500;
    if (!recent) return false;
    const a = new Set(text.toLowerCase().split(/\s+/)), b = new Set(this.lastSpoken.text.toLowerCase().split(/\s+/));
    let inter = 0; for (const w of a) if (b.has(w)) inter++;
    return inter / Math.max(1, a.size) >= 0.6;
  }

  // -------------------------------------------------------------- TTS
  speak(text, lang) {
    if (!("speechSynthesis" in window) || !text) return;
    const tag = SPEECH_TAG[lang] || "en-US";
    const u = new SpeechSynthesisUtterance(text);
    u.lang = tag; u.rate = 0.95; u.pitch = 1.05;
    const exact = this.voices.filter(v => v.lang?.toLowerCase() === tag.toLowerCase());
    const base = this.voices.filter(v => v.lang?.toLowerCase().startsWith(tag.slice(0, 2).toLowerCase()));
    const pick = arr => arr.find(v => /natural|neural|premium|enhanced|google/i.test(v.name)) || arr[0];
    const voice = pick(exact) || pick(base);
    if (voice) u.voice = voice;
    u.onstart = () => { this.speaking = true; this.onTTS?.(true); };
    u.onend = u.onerror = () => { this.speaking = false; this.lastSpoken.end = performance.now(); this.onTTS?.(false); };
    this.lastSpoken.text = text;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }

  // ------------------------------------------------------------ alarms
  alarm(level) {
    clearInterval(this.alarmTimer); this.alarmTimer = null;
    if (!level || level === "off") return;
    const ac = this.ctx || (this._ac ||= new (window.AudioContext || window.webkitAudioContext)());
    const tone = (f, d, v = 0.15, when = 0) => {
      const o = ac.createOscillator(), g = ac.createGain();
      o.type = "sine"; o.frequency.value = f; o.connect(g); g.connect(ac.destination);
      const t = ac.currentTime + when; g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(v, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d); o.start(t); o.stop(t + d + 0.05);
    };
    if (level === "chime") { tone(660, 0.35); tone(880, 0.5, 0.15, 0.25); return; }
    const pattern = level === "loud" ? () => { tone(880, 0.3, 0.3); tone(660, 0.3, 0.3, 0.35); } : () => tone(740, 0.25, 0.12);
    pattern();
    this.alarmTimer = setInterval(pattern, level === "loud" ? 800 : 2000);
  }
}

function downsample(buf, from, to) {
  if (from === to) return buf;
  const ratio = from / to, n = Math.floor(buf.length / ratio), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.floor(i * ratio), e = Math.min(buf.length, Math.floor((i + 1) * ratio));
    let acc = 0; for (let j = s; j < e; j++) acc += buf[j]; out[i] = acc / Math.max(1, e - s);
  }
  return out;
}

function encodeWav(samples, rate) {
  const b = new ArrayBuffer(44 + samples.length * 2), v = new DataView(b);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return new Blob([b], { type: "audio/wav" });
}
