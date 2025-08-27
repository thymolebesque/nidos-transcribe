const $ = (q) => document.querySelector(q);

function fmtTime(t) {
  if (t === undefined || t === null) return "";
  const s = Number(t);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(2).padStart(5, "0")}`;
}

/* ---------------- Microphone recorder → WAV (mono) ---------------- */

class MicRecorder {
  constructor() {
    this.audioCtx = null;
    this.stream = null;
    this.source = null;
    this.proc = null;
    this.buffers = [];
    this.sampleRate = 0;
    this.recording = false;
  }

  async start() {
    if (this.recording) return;
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this.audioCtx.sampleRate; // typically 48000
    this.source = this.audioCtx.createMediaStreamSource(this.stream);

    // ScriptProcessorNode is deprecated but widely supported; simple for our use case.
    const bufferSize = 4096;
    this.proc = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);
    this.source.connect(this.proc);
    this.proc.connect(this.audioCtx.destination);

    this.proc.onaudioprocess = (e) => {
      if (!this.recording) return;
      const input = e.inputBuffer.getChannelData(0); // mono
      this.buffers.push(new Float32Array(input));    // copy
    };

    this.recording = true;
  }

  async stop() {
    if (!this.recording) return null;
    this.recording = false;

    if (this.proc) this.proc.disconnect();
    if (this.source) this.source.disconnect();
    if (this.audioCtx) await this.audioCtx.close();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());

    // Concatenate buffers
    let length = this.buffers.reduce((a, b) => a + b.length, 0);
    let pcm = new Float32Array(length);
    let offset = 0;
    for (const b of this.buffers) {
      pcm.set(b, offset);
      offset += b.length;
    }
    this.buffers = [];

    // Encode WAV (PCM16, mono) at native sampleRate (server will resample to 16k)
    const wav = encodeWavMonoPCM16(pcm, this.sampleRate);
    return new Blob([wav], { type: "audio/wav" });
  }
}

function encodeWavMonoPCM16(float32, sampleRate) {
  // Convert float [-1,1] → int16
  const len = float32.length;
  const buffer = new ArrayBuffer(44 + len * 2);
  const view = new DataView(buffer);

  // RIFF/WAVE header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + len * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true);  // format = PCM
  view.setUint16(22, 1, true);  // channels = 1
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * blockAlign)
  view.setUint16(32, 2, true);  // block align (channels * bytesPerSample)
  view.setUint16(34, 16, true); // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, len * 2, true);

  // PCM samples
  let offset = 44;
  for (let i = 0; i < len; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}
function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

/* ---------------- Existing file-upload flows ---------------- */

async function enrollFromFile() {
  const status = $("#enroll-status");
  const f = $("#enroll-wav").files[0];
  if (!f) { status.textContent = "Choose a WAV first."; return; }
  status.textContent = "Uploading...";
  const fd = new FormData();
  fd.append("file", f, f.name);
  try {
    const res = await fetch("/enroll", { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || res.statusText);
    status.textContent = `Enrolled ${j.speaker} (dim=${j.embedding_dim})`;
  } catch (e) { status.textContent = `Error: ${e.message}`; }
}

async function transcribeFromFile() {
  const status = $("#trans-status");
  const chat = $("#chat"); const meta = $("#meta");
  chat.innerHTML = ""; meta.textContent = "";
  const f = $("#trans-wav").files[0];
  if (!f) { status.textContent = "Choose a WAV first."; return; }
  status.textContent = "Processing...";
  const fd = new FormData();
  fd.append("file", f, f.name);
  fd.append("language", $("#language").value);
  fd.append("coach_threshold", $("#thr").value);
  fd.append("use_word_timestamps", $("#wordts").checked ? "true" : "false");
  fd.append("max_speakers", document.getElementById("max_speakers").value);
  await doTranscribe(fd, status, meta, chat);
}

/* ---------------- New mic-record flows ---------------- */

const recEnroll = new MicRecorder();
const recTrans = new MicRecorder();
let enrollBlob = null;
let transBlob = null;

async function startRecEnroll() {
  $("#btn-rec-enroll").disabled = true;
  $("#btn-stop-enroll").disabled = false;
  $("#btn-enroll-rec").disabled = true;
  $("#enroll-status").textContent = "Recording…";
  await recEnroll.start();
}

async function stopRecEnroll() {
  $("#btn-stop-enroll").disabled = true;
  const blob = await recEnroll.stop();
  enrollBlob = blob;
  $("#btn-enroll-rec").disabled = !blob;
  $("#btn-rec-enroll").disabled = false;

  if (blob) {
    $("#enroll-status").textContent = "Recording ready";
    $("#enroll-audio").src = URL.createObjectURL(blob);
  } else {
    $("#enroll-status").textContent = "No audio captured";
  }
}

async function enrollFromRecording() {
  const status = $("#enroll-status");
  if (!enrollBlob) { status.textContent = "No recording yet."; return; }
  status.textContent = "Uploading recording…";
  const fd = new FormData();
  fd.append("file", new File([enrollBlob], "enroll.wav", { type: "audio/wav" }));
  try {
    const res = await fetch("/enroll", { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || res.statusText);
    status.textContent = `Enrolled ${j.speaker} (dim=${j.embedding_dim})`;
  } catch (e) { status.textContent = `Error: ${e.message}`; }
}

async function startRecTrans() {
  $("#btn-rec-trans").disabled = true;
  $("#btn-stop-trans").disabled = false;
  $("#btn-trans-rec").disabled = true;
  $("#trans-status").textContent = "Recording…";
  await recTrans.start();
}

async function stopRecTrans() {
  $("#btn-stop-trans").disabled = true;
  const blob = await recTrans.stop();
  transBlob = blob;
  $("#btn-trans-rec").disabled = !blob;
  $("#btn-rec-trans").disabled = false;

  if (blob) {
    $("#trans-status").textContent = "Recording ready";
    $("#trans-audio").src = URL.createObjectURL(blob);
  } else {
    $("#trans-status").textContent = "No audio captured";
  }
}

async function transcribeFromRecording() {
  const status = $("#trans-status");
  const chat = $("#chat"); const meta = $("#meta");
  chat.innerHTML = ""; meta.textContent = "";
  if (!transBlob) { status.textContent = "No recording yet."; return; }
  status.textContent = "Processing recording…";
  const fd = new FormData();
  fd.append("file", new File([transBlob], "session.wav", { type: "audio/wav" }));
  fd.append("language", $("#language").value);
  fd.append("coach_threshold", $("#thr").value);
  fd.append("use_word_timestamps", $("#wordts").checked ? "true" : "false");
  fd.append("max_speakers", document.getElementById("max_speakers").value);
  await doTranscribe(fd, status, meta, chat);
}

/* ---------------- Shared render logic ---------------- */

async function doTranscribe(formData, status, meta, chat) {
  try {
    const res = await fetch("/transcribe", { method: "POST", body: formData });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || res.statusText);

    meta.innerHTML = `
      <div>Session: <code>${j.session_id}</code></div>
      <div>Language: <strong>${j.language}</strong></div>
      <div>Model: ${j.metrics.model}; Processing ${j.metrics.processing_sec.toFixed(2)}s</div>
      <div>Utterances: ${j.utterances.length}</div>
    `;

    chat.innerHTML = "";
    for (const u of j.utterances) {
      const bubble = document.createElement("div");
      bubble.className = "bubble";

      const chip = document.createElement("span");
      const spk = (u.speaker || "").toUpperCase();
      chip.className = `chip ${spk === "COACH" ? "coach" : (spk === "JONGERE" ? "jongere" : "")}`;
      chip.textContent = spk;

      const metaLine = document.createElement("div");
      metaLine.className = "meta";
      metaLine.appendChild(chip);
      const tspan = document.createElement("span");
      tspan.textContent = `${fmtTime(u.start)} → ${fmtTime(u.end)} (${(u.end-u.start).toFixed(2)}s)`;
      metaLine.appendChild(tspan);

      const text = document.createElement("div");
      text.textContent = u.text;

      bubble.appendChild(metaLine);
      bubble.appendChild(text);

      if (u.words && u.words.length) {
        const wordsDiv = document.createElement("div");
        wordsDiv.className = "words";
        for (const w of u.words) {
          const wspan = document.createElement("span");
          wspan.className = "word";
          wspan.title = `${fmtTime(w.start)}–${fmtTime(w.end)} (${(w.end - w.start).toFixed(2)}s) • ${w.speaker}`;
          wspan.textContent = w.w;
          wordsDiv.appendChild(wspan);
        }
        bubble.appendChild(wordsDiv);
      }

      chat.appendChild(bubble);
    }

    status.textContent = "Done.";
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
}

/* ---------------- Wire up buttons ---------------- */

$("#btn-enroll-file").addEventListener("click", enrollFromFile);
$("#btn-trans-file").addEventListener("click", transcribeFromFile);

$("#btn-rec-enroll").addEventListener("click", startRecEnroll);
$("#btn-stop-enroll").addEventListener("click", stopRecEnroll);
$("#btn-enroll-rec").addEventListener("click", enrollFromRecording);

$("#btn-rec-trans").addEventListener("click", startRecTrans);
$("#btn-stop-trans").addEventListener("click", stopRecTrans);
$("#btn-trans-rec").addEventListener("click", transcribeFromRecording);
