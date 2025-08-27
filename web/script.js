const $ = (q) => document.querySelector(q);

function fmtTime(t) {
  if (t === undefined || t === null) return "";
  const s = Number(t);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(2).padStart(5, "0")}`;
}

async function enroll() {
  const status = $("#enroll-status");
  const f = $("#enroll-wav").files[0];
  if (!f) { status.textContent = "Choose a WAV first."; return; }
  status.textContent = "Uploading...";
  const fd = new FormData();
  fd.append("file", f, f.name);
  // optional: fd.append("speaker_name","COACH")
  try {
    const res = await fetch("/enroll", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(()=>({detail: res.statusText}));
      throw new Error(err.detail || res.statusText);
    }
    const j = await res.json();
    status.textContent = `Enrolled ${j.speaker} (dim=${j.embedding_dim})`;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
}

async function transcribe() {
  const status = $("#trans-status");
  const chat = $("#chat");
  const meta = $("#meta");
  chat.innerHTML = "";
  meta.textContent = "";

  const f = $("#trans-wav").files[0];
  if (!f) { status.textContent = "Choose a WAV first."; return; }
  status.textContent = "Processing...";
  const fd = new FormData();
  fd.append("file", f, f.name);
  fd.append("language", $("#language").value);
  fd.append("coach_threshold", $("#thr").value);
  fd.append("use_word_timestamps", $("#wordts").checked ? "true" : "false");

  try {
    const res = await fetch("/transcribe", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(()=>({detail: res.statusText}));
      throw new Error(err.detail || res.statusText);
    }
    const j = await res.json();

    meta.innerHTML = `
      <div>Session: <code>${j.session_id}</code></div>
      <div>Language: <strong>${j.language}</strong></div>
      <div>Model: ${j.metrics.model}; Processing ${j.metrics.processing_sec.toFixed(2)}s</div>
      <div>Utterances: ${j.utterances.length}</div>
    `;

    // Render chat
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

      $("#chat").appendChild(bubble);
    }

    status.textContent = "Done.";
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
}

$("#btn-enroll").addEventListener("click", enroll);
$("#btn-trans").addEventListener("click", transcribe);
