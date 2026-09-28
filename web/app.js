const $ = (id) => document.getElementById(id);
let after = 0;
let pollTimer = null;
let jobId = null;
let followingLive = true;
const identityKind = {};
const markerEvents = [];
// Global IDs whose bounding boxes are hidden in the live frame.
const hiddenIds = new Set(
  JSON.parse(localStorage.getItem("mct:hiddenIds") || "[]")
);
let lastPeople = [];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Video/file names come from uploads; never put them into HTML unescaped.
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Retry transient failures (server starting/busy, network blips) for a while
// before giving up: 400/408/429/5xx and fetch() network errors are retried.
async function fetchWithRetry(url, opts, { retries = 5, baseDelay = 400 } = {}) {
  let lastError = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    let res = null;
    try {
      res = await fetch(url, opts);
    } catch (err) {
      lastError = err;
    }
    if (res && res.ok) return res;
    if (res) {
      const retryable =
        [400, 408, 429].includes(res.status) || res.status >= 500;
      const body = await res.json().catch(() => ({}));
      lastError = new Error(body.detail || res.statusText || `HTTP ${res.status}`);
      if (!retryable) throw lastError;
    }
    if (attempt < retries) await sleep(baseDelay * (attempt + 1));
  }
  throw lastError;
}

async function jsonFetch(url, opts, retryOpts) {
  const res = await fetchWithRetry(url, opts, retryOpts);
  return res.json();
}

function logLine(text, cls, frame) {
  const row = document.createElement("div");
  if (cls) row.className = cls;
  row.textContent = frame == null ? text : `f${frame} · ${text}`;
  if (frame != null) {
    row.dataset.frame = String(frame);
    row.title = "Click to jump to this frame";
    row.addEventListener("click", () => seekTo(Number(frame)));
  }
  $("log").prepend(row);
}

function renderFrameIds(people, frame) {
  const ids = (people || [])
    .map((p) => `${p.global_id || "unresolved"} (local ${p.local_track})`)
    .join(" · ");
  $("frameIds").textContent = `Frame ${frame}: ${ids || "no people"}`;
}

function renderMarks(total) {
  const box = $("marks");
  if (!total || !markerEvents.length) {
    box.innerHTML = "";
    return;
  }
  box.innerHTML = markerEvents
    .map((ev) => {
      const pct = (Number(ev.frame || 0) / Math.max(total - 1, 1)) * 100;
      const cls = ev.type === "id_changed" ? "same" : ev.type === "id_cleared" ? "unres" : "new";
      return `<i class="${cls}" style="left:${pct}%" title="f${ev.frame} ${ev.type} ${ev.global_id || ev.previous_id || ""}"></i>`;
    })
    .join("");
}

function setScrubMax(lastFrame) {
  const scrub = $("scrub");
  if (lastFrame < 0) return;
  scrub.disabled = false;
  scrub.max = String(lastFrame);
  if (followingLive) scrub.value = String(lastFrame);
  renderMarks(lastFrame + 1);
}

function rememberEvent(ev) {
  if (!["id_assigned", "id_changed", "id_cleared", "new_id", "reid_same_video", "reid_cross_video"].includes(ev.type)) {
    return;
  }
  if (ev.frame == null) return;
  markerEvents.push(ev);
}

function setIdentityKind(globalId, kind) {
  identityKind[globalId] = kind;
  try {
    localStorage.setItem(`mct:idkind:${globalId}`, kind);
  } catch (_) { /* private mode / quota */ }
}

async function seekTo(frame) {
  if (!jobId || frame < 0) return;
  followingLive = false;
  $("followLive").disabled = false;
  $("scrub").value = String(frame);
  $("frame").src = `/gui/review/${jobId}/frame/${frame}`;
  try {
    const at = await jsonFetch(`/gui/review/${jobId}/at/${frame}`);
    renderFrameIds(at.people || [], frame);
  } catch (err) {
    $("frameIds").textContent = err.message;
  }
}

function setDownload(videos) {
  const link = $("downloadVideo");
  const first = (videos || [])[0];
  if (!first) {
    link.hidden = true;
    return;
  }
  link.hidden = false;
  link.href = first.url;
  link.download = first.name;
  link.textContent = `Download ${first.name}`;
}

async function refreshOutputs() {
  const items = await jsonFetch("/gui/outputs");
  if (!items.length) {
    $("savedVideos").textContent = "None yet. Process a video to write an annotated MP4 into outputs/.";
    return;
  }
  $("savedVideos").innerHTML = items
    .map((v) => `<div><a class="download" href="${esc(v.url)}" download="${esc(v.name)}">${esc(v.name)}</a></div>`)
    .join("");
}

function photoSlot(label, image) {
  if (image && image.url) {
    const title = `${image.video_name || ""} frame ${image.frame ?? "?"}`;
    return `<figure><figcaption>${label}</figcaption><img src="${esc(image.url)}" alt="${label}" title="${esc(title)}" /></figure>`;
  }
  return `<figure><figcaption>${label}</figcaption><div class="noimg">no image</div></figure>`;
}

function setTransport(status) {
  const btn = $("playPause");
  const running = status === "running" || status === "queued";
  const paused = status === "paused";
  const stopping = status === "stopping";
  const busy = running || paused || stopping;

  $("start").disabled = busy;
  $("stop").disabled = !busy;

  if (running) {
    btn.dataset.state = "running";
    btn.textContent = "⏸ Pause";
    btn.className = "btn btn-pause";
    btn.disabled = false;
  } else if (paused) {
    btn.dataset.state = "paused";
    btn.textContent = "▶ Resume";
    btn.className = "btn btn-play";
    btn.disabled = false;
  } else if (stopping) {
    btn.dataset.state = "stopping";
    btn.textContent = "⏸ …";
    btn.className = "btn btn-pause";
    btn.disabled = true;
  } else {
    // idle / queued / stopped / done / error → show Play (starts a new run)
    btn.dataset.state = "idle";
    btn.textContent = "▶ Play";
    btn.className = "btn btn-play";
    btn.disabled = false;
  }
}

function badgeFor(person) {
  if ((person.status || "active") !== "active") {
    return { cls: "same", label: `Archived (${person.status})` };
  }
  const kind = identityKind[person.global_person_id]
    || localStorage.getItem(`mct:idkind:${person.global_person_id}`);
  if (kind === "reid_cross_video") return { cls: "reid", label: "Re-ID from another video" };
  if (kind === "reid_same_video") return { cls: "same", label: "Re-ID in this video" };
  if (kind === "new_id") return { cls: "new", label: "New Global ID" };
  if ((person.videos || []).length > 1) return { cls: "reid", label: "Seen on multiple videos" };
  return { cls: "new", label: "In gallery" };
}

function visibleActiveIds(people) {
  return (people || [])
    .filter((p) => (p.status || "active") === "active")
    .map((p) => p.global_person_id);
}

function persistHidden() {
  try {
    localStorage.setItem("mct:hiddenIds", JSON.stringify([...hiddenIds]));
  } catch (_) { /* private mode / quota */ }
}

function pushVisibility() {
  jsonFetch("/gui/visibility", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hidden: [...hiddenIds] }),
  }).catch(() => { /* server may not have a processor yet */ });
}

function updateAllBox() {
  const el = $("showAll");
  if (!el) return;
  el.checked = visibleActiveIds(lastPeople).every(
    (id) => !hiddenIds.has(id)
  );
}

function renderGallery(people) {
  lastPeople = people || [];
  const box = $("gallery");
  if (!lastPeople.length) {
    box.innerHTML = '<div class="empty">Gallery empty.</div>';
    updateAllBox();
    return;
  }
  // Active identities on top, archived/inactive at the bottom.
  const ordered = lastPeople.slice().sort((a, b) => {
    const aa = (a.status || "active") === "active" ? 0 : 1;
    const bb = (b.status || "active") === "active" ? 0 : 1;
    if (aa !== bb) return aa - bb;
    return String(a.global_person_id).localeCompare(String(b.global_person_id));
  });
  box.innerHTML = ordered
    .map((p) => {
      const active = (p.status || "active") === "active";
      const tick = active
        ? `<input type="checkbox" class="vis" data-gid="${p.global_person_id}" ` +
          `${hiddenIds.has(p.global_person_id) ? "" : "checked"} ` +
          `title="Show/hide this person's bounding box" />`
        : "";
      const badge = badgeFor(p);
      const videos = esc((p.videos || []).join(", ") || "—");
      const obs = (p.observations || [])
        .slice(-4)
        .reverse()
        .map(
          (o) =>
            `${esc(o.modality)} · ${esc(o.video_name || o.camera_id)} · frame ${esc(o.frame ?? "?")} · q=${Number(o.quality || 0).toFixed(2)} · ${esc(o.viewpoint || "?")}`
        )
        .join("<br/>");
      return `<article class="card clickable" id="card-${p.global_person_id}" title="Open details for ${p.global_person_id}">
        <h3><span class="hdr-left">${tick}${p.global_person_id}</span><span class="badge ${badge.cls}">${badge.label}</span></h3>
        <div class="obs">Cameras: ${p.first_camera_id || "?"} → ${p.last_camera_id || "?"}</div>
        <div class="obs">Source videos: ${videos}</div>
        <div class="obs">Body embeddings: ${p.body_count} · Face embeddings: ${p.face_count}</div>
        <div class="obs">Tracks (${(p.tracks || []).length}): ${esc((p.tracks || []).map((t) => ((p.videos || []).length > 1 && t.video_name ? `${t.track_id} (${t.video_name})` : t.track_id)).join(", ") || "—")}</div>
        <div class="id-photos">${photoSlot("Person", p.body_image)}${photoSlot("Face", p.face_image)}</div>
        <div class="obs">${obs}</div>
        <div class="card-cta">Identity details →</div>
      </article>`;
    })
    .join("");
  box.querySelectorAll("input.vis").forEach((cb) => {
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      const gid = cb.dataset.gid;
      if (cb.checked) hiddenIds.delete(gid);
      else hiddenIds.add(gid);
      persistHidden();
      pushVisibility();
      updateAllBox();
    });
  });
  box.querySelectorAll(".card").forEach((card) => {
    const pid = card.id.replace(/^card-/, "");
    card.addEventListener("click", (e) => {
      // Do not hijack clicks on inner interactive elements.
      if (e.target.closest("a, button, input, label")) return;
      window.open(`/ui/person.html?id=${encodeURIComponent(pid)}`, "_blank");
    });
  });
  updateAllBox();
}

async function refreshGallery() {
  const people = await jsonFetch("/gui/persons");
  renderGallery(people);
}

async function saveGates() {
  await jsonFetch("/gui/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      require_body: $("requireBody").checked,
      require_face: $("requireFace").checked,
    }),
  });
}

function applyEvent(ev) {
  rememberEvent(ev);
  if (ev.type === "frame") {
    setScrubMax(ev.frame);
    if (followingLive) {
      if (ev.jpeg) $("frame").src = `data:image/jpeg;base64,${ev.jpeg}`;
      renderFrameIds(ev.people || [], ev.frame);
    }
    const pct = ev.total ? Math.round((ev.frame / ev.total) * 100) : 0;
    $("progress").textContent = `${ev.video_name} as camera ${ev.camera_id} · frame ${ev.frame}/${ev.total || "?"} · ${pct}%`;
    $("bar").style.width = `${pct}%`;
  } else if (ev.type === "id_assigned") {
    logLine(`ASSIGN ${ev.global_id} ← local ${ev.local_track}`, "new", ev.frame);
  } else if (ev.type === "id_changed") {
    logLine(`CHANGE local ${ev.local_track}: ${ev.previous_id} → ${ev.global_id}`, "same", ev.frame);
  } else if (ev.type === "id_cleared") {
    logLine(`CLEARED local ${ev.local_track} (was ${ev.previous_id})`, "unres", ev.frame);
  } else if (ev.type === "new_id") {
    setIdentityKind(ev.global_id, "new_id");
    logLine(`NEW ${ev.global_id} on ${ev.video_name} (local ${ev.local_track})`, "new", ev.frame);
    refreshGallery();
  } else if (ev.type === "reid_same_video") {
    const prev = identityKind[ev.global_id] === "reid_cross_video" ? "reid_cross_video" : "reid_same_video";
    setIdentityKind(ev.global_id, prev);
    logLine(`SAME VIDEO re-ID ${ev.global_id} on ${ev.video_name} (local ${ev.local_track}, score ${Number(ev.score || 0).toFixed(2)})`, "same", ev.frame);
    refreshGallery();
  } else if (ev.type === "gallery_updated") {
    refreshGallery();
  } else if (ev.type === "duplicates") {
    renderDuplicates(ev.pairs || []);
    refreshGallery();
  } else if (ev.type === "reid_cross_video") {
    setIdentityKind(ev.global_id, "reid_cross_video");
    logLine(`CROSS VIDEO re-ID ${ev.global_id}: first seen on ${ev.first_video}, now ${ev.video_name}`, "reid", ev.frame);
    refreshGallery();
    const card = document.getElementById(`card-${ev.global_id}`);
    if (card) card.classList.add("flash");
  } else if (ev.type === "paused") {
    $("progress").textContent = `Paused at frame ${ev.frame || "?"}`;
    setTransport("paused");
    logLine("Paused.");
    refreshGallery();
  } else if (ev.type === "resumed") {
    setTransport("running");
    logLine("Resumed.");
  } else if (ev.type === "stopping") {
    setTransport("stopping");
    logLine("Stopping…");
  } else if (ev.type === "stopped") {
    $("progress").textContent = `Stopped. frames=${ev.frames} tracks=${ev.tracks} new=${ev.new} reused=${ev.reused}`;
    setTransport("stopped");
    logLine(`Stopped ${ev.run_id || ""}. New IDs: ${ev.new}. Reused IDs: ${ev.reused}.`, "unres");
    setDownload(ev.output_videos);
    refreshGallery();
    refreshOutputs();
  } else if (ev.type === "done") {
    $("progress").textContent = `Done. frames=${ev.frames} tracks=${ev.tracks} new=${ev.new} reused=${ev.reused}`;
    $("bar").style.width = "100%";
    setTransport("done");
    logLine(`Finished ${ev.run_id}. New IDs: ${ev.new}. Reused IDs: ${ev.reused}. Click a log line or drag the slider to review.`, "reid");
    setDownload(ev.output_videos);
    refreshGallery();
    refreshOutputs();
  } else if (ev.type === "error") {
    $("progress").textContent = `Error: ${ev.message}`;
    setTransport("error");
    logLine(ev.message, "unres");
  } else if (ev.type === "started") {
    setTransport("running");
    logLine(`Processing ${ev.video_name} as camera ${ev.camera_id}`);
  }
}

async function pollEvents() {
  try {
    const data = await jsonFetch(`/gui/events?after=${after}`);
    for (const ev of data.events || []) {
      after = Math.max(after, ev.seq || 0);
      applyEvent(ev);
    }
    const job = await jsonFetch("/gui/job");
    setTransport(job.status || "idle");
    if (["running", "queued", "paused", "stopping"].includes(job.status)) {
      pollTimer = setTimeout(pollEvents, 250);
    }
  } catch (err) {
    $("progress").textContent = err.message;
    pollTimer = setTimeout(pollEvents, 1500);
  }
}

async function startProcessing() {
  const file = $("file").files[0];
  if (!file) {
    alert("Choose a video from your computer first.");
    return;
  }
  await saveGates();
  const max = $("maxFrames").value ? Number($("maxFrames").value) : "";
  const body = new FormData();
  body.append("video", file, file.name);
  if (max) body.append("max_frames", String(max));
  body.append("require_body", $("requireBody").checked ? "true" : "false");
  body.append("require_face", $("requireFace").checked ? "true" : "false");
  setTransport("queued");
  $("log").innerHTML = "";
  markerEvents.length = 0;
  followingLive = true;
  after = 0;
  try {
    const started = await jsonFetch("/gui/process", { method: "POST", body });
    jobId = started.job_id;
    $("progress").textContent = `Starting ${started.video_name} as camera ${started.camera_id}`;
    $("followLive").disabled = true;
    $("downloadVideo").hidden = true;
    pollEvents();
  } catch (err) {
    setTransport("idle");
    alert(err.message);
  }
}

$("start").addEventListener("click", startProcessing);

$("playPause").addEventListener("click", async () => {
  const state = $("playPause").dataset.state;
  try {
    if (state === "running") {
      await jsonFetch("/gui/pause", { method: "POST" });
      setTransport("paused");
      return;
    }
    if (state === "paused") {
      await jsonFetch("/gui/resume", { method: "POST" });
      followingLive = true;
      $("followLive").disabled = true;
      setTransport("running");
      if (!pollTimer) pollEvents();
      return;
    }
    // idle / done / stopped / error → start a new run
    await startProcessing();
  } catch (err) {
    alert(err.message);
  }
});

$("stop").addEventListener("click", async () => {
  try {
    await jsonFetch("/gui/stop", { method: "POST" });
    setTransport("stopping");
  } catch (err) {
    alert(err.message);
  }
});

$("scrub").addEventListener("input", () => {
  seekTo(Number($("scrub").value));
});

$("followLive").addEventListener("click", () => {
  followingLive = true;
  $("followLive").disabled = true;
  setScrubMax(Number($("scrub").max || 0));
});

$("requireBody").addEventListener("change", () => saveGates().catch((err) => alert(err.message)));
$("requireFace").addEventListener("change", () => saveGates().catch((err) => alert(err.message)));

$("reset").addEventListener("click", async () => {
  if (!confirm("Delete every Global ID, embedding, snapshot, and uploaded video?")) return;
  await jsonFetch("/gui/reset", { method: "POST" });
  Object.keys(identityKind).forEach((k) => delete identityKind[k]);
  $("gallery").innerHTML = '<div class="empty">Gallery empty.</div>';
  $("log").innerHTML = "";
  markerEvents.length = 0;
  jobId = null;
  $("frame").removeAttribute("src");
  $("progress").textContent = "Gallery cleared. Annotated videos in outputs/ were kept.";
  $("bar").style.width = "0";
  $("frameIds").textContent = "IDs on the current frame will appear here.";
  $("marks").innerHTML = "";
  $("scrub").value = "0";
  $("scrub").max = "0";
  $("scrub").disabled = true;
  $("downloadVideo").hidden = true;
  setTransport("idle");
  refreshOutputs().catch(() => {});
});

setTransport("idle");
refreshGallery().catch(() => {});

// Master show/hide control for bounding boxes.
$("showAll").addEventListener("change", () => {
  const ids = visibleActiveIds(lastPeople);
  if ($("showAll").checked) hiddenIds.clear();
  else ids.forEach((id) => hiddenIds.add(id));
  persistHidden();
  pushVisibility();
  renderGallery(lastPeople);
});
pushVisibility();

$("deleteInactive").addEventListener("click", async () => {
  const inactive = lastPeople.filter((p) => (p.status || "active") !== "active");
  if (!inactive.length) {
    alert("There are no inactive identities.");
    return;
  }
  if (!confirm(`Delete ${inactive.length} inactive identity(ies) and their saved images? This cannot be undone.`)) {
    return;
  }
  try {
    await jsonFetch("/gui/persons/delete_inactive", { method: "POST" });
    await refreshGallery();
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
});

function renderDuplicates(pairs) {
  const box = $("duplicates");
  if (!pairs || !pairs.length) {
    box.hidden = false;
    box.innerHTML = '<div class="empty">No near-duplicate identities found.</div>';
    return;
  }
  const block = (label, x) => {
    if (!x) return "";
    const flag = x.similar ? "✅ similar" : "⚠️ not similar";
    const im = (u, o) =>
      u
        ? `<figure><img src="${esc(u)}" alt="${esc(o.global_id)} ${esc(o.track)}" /><figcaption>${esc(o.global_id)} · ${esc(o.track)} · f${esc(o.frame)} · ${esc(o.video)} (${esc(o.viewpoint)})</figcaption></figure>`
        : "";
    return `<div class="dup-block">
      <div class="obs"><b>${label}</b> · cosine <b>${x.cosine.toFixed(3)}</b> · ${flag} (threshold ${x.threshold})</div>
      <div class="id-photos">${im(x.image_a, x.a)}${im(x.image_b, x.b)}</div>
    </div>`;
  };
  box.hidden = false;
  box.innerHTML = `<h2>Different IDs, same-looking tracks (${pairs.length})</h2>` + pairs
    .map((p) => `<article class="dup-card">
      <div class="obs"><b>${esc(p.pair)}</b> — different Global IDs, but the tracks look alike</div>
      ${p.seen_together ? `<div class="obs"><b>⛔ ${esc(p.a)} and ${esc(p.b)} were on screen together in the same frame — two different people. Do not merge.</b></div>` : ""}
      ${block("Body", p.body)}
      ${block("Face", p.face)}
      <div class="dup-actions">
        <button type="button" class="primary small" data-keep="${p.a}" data-other="${p.b}">Merge → keep ${p.a}</button>
        <button type="button" class="primary small" data-keep="${p.b}" data-other="${p.a}">Merge → keep ${p.b}</button>
      </div>
    </article>`)
    .join("");
  box.querySelectorAll("button[data-keep]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const keep = btn.dataset.keep;
      const other = btn.dataset.other;
      if (!confirm(`Merge ${other} into ${keep}? All its embeddings, tracks and images move to ${keep}.`)) return;
      const merge = (force) =>
        jsonFetch("/gui/persons/merge", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: other, target: keep, force }),
        });
      try {
        try {
          await merge(false);
        } catch (err) {
          if (!/on screen together/.test(err.message)) throw err;
          if (!confirm(`${err.message}\n\nMerge anyway?`)) return;
          await merge(true);
        }
        await refreshGallery();
        $("findDuplicates").click();
      } catch (err) {
        alert(`Merge failed: ${err.message}`);
      }
    });
  });
}

$("findDuplicates").addEventListener("click", async () => {
  $("duplicates").hidden = false;
  $("duplicates").innerHTML = '<div class="empty">Checking…</div>';
  try {
    const res = await jsonFetch("/gui/find_duplicates", { method: "POST" });
    renderDuplicates(res.pairs || []);
  } catch (err) {
    $("duplicates").innerHTML = `<div class="empty">Failed: ${esc(err.message)}</div>`;
  }
});
jsonFetch("/gui/settings")
  .then((s) => {
    $("requireBody").checked = !!s.require_body;
    $("requireFace").checked = !!s.require_face;
  })
  .catch(() => {});
refreshOutputs().catch(() => {});
jsonFetch("/gui/job").then((job) => {
  if (job.job_id) {
    jobId = job.job_id;
    (job.id_events || []).forEach(rememberEvent);
    setScrubMax((job.reviewed_frames || 1) - 1);
    setDownload(job.output_videos);
  }
  if (["running", "queued", "paused", "stopping"].includes(job.status)) {
    setTransport(job.status);
    pollEvents();
  }
});

/* ── Draggable splitter between the video and the log ────────────────
 * Lets the operator choose how much screen goes to the video vs. the log.
 * - Minimum video height: 300px  (never too small to see detections)
 * - Minimum log height:   120px  (always room for a few lines)
 * - Saved to localStorage so the chosen size survives a reload.
 */
(function initSplitter() {
  const stage = document.querySelector(".stage");
  const splitter = document.getElementById("splitter");
  if (!stage || !splitter) return;

  const STORAGE_KEY = "mct:stage-split";
  const SPLITTER_H = 6;
  const MIN_VIDEO = 300;
  const MIN_LOG = 120;

  function applySplit(videoHeight) {
    const total = stage.getBoundingClientRect().height;
    if (total <= 0) return videoHeight;
    const max = total - SPLITTER_H - MIN_LOG;
    const clamped = Math.max(MIN_VIDEO, Math.min(videoHeight, Math.max(MIN_VIDEO, max)));
    const logH = total - clamped - SPLITTER_H;
    stage.style.gridTemplateRows = `${clamped}px ${SPLITTER_H}px ${logH}px`;
    return clamped;
  }

  function restoreOrPickDefault() {
    const total = stage.getBoundingClientRect().height;
    if (total <= 0) return;
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    if (saved && saved >= MIN_VIDEO) {
      applySplit(saved);
    } else {
      // Default: video gets most of the space; log gets ~280px.
      applySplit(Math.max(MIN_VIDEO, total - 280 - SPLITTER_H));
    }
  }

  let dragging = false;
  let startY = 0;
  let startHeight = 0;

  splitter.addEventListener("mousedown", (e) => {
    const topRect = stage.querySelector(".stage-top").getBoundingClientRect();
    dragging = true;
    startY = e.clientY;
    startHeight = topRect.height;
    splitter.classList.add("dragging");
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const delta = e.clientY - startY;
    const newH = applySplit(startHeight + delta);
    try {
      localStorage.setItem(STORAGE_KEY, String(Math.round(newH)));
    } catch (_) { /* quota or private mode */ }
  });

  function endDrag() {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.style.userSelect = "";
  }
  window.addEventListener("mouseup", endDrag);
  window.addEventListener("blur", endDrag);

  // Keep the split sensible when the window is resized.
  window.addEventListener("resize", restoreOrPickDefault);

  // Run once layout is settled.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", restoreOrPickDefault);
  } else {
    // Small delay lets grid finish layout before we measure.
    requestAnimationFrame(restoreOrPickDefault);
  }
})();
