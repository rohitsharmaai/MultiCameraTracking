/* Identity detail page: /ui/person.html?id=P0001 */
const $ = (id) => document.getElementById(id);

const params = new URLSearchParams(window.location.search);
const personId = params.get("id") || "";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtTime(epoch) {
  if (!epoch) return "—";
  try {
    return new Date(Number(epoch) * 1000).toLocaleString();
  } catch (_) {
    return String(epoch);
  }
}

function fact(label, value) {
  const box = document.createElement("div");
  box.className = "fact";
  const b = document.createElement("b");
  b.textContent = label;
  box.appendChild(b);
  box.appendChild(document.createTextNode(value == null || value === "" ? "—" : String(value)));
  return box;
}

function badgeFor(p) {
  if ((p.status || "active") !== "active") {
    return { cls: "same", label: `Archived (${p.status})` };
  }
  const kind = (localStorage.getItem(`mct:idkind:${p.global_person_id}`) || "").trim();
  if (kind === "reid_cross_video") return { cls: "reid", label: "Re-ID from another video" };
  if (kind === "reid_same_video") return { cls: "same", label: "Re-ID in this video" };
  if (kind === "new_id") return { cls: "new", label: "New Global ID" };
  if ((p.videos || []).length > 1) return { cls: "reid", label: "Seen on multiple videos" };
  return { cls: "new", label: "In gallery" };
}

function fileUrl(path) {
  if (!path) return "";
  return "file://" + path.split("/").map(encodeURIComponent).join("/");
}

function folderFact(p) {
  const box = document.createElement("div");
  box.className = "fact";
  const b = document.createElement("b");
  b.textContent = "Image folder";
  box.appendChild(b);
  if (!p.snapshot_dir) {
    box.appendChild(document.createTextNode("—"));
    return box;
  }

  const openFolder = async () => {
    try {
      const r = await fetch(`/gui/snapshots/${encodeURIComponent(p.global_person_id)}/open`, { method: "POST" });
      const j = await r.json();
      if (!j.ok) window.alert(`Could not open folder: ${j.error || "unknown"}\n${j.path || ""}`);
    } catch (err) {
      window.alert(`Could not open folder: ${err.message}`);
    }
  };

  // The path itself is the link. Browsers block file:// navigation from an
  // http page, so clicking it asks the server to open the folder instead.
  const a = document.createElement("a");
  a.href = "#";
  a.title = "Open this folder in the file manager on the machine running the server";
  a.textContent = p.snapshot_dir;
  a.style.cssText = "display:block;word-break:break-all;color:var(--new);font-size:12px;margin-bottom:6px;";
  a.addEventListener("click", (e) => { e.preventDefault(); openFolder(); });
  box.appendChild(a);

  const hint = document.createElement("div");
  hint.className = "empty-note";
  hint.style.cssText = "font-size:11px;margin-bottom:6px;";
  hint.textContent = p.snapshot_dir_exists ? "Click the path or the button to open it." : "Nothing saved here yet.";
  box.appendChild(hint);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";

  const openBtn = document.createElement("button");
  openBtn.type = "button";
  openBtn.className = "btn";
  openBtn.textContent = "Open folder";
  openBtn.addEventListener("click", openFolder);
  row.appendChild(openBtn);

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "btn";
  copyBtn.textContent = "Copy path";
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(p.snapshot_dir);
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = "Copy path"; }, 1200);
    } catch (_) {
      window.prompt("Copy the folder path:", p.snapshot_dir);
    }
  });
  row.appendChild(copyBtn);
  box.appendChild(row);
  return box;
}

function showLightbox(url, caption) {
  const lb = $("lightbox");
  const img = lb.querySelector("img");
  img.src = url;
  img.alt = caption || "crop";
  lb.classList.add("on");
}

let personData = null;

function fillGrid(grid, snaps) {
  grid.innerHTML = snaps
    .map((s) => {
      const cap = `<span class="mod">${esc(s.modality)}</span> · ${esc(s.video_name || s.camera_id || "?")}<br/>` +
        `frame ${esc(s.frame_number ?? "?")} · track ${esc(s.local_track ?? "?")}<br/>` +
        `q=${Number(s.quality || 0).toFixed(2)} · ${esc(s.viewpoint || "?")} · ${esc(s.match_kind || "")}`;
      return `<figure class="snap"><img loading="lazy" src="${esc(s.url)}" alt="${esc(s.modality)} crop" data-full="${esc(s.url)}" data-cap="${esc(s.modality)} ${esc(s.video_name || "")} frame ${esc(s.frame_number ?? "")}" /><figcaption class="cap">${cap}</figcaption></figure>`;
    })
    .join("");
  grid.querySelectorAll("img").forEach((img) => {
    img.addEventListener("click", () => showLightbox(img.dataset.full, img.dataset.cap));
  });
}

// Local track IDs (A_1, ...) restart for every video, so a track is the pair
// (video, track). Snapshot rows store either the file name or its stem.
const stem = (v) => String(v || "").replace(/\.[^.]*$/, "");
const trackKey = (video, track) => `${track} · ${stem(video)}`;

function renderTracks() {
  const snaps = (personData.snapshots || []).map((s) => ({
    ...s,
    track_key: s.local_track ? trackKey(s.video_name, s.local_track) : "",
  }));
  const videos = new Set([
    ...(personData.tracks || []).map((t) => stem(t.video_name)),
    ...snaps.map((s) => stem(s.video_name)),
  ]);
  const label = (k) => (videos.size > 1 ? k : k.split(" · ")[0]);
  const tracks = (personData.tracks || []).map((t) => trackKey(t.video_name, t.track_id));
  const snapTracks = snaps.map((s) => s.track_key).filter(Boolean);
  const allTracks = Array.from(new Set([...tracks, ...snapTracks])).sort();
  const safeId = (t) => "t_" + String(t).replace(/[^A-Za-z0-9_-]/g, "_");

  const list = $("trackList");
  if (!allTracks.length) {
    list.textContent = "No tracks recorded for this identity.";
  } else {
    list.innerHTML = allTracks
      .map((t) => {
        const n = snaps.filter((s) => s.track_key === t).length;
        return `<span class="track-chip">${esc(label(t))} · ${n} image${n === 1 ? "" : "s"}</span>`;
      })
      .join("");
  }

  const container = $("trackSections");
  if (!allTracks.length) {
    container.innerHTML = '<div class="empty-note">No images saved yet.</div>';
    return;
  }
  container.innerHTML = allTracks
    .map((t) => {
      const body = snaps.filter((s) => s.track_key === t && s.modality === "body");
      const face = snaps.filter((s) => s.track_key === t && s.modality === "face");
      return `<section class="track-block">
        <h3>Track ${esc(label(t))} <span class="empty-note">${body.length} body · ${face.length} face</span></h3>
        <div class="snap-columns">
          <div><h4>Body</h4><div class="snap-grid" id="${safeId(t)}-body"></div></div>
          <div><h4>Face</h4><div class="snap-grid" id="${safeId(t)}-face"></div></div>
        </div>
      </section>`;
    })
    .join("");
  allTracks.forEach((t) => {
    fillGrid(document.getElementById(`${safeId(t)}-body`), snaps.filter((s) => s.track_key === t && s.modality === "body"));
    fillGrid(document.getElementById(`${safeId(t)}-face`), snaps.filter((s) => s.track_key === t && s.modality === "face"));
  });
}

function render(p) {
  personData = p;
  document.title = `${p.global_person_id} — Global Identity`;
  $("pid").textContent = p.global_person_id;
  const badge = badgeFor(p);
  const badgeEl = $("badge");
  badgeEl.className = `badge ${badge.cls}`;
  badgeEl.textContent = badge.label;

  const facts = $("facts");
  facts.innerHTML = "";
  facts.append(
    fact("Status", p.status),
    fact("First camera", p.first_camera_id),
    fact("Last camera", p.last_camera_id),
    fact("Last seen", fmtTime(p.last_seen_at)),
    fact("Source videos", (p.videos || []).join(", ") || "—"),
    fact("Body embeddings", p.body_count),
    fact("Face embeddings", p.face_count),
    fact("Saved crops", (p.snapshots || []).length),
    fact("Tracks", (p.tracks || []).map((t) => trackKey(t.video_name, t.track_id)).join(", ") || "—"),
    folderFact(p)
  );

  const base = $("baseCrops");
  base.innerHTML = "";
  const figures = [
    ["Person (body)", p.body_image],
    ["Face", p.face_image],
  ];
  let anyFigure = false;
  for (const [label, image] of figures) {
    const fig = document.createElement("figure");
    const cap = document.createElement("figcaption");
    cap.textContent = image && image.url
      ? `${label} · ${image.video_name || image.camera_id || "?"} frame ${image.frame ?? "?"} · q=${Number(image.quality || 0).toFixed(2)}`
      : `${label} · no image saved`;
    if (image && image.url) {
      anyFigure = true;
      const img = document.createElement("img");
      img.src = image.url;
      img.alt = label;
      img.addEventListener("click", () => showLightbox(image.url, label));
      fig.appendChild(img);
    } else {
      const ph = document.createElement("div");
      ph.className = "noimg";
      ph.textContent = "no image";
      ph.style.cssText = "height:220px;width:220px;display:flex;align-items:center;justify-content:center;color:var(--muted);background:#0b1016;";
      fig.appendChild(ph);
    }
    fig.appendChild(cap);
    base.appendChild(fig);
  }
  if (!anyFigure) {
    const note = document.createElement("div");
    note.className = "empty-note";
    note.textContent = "No reference crops saved for this identity yet.";
    base.appendChild(note);
  }

  $("obsBody").innerHTML = (p.observations || [])
    .map((o) => `<tr><td>${esc(o.modality)}</td><td>${esc(o.camera_id)}</td><td>${esc(o.video_name)}</td><td>${esc(o.frame)}</td><td>${esc(o.track_id)}</td><td>${Number(o.quality || 0).toFixed(2)}</td><td>${esc(o.viewpoint)}</td><td>${esc(o.model)}</td></tr>`)
    .join("");
  $("obsNote").hidden = (p.observations || []).length > 0;

  renderTracks();
  loadTrackAudit();
}

// The embedding's frame, and the frame of the saved crop when that differs.
function frameText(o) {
  if (o.crop_frame != null && o.crop_frame !== o.frame) {
    return `embedding f${o.frame} · image f${o.crop_frame}`;
  }
  return `frame ${o.frame}`;
}

async function loadTrackAudit() {
  const box = $("trackAudit");
  if (!box || !personId) return;
  box.textContent = "Checking most-similar tracks…";
  try {
    const res = await fetch(`/gui/persons/${encodeURIComponent(personId)}/audit`, { method: "POST" });
    if (!res.ok) {
      box.textContent = `Track comparison unavailable (${res.status}).`;
      return;
    }
    const d = await res.json();
    const part = (label, x) => {
      if (!x) {
        return `<div class="dup-card"><div class="obs"><b>${label}</b> — fewer than two tracks</div></div>`;
      }
      const flag = x.similar ? "✅ similar" : "⚠️ not similar";
      const cap = (o) => `${o.track} · ${stem(o.video)} · ${frameText(o)} (${o.viewpoint})`;
      const im = (u, c) =>
        u ? `<figure><img src="${esc(u)}" alt="${esc(c)}" /><figcaption>${esc(c)}</figcaption></figure>` : "";
      return `<div class="dup-card">
        <div class="obs"><b>${label}</b> · most similar pair cosine <b>${x.cosine.toFixed(3)}</b> · ${flag} (threshold ${x.threshold})</div>
        <div class="id-photos">${im(x.image_a, cap(x.a))}${im(x.image_b, cap(x.b))}</div>
      </div>`;
    };
    box.innerHTML = part("Body", d.body) + part("Face", d.face);

    const co = $("coTracks");
    if (co) {
      const c = d.cooccurring;
      if (!c || !c.pair) {
        co.innerHTML = '<div class="empty-note">No two tracks of this identity are on screen at the same time.</div>';
      } else {
        const blk = (label, x) => {
          if (!x) return "";
          const flag = x.similar ? "✅ similar" : "⚠️ not similar";
          const cap = (o) => `${o.global_id} · ${o.track} · ${frameText(o)}`;
          const im = (u, cc) =>
            u ? `<figure><img src="${esc(u)}" alt="${esc(cc)}" /><figcaption>${esc(cc)}</figcaption></figure>` : "";
          return `<div class="dup-block">
            <div class="obs"><b>${label}</b> · cosine <b>${x.cosine.toFixed(3)}</b> · ${flag} (threshold ${x.threshold})</div>
            <div class="id-photos">${im(x.image_a, cap(x.a))}${im(x.image_b, cap(x.b))}</div>
          </div>`;
        };
        co.innerHTML = `<div class="dup-card">
          <div class="obs"><b>Tracks ${esc(c.pair[0])} &amp; ${esc(c.pair[1])}</b>${c.video ? ` of ${esc(c.video)}` : ""} on screen at the same time (frames ${esc(c.overlap_frames[0])}–${esc(c.overlap_frames[1])})</div>
          ${blk("Body", c.body)}${blk("Face", c.face)}
        </div>`;
      }
    }
  } catch (err) {
    box.textContent = `Track comparison failed: ${err.message}`;
  }
}

$("lightbox").addEventListener("click", () => $("lightbox").classList.remove("on"));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("lightbox").classList.remove("on");
});

$("deletePerson").addEventListener("click", async () => {
  if (!personId) return;
  if (!confirm(`Delete ${personId} and all its saved images? This cannot be undone.`)) return;
  try {
    const res = await fetch(`/gui/persons/${encodeURIComponent(personId)}`, { method: "DELETE" });
    if (res.ok) {
      window.location.href = "/";
      return;
    }
    const body = await res.json().catch(() => ({}));
    alert(`Delete failed: ${body.detail || res.status}`);
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
});

async function loadPerson() {
  let lastErr;
  for (let attempt = 0; attempt <= 5; attempt++) {
    try {
      const res = await fetch(`/gui/persons/${encodeURIComponent(personId)}`);
      if (res.ok) {
        render(await res.json());
        return;
      }
      if (res.status === 404) throw new Error(`Identity ${personId} not found`);
      const body = await res.json().catch(() => ({}));
      lastErr = new Error(body.detail || `${res.status} ${res.statusText}`);
      if (!([400, 408, 429].includes(res.status) || res.status >= 500)) throw lastErr;
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
  }
  throw lastErr;
}

async function refreshWhileProcessing() {
  try {
    const job = await (await fetch("/gui/job")).json();
    if (["running", "queued", "paused", "stopping"].includes(job.status)) {
      await loadPerson();
    }
  } catch (_) {
    /* transient; the next tick retries */
  }
}

if (!personId) {
  $("pid").textContent = "No identity selected";
  $("obsNote").textContent = "Missing ?id=P0000 in the URL.";
  $("obsNote").hidden = false;
} else {
  loadPerson().catch((err) => {
    $("pid").textContent = personId;
    $("obsNote").textContent = err.message;
    $("obsNote").hidden = false;
  });
  // Keep the reference crops current while a video is being processed.
  setInterval(refreshWhileProcessing, 4000);
}
