/**
 * LensModule — Google Lens-style selection + AI chat for the HeartScape 3D canvas.
 *
 * Features:
 *  - Drag-to-select a region on the 3D canvas
 *  - Expands raycaster hits to full anatomical sections (all same-named meshes)
 *  - Extracts heuristic mesh metrics (volume, surface area, bbox, elongation)
 *    and compares patient (right scene) to reference (left scene, Patient 1)
 *  - Grounded Gemini chat session with full metric context baked into system prompt
 *  - Multi-turn follow-up Q&A in a scrollable chat thread
 *  - "Save as Note" fires a custom event → heart-scene.js creates a sticky note
 */

import * as THREE from 'three';

// ── Clinical fallback descriptions ────────────────────────────────────────────
const FALLBACK = {
  LV:    'Left Ventricle (LV): Primary systemic pump. In CHD, may be hypoplastic (HLHS) or pressure-overloaded (L-TGA). Volume reduction vs. reference signals remodeling or underdevelopment. LV/RV ratio is a key severity indicator.',
  RV:    'Right Ventricle (RV): Low-pressure pump to the lungs normally. In DORV or D-TGA, the RV bears systemic load and hypertrophies markedly. Dilation vs. reference implies increased afterload or volume overload.',
  LA:    'Left Atrium (LA): Receives oxygenated pulmonary return. Dilation occurs with mitral regurgitation, elevated pulmonary venous pressure, or left-to-right shunts (ASD, VSD). Severely small LA indicates restrictive physiology (e.g., HLHS).',
  RA:    'Right Atrium (RA): Collects systemic venous return. Enlargement is prominent in ASD, Fontan circulation, Ebstein\'s anomaly, and tricuspid regurgitation — often preceding right heart failure.',
  Aorta: 'Aorta: The systemic great vessel. In D-TGA, arises anteriorly from the RV. In DORV, may override both ventricles. Dilation indicates aortopathy (Marfan, bicuspid AV) or post-arterial-switch remodeling.',
  PA:    'Pulmonary Artery (PA): Carries deoxygenated blood to the lungs. PA atresia/stenosis defines several CHD subtypes. Dilation indicates pulmonary hypertension. PA-to-Aorta ratio reflects vascular balance.',
  SVC:   'Superior Vena Cava (SVC, Label 7): Returns upper-body venous blood to the RA. Bilateral SVC is found in ~3% of CHD, especially heterotaxy. Post-Glenn, SVC is anastomosed directly to the pulmonary artery.',
  IVC:   'Inferior Vena Cava (IVC, Label 8): Returns lower-body blood. IVC interruption with azygos continuation is a hallmark of left isomerism. Critical landmark for Fontan circuit planning.',
};

function fallbackText(name) {
  return FALLBACK[name] ||
    `${name}: Cardiac structure in 3D segmentation. Compare size/shape to the reference heart (Patient 1, left canvas) to assess CHD-related morphological changes.`;
}

function pctDiff(patient, ref) {
  if (!ref || ref < 0.001) return '';
  const d = ((patient - ref) / ref) * 100;
  return d >= 0 ? `+${d.toFixed(1)}%` : `${d.toFixed(1)}%`;
}

// ── Anatomy name → feature key (matches heart-scene.js PART_NAME_TO_VOL_KEY) ──
const NAME_TO_FEATURE_KEY = {
  LV:    'Label_1_vol_ml',
  RV:    'Label_2_vol_ml',
  LA:    'Label_3_vol_ml',
  RA:    'Label_4_vol_ml',
  Aorta: 'Label_5_vol_ml',  AO: 'Label_5_vol_ml',
  PA:    'Label_6_vol_ml',
  SVC:   'Label_7_vol_ml',  PV: 'Label_7_vol_ml',
  IVC:   'Label_8_vol_ml',
};

// ── Geometry helpers (bbox + elongation only; volumes come from JSON) ──────────
function meshBboxMetrics(meshArray) {
  if (!meshArray.length) return null;
  const box = new THREE.Box3();
  let faces = 0;
  for (const m of meshArray) {
    box.expandByObject(m);
    const geo = m.geometry;
    if (geo) faces += geo.index ? geo.index.count / 3 : (geo.attributes?.position?.count ?? 0) / 3;
  }
  const size = box.getSize(new THREE.Vector3());
  const dims = [size.x, size.y, size.z].sort((a, b) => a - b);
  const elongation = dims[2] > 0 ? dims[2] / Math.max(dims[0], 0.001) : 1;
  return { faces: Math.round(faces), bbox: size, elongation };
}

// ─────────────────────────────────────────────────────────────────────────────
export class LensModule {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Camera}        camera         Left (reference) camera
   * @param {function}            getLeftMeshes  () => THREE.Mesh[]
   * @param {function}            getRightMeshes () => THREE.Mesh[]
   * @param {function}            getCameraRight () => THREE.Camera | null
   */
  constructor({ renderer, camera, getLeftMeshes, getRightMeshes, getCameraRight,
                getPatientFeatures = () => null, getReferenceFeatures = () => null }) {
    this.renderer             = renderer;
    this.camera               = camera;
    this.getLeftMeshes        = getLeftMeshes;
    this.getRightMeshes       = getRightMeshes;
    this.getCameraRight       = getCameraRight;
    this.getPatientFeatures   = getPatientFeatures;   // () => {Label_1_vol_ml: X, …} | null
    this.getReferenceFeatures = getReferenceFeatures; // () => {Label_1_vol_ml: X, …} | null

    this.active      = false;
    this.overlay     = null;
    this.ctx         = null;
    this.isSelecting = false;
    this.startX = 0; this.startY = 0;
    this.endX   = 0; this.endY   = 0;

    this._geminiState = 'uninit'; // 'uninit' | 'ready' | 'skip'
    this.geminiModel  = null;
    this.activeChat   = null;   // current multi-turn chat session
    this.chatHistory  = [];     // [{role:'ai'|'user', text}] for note export
    this.popup        = null;

    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp   = this._onMouseUp.bind(this);
    this._onKeyDown   = this._onKeyDown.bind(this);
    this._onResize    = this._onResize.bind(this);
  }

  init() {
    const parent = this.renderer.domElement.parentElement;
    this.overlay = document.createElement('canvas');
    Object.assign(this.overlay.style, {
      position: 'absolute', inset: '0', zIndex: '20',
      cursor: 'crosshair', display: 'none', pointerEvents: 'none',
    });
    parent.appendChild(this.overlay);
    this.ctx = this.overlay.getContext('2d');
    this._syncSize();
    window.addEventListener('resize', this._onResize);
    this._injectStyles();
  }

  // ── Activation ───────────────────────────────────────────────────────────────

  toggle() { this.active ? this.deactivate() : this.activate(); return this.active; }

  activate() {
    this.active = true;
    this.overlay.style.display = 'block';
    this.overlay.style.pointerEvents = 'all';
    this.overlay.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('keydown', this._onKeyDown);
  }

  deactivate() {
    this.active = false;
    this.overlay.style.display = 'none';
    this.overlay.style.pointerEvents = 'none';
    this.overlay.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
    window.removeEventListener('keydown', this._onKeyDown);
    this._clearCanvas();
    this.isSelecting = false;
  }

  // ── Canvas sizing ────────────────────────────────────────────────────────────

  _syncSize() {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.overlay.width = r.width; this.overlay.height = r.height;
  }
  _onResize() { this._syncSize(); }

  // ── Mouse ────────────────────────────────────────────────────────────────────

  _onMouseDown(e) {
    if (e.button !== 0) return;
    const rect = this.overlay.getBoundingClientRect();
    this.startX = e.clientX - rect.left; this.startY = e.clientY - rect.top;
    this.endX = this.startX; this.endY = this.startY;
    this.isSelecting = true;
    window.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup',   this._onMouseUp);
  }

  _onMouseMove(e) {
    if (!this.isSelecting) return;
    const rect = this.overlay.getBoundingClientRect();
    this.endX = e.clientX - rect.left; this.endY = e.clientY - rect.top;
    this._drawSelection();
  }

  _onMouseUp(e) {
    if (!this.isSelecting) return;
    this.isSelecting = false;
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup',   this._onMouseUp);
    const rect = this.overlay.getBoundingClientRect();
    this.endX = e.clientX - rect.left; this.endY = e.clientY - rect.top;
    this._clearCanvas();
    if (Math.abs(this.endX - this.startX) < 8 && Math.abs(this.endY - this.startY) < 8) return;
    this._handleSelection(e.clientX, e.clientY);
  }

  _onKeyDown(e) {
    if (e.key === 'Escape') {
      this.deactivate();
      const btn = document.getElementById('lens-btn');
      if (btn) { btn.classList.remove('active'); btn.setAttribute('aria-pressed', 'false'); }
    }
  }

  // ── Drawing ──────────────────────────────────────────────────────────────────

  _clearCanvas() { if (this.ctx) this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height); }

  _drawSelection() {
    const ctx = this.ctx, W = this.overlay.width, H = this.overlay.height;
    ctx.clearRect(0, 0, W, H);
    const x = Math.min(this.startX, this.endX), y = Math.min(this.startY, this.endY);
    const w = Math.abs(this.endX - this.startX), h = Math.abs(this.endY - this.startY);
    ctx.fillStyle = 'rgba(0,0,0,0.22)'; ctx.fillRect(0, 0, W, H);
    ctx.clearRect(x, y, w, h);
    ctx.fillStyle = 'rgba(59,130,246,0.1)'; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(99,160,255,0.9)'; ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]); ctx.strokeRect(x+.5, y+.5, w-1, h-1); ctx.setLineDash([]);
    ctx.fillStyle = '#60a5fa';
    [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([hx,hy]) => ctx.fillRect(hx-4, hy-4, 8, 8));
    ctx.fillStyle = 'rgba(96,165,250,0.55)';
    [[x+w/2,y],[x+w/2,y+h],[x,y+h/2],[x+w,y+h/2]].forEach(([hx,hy]) => ctx.fillRect(hx-2.5, hy-2.5, 5, 5));
  }

  // ── Raycasting + section grouping ────────────────────────────────────────────

  _pickMeshes() {
    const x1 = Math.min(this.startX, this.endX), y1 = Math.min(this.startY, this.endY);
    const x2 = Math.max(this.startX, this.endX), y2 = Math.max(this.startY, this.endY);
    const OW = this.overlay.width, OH = this.overlay.height;
    const samples = [];
    for (const rx of [0, 0.5, 1]) for (const ry of [0, 0.5, 1]) {
      samples.push([x1 + (x2-x1)*rx, y1 + (y2-y1)*ry]);
    }
    const raycaster = new THREE.Raycaster();
    const hitNames = new Set();
    // name → { meshes: THREE.Mesh[], side: 'left'|'right' }
    const groups = new Map();
    const leftSet = new Set(this.getLeftMeshes());

    for (const [px, py] of samples) {
      const normX = px / OW, normY = py / OH;
      let ndc, allMeshes, cam;
      if (normX < 0.5) {
        ndc = new THREE.Vector2((normX / 0.5) * 2 - 1, -(normY * 2 - 1));
        allMeshes = this.getLeftMeshes(); cam = this.camera;
      } else {
        ndc = new THREE.Vector2(((normX - 0.5) / 0.5) * 2 - 1, -(normY * 2 - 1));
        allMeshes = this.getRightMeshes(); cam = this.getCameraRight();
      }
      if (!allMeshes?.length || !cam) continue;
      raycaster.setFromCamera(ndc, cam);
      const hits = raycaster.intersectObjects(allMeshes, false);

      for (const hit of hits) {
        const name = (hit.object.name || '').trim() || 'Heart';
        if (hitNames.has(name) || groups.size >= 5) continue;
        hitNames.add(name);

        // Expand hit to ALL same-named meshes in the same scene pool
        const side = leftSet.has(hit.object) ? 'left' : 'right';
        const pool = side === 'left' ? this.getLeftMeshes() : this.getRightMeshes();
        const grouped = pool.filter(m => (m.name || '').trim() === name);
        groups.set(name, { meshes: grouped.length ? grouped : [hit.object], side });
      }
    }
    return groups; // Map<name, {meshes, side}>
  }

  // ── Metrics extraction ────────────────────────────────────────────────────────

  _extractMetrics(groups) {
    // Volumes & surface area come from pre-computed JSON (accurate mL values).
    // Bounding box + elongation come from the live Three.js geometry.
    const patFeatures = this.getPatientFeatures()   || {};
    const refFeatures = this.getReferenceFeatures() || {};

    const leftByName = new Map();
    for (const m of this.getLeftMeshes()) {
      const n = (m.name || '').trim() || 'Heart';
      if (!leftByName.has(n)) leftByName.set(n, []);
      leftByName.get(n).push(m);
    }
    const rightByName = new Map();
    for (const m of this.getRightMeshes()) {
      const n = (m.name || '').trim() || 'Heart';
      if (!rightByName.has(n)) rightByName.set(n, []);
      rightByName.get(n).push(m);
    }

    const result = new Map();
    for (const [name, { side }] of groups) {
      const featureKey = NAME_TO_FEATURE_KEY[name] ?? null;
      const patVolMl   = featureKey ? (patFeatures[featureKey] ?? 0) : 0;
      const refVolMl   = featureKey ? (refFeatures[featureKey] ?? 0) : 0;

      const patBbox = meshBboxMetrics(rightByName.get(name) || []);
      const refBbox = meshBboxMetrics(leftByName.get(name)  || []);

      result.set(name, {
        patient:   { volMl: patVolMl, ...(patBbox || { faces: 0, bbox: new THREE.Vector3(), elongation: 0 }) },
        reference: { volMl: refVolMl, ...(refBbox || { faces: 0, bbox: new THREE.Vector3(), elongation: 0 }) },
        side,
      });
    }
    return result;
  }

  // ── Prompt building ───────────────────────────────────────────────────────────

  _buildSystemPrompt(names, metricsMap) {
    let ctx =
      'You are a cardiac anatomy and congenital heart disease (CHD) expert integrated into ' +
      'HeartScape — a 3D visualization tool using HVSMR-2.0 MRI segmentation data from paediatric CHD patients.\n\n' +
      'The user has selected the following cardiac structures from the 3D rendering. ' +
      'Below are heuristic geometric metrics extracted directly from the Three.js mesh geometry.\n\n' +
      'Volumes are in mL from the HVSMR-2.0 feature extraction pipeline. ' +
      'Bounding-box dimensions are in normalised canvas units (not mm). ' +
      'Elongation = longest bbox axis / shortest bbox axis. ' +
      'When volumes are 0 it means the patient scan has not been uploaded yet — analyse using elongation and face count only.\n\n' +
      'MESH METRICS:\n';

    for (const name of names) {
      const m = metricsMap.get(name);
      if (!m) { ctx += `\n${name}: metrics unavailable\n`; continue; }
      const p = m.patient, r = m.reference;
      ctx += `\n${name}:\n`;
      ctx += `  Patient:   vol=${p.volMl > 0 ? p.volMl.toFixed(2) + ' mL' : '0 (scan not loaded)'}, ` +
             `bbox=${p.bbox.x.toFixed(2)}×${p.bbox.y.toFixed(2)}×${p.bbox.z.toFixed(2)} canvas-units, ` +
             `elongation=${p.elongation.toFixed(2)}, faces=${p.faces.toLocaleString()}\n`;
      ctx += `  Reference: vol=${r.volMl > 0 ? r.volMl.toFixed(2) + ' mL' : '0 (reference not loaded)'}, ` +
             `bbox=${r.bbox.x.toFixed(2)}×${r.bbox.y.toFixed(2)}×${r.bbox.z.toFixed(2)} canvas-units, ` +
             `elongation=${r.elongation.toFixed(2)}, faces=${r.faces.toLocaleString()}\n`;
      if (p.volMl > 0 && r.volMl > 0) {
        ctx += `  Delta:     volume ${pctDiff(p.volMl, r.volMl)}\n`;
        if (p.elongation > r.elongation * 1.15)
          ctx += `  Note:      structure is more elongated than reference — possible dilation\n`;
        if (p.elongation < r.elongation * 0.85)
          ctx += `  Note:      structure is more spherical than reference\n`;
      }
    }

    ctx +=
      '\nRESPONSE RULES:\n' +
      '- Always reference the specific numeric metrics above when making clinical observations\n' +
      '- Relate findings to known CHD conditions (HLHS, TGA, DORV, Tetralogy of Fallot, Fontan, etc.)\n' +
      '- Keep initial analysis to 4-6 sentences; follow-up answers can be longer if needed\n' +
      '- Do not use markdown headers or bullet lists — use flowing prose\n' +
      '- If follow-up questions go beyond these structures, use your clinical knowledge freely\n';

    return ctx;
  }

  // ── Selection handler ─────────────────────────────────────────────────────────

  _handleSelection(cursorClientX, cursorClientY) {
    const groups = this._pickMeshes();
    if (!groups.size) {
      this._showPopup(cursorClientX, cursorClientY, [], null, null);
      return;
    }

    // Expand highlighting to all meshes in each group
    const allHitMeshes = [];
    for (const { meshes } of groups.values()) allHitMeshes.push(...meshes);
    this._highlightMeshes(allHitMeshes);

    const names = [...groups.keys()];
    const metricsMap = this._extractMetrics(groups);

    this._showPopup(cursorClientX, cursorClientY, names, metricsMap, null);
    // Start async analysis (no queue needed — each popup gets its own chat)
    this._startChat(names, metricsMap);
  }

  // ── Highlighting ──────────────────────────────────────────────────────────────

  _highlightMeshes(meshes) {
    const saved = meshes.map(m => ({ mesh: m, mat: m.material }));
    const hl = new THREE.MeshPhongMaterial({
      color: 0x2563eb, emissive: 0x1e3a8a, emissiveIntensity: 0.55,
      transparent: true, opacity: 0.8, shininess: 60,
    });
    meshes.forEach(m => { m.material = hl.clone(); });
    setTimeout(() => { saved.forEach(({ mesh, mat }) => { mesh.material = mat; }); }, 2000);
  }

  // ── Gemini ────────────────────────────────────────────────────────────────────

  async _ensureGemini() {
    if (this._geminiState === 'ready') return true;
    if (this._geminiState === 'skip')  return false;
    const key = window.prompt(
      'HeartScape Lens — AI Analysis\n\n' +
      'Enter your Google Gemini API key for AI-powered cardiac chat.\n' +
      '(Press Cancel to use built-in clinical descriptions instead.)'
    );
    if (!key?.trim()) { this._geminiState = 'skip'; return false; }
    try {
      const { GoogleGenerativeAI } = await import('https://esm.sh/@google/generative-ai');
      this.geminiModel = new GoogleGenerativeAI(key.trim()).getGenerativeModel({ model: 'gemini-2.5-flash' });
      this._geminiState = 'ready';
      return true;
    } catch (err) {
      console.warn('[LensModule] Gemini init failed:', err);
      this._geminiState = 'skip';
      return false;
    }
  }

  async _startChat(names, metricsMap) {
    this.activeChat  = null;
    this.chatHistory = [];

    const hasGemini = await this._ensureGemini();
    if (!hasGemini) {
      const text = names.map(fallbackText).join('\n\n');
      this.chatHistory.push({ role: 'ai', text });
      this._appendMessage('ai', this._textToHtml(text));
      this._setInputEnabled(true);
      return;
    }

    const systemPrompt = this._buildSystemPrompt(names, metricsMap);
    const firstUserMsg = `Please provide an initial clinical analysis of the selected structure(s): ${names.join(', ')}.`;

    try {
      // Start a stateful multi-turn chat; bake mesh metrics into history as context
      this.activeChat = this.geminiModel.startChat({
        history: [
          { role: 'user',  parts: [{ text: systemPrompt }] },
          { role: 'model', parts: [{ text: 'Understood. I have the mesh metrics and will ground all analysis in these measurements.' }] },
        ],
      });
      const result = await this._sendWithRetry(firstUserMsg);
      const text = result.response.text();
      this.chatHistory.push({ role: 'ai', text });
      this._appendMessage('ai', this._textToHtml(text));
      this._setInputEnabled(true);
    } catch (err) {
      console.warn('[LensModule] Chat start failed:', err);
      const fallback = names.map(fallbackText).join('\n\n');
      this.chatHistory.push({ role: 'ai', text: fallback });
      this._appendMessage('ai', this._textToHtml(fallback));
      this._setInputEnabled(true);
    }
  }

  async _sendFollowUp(userText) {
    if (!userText.trim() || !this.popup) return;
    this._setInputEnabled(false);
    this.chatHistory.push({ role: 'user', text: userText });
    this._appendMessage('user', `<p>${this._escapeHtml(userText)}</p>`);
    const thinkingId = 'lens-thinking-' + Date.now();
    this._appendMessage('ai', '<p class="lens-loading">Thinking\u2026</p>', thinkingId);

    try {
      let text;
      if (this.activeChat) {
        const result = await this._sendWithRetry(userText);
        text = result.response.text();
      } else {
        // Fallback — no session
        text = 'Gemini is not connected. ' + names.map(fallbackText).join(' ');
      }
      this.chatHistory.push({ role: 'ai', text });
      const el = this.popup?.querySelector(`#${thinkingId}`);
      if (el) { el.innerHTML = this._textToHtml(text); el.className = 'lens-msg-bubble'; }
    } catch (err) {
      const el = this.popup?.querySelector(`#${thinkingId}`);
      if (el) { el.textContent = 'Error: ' + (err.message || 'request failed'); }
    } finally {
      this._setInputEnabled(true);
      this._scrollThread();
    }
  }

  async _sendWithRetry(text, retries = 3) {
    for (let i = 0; i < retries; i++) {
      try {
        return await this.activeChat.sendMessage(text);
      } catch (err) {
        const is429 = err.status === 429 || String(err.message).includes('429');
        if (is429 && i < retries - 1) {
          await new Promise(r => setTimeout(r, 1800 * (i + 1)));
        } else throw err;
      }
    }
  }

  // ── Popup ─────────────────────────────────────────────────────────────────────

  _showPopup(clientX, clientY, names, metricsMap, _unused) {
    this._dismissPopup();
    this.chatHistory = [];

    const popup = document.createElement('div');
    popup.className = 'lens-popup';

    if (!names.length) {
      popup.innerHTML = `
        <div class="lens-popup-header">
          <span class="lens-popup-icon">🔍</span>
          <span class="lens-popup-title">No structures found</span>
          <button class="lens-popup-close" aria-label="Close">&#x2715;</button>
        </div>
        <div class="lens-chat-thread">
          <p class="lens-no-hit">No cardiac structures detected in the selected region.<br>Try selecting directly over the heart mesh.</p>
        </div>`;
    } else {
      const tags = names.map(n => `<span class="lens-tag">${n}</span>`).join(' ');

      // Inline metric snapshot for display
      let metricRows = '';
      if (metricsMap) {
        for (const name of names) {
          const m = metricsMap.get(name);
          if (!m) continue;
          const p = m.patient, r = m.reference;
          if (p || r) {
            const pv = p?.volMl ?? 0, rv = r?.volMl ?? 0;
          const volLine = (pv > 0 && rv > 0)
            ? `${pv.toFixed(1)} mL vs ${rv.toFixed(1)} mL ref (${pctDiff(pv, rv)})`
            : pv > 0 ? `${pv.toFixed(1)} mL` : rv > 0 ? `ref ${rv.toFixed(1)} mL` : '0 mL (upload a scan)';
            metricRows += `<div class="lens-metric-row"><span class="lens-metric-name">${name}</span><span class="lens-metric-val">${volLine}</span></div>`;
          }
        }
      }

      popup.innerHTML = `
        <div class="lens-popup-header">
          <span class="lens-popup-icon">🔍</span>
          <span class="lens-popup-title">${names.join(', ')}</span>
          <button class="lens-save-note" title="Save conversation as sticky note">📌</button>
          <button class="lens-popup-close" aria-label="Close">&#x2715;</button>
        </div>
        <div class="lens-tags-row">${tags}</div>
        ${metricRows ? `<div class="lens-metric-block">${metricRows}</div>` : ''}
        <div class="lens-chat-thread" id="lens-chat-thread">
          <div class="lens-msg lens-msg-ai">
            <span class="lens-msg-label">AI</span>
            <div class="lens-msg-bubble"><p class="lens-loading">Analyzing\u2026</p></div>
          </div>
        </div>
        <div class="lens-chat-input-row">
          <input type="text" class="lens-chat-input" placeholder="Ask a follow-up question\u2026" disabled>
          <button class="lens-chat-send" disabled aria-label="Send">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M6 1l5 10-5-3-5 3z"/></svg>
          </button>
        </div>`;
    }

    Object.assign(popup.style, {
      left: `${Math.min(clientX + 14, window.innerWidth  - 336)}px`,
      top:  `${Math.min(clientY + 14, window.innerHeight - 80)}px`,
    });
    document.body.appendChild(popup);
    this.popup = popup;

    popup.querySelector('.lens-popup-close')?.addEventListener('click', () => this._dismissPopup());

    popup.querySelector('.lens-save-note')?.addEventListener('click', () => {
      const lines = [`[Lens Analysis: ${names.join(', ')}]\n`];
      for (const { role, text } of this.chatHistory) {
        lines.push(`${role === 'ai' ? 'AI' : 'You'}: ${text}`);
      }
      const noteText = lines.join('\n\n');
      const pr = popup.getBoundingClientRect();
      window.dispatchEvent(new CustomEvent('lens:save-note', {
        detail: { text: noteText, clientX: pr.left, clientY: pr.top },
      }));
    });

    const input = popup.querySelector('.lens-chat-input');
    const sendBtn = popup.querySelector('.lens-chat-send');
    const doSend = () => {
      const v = input?.value?.trim();
      if (!v) return;
      if (input) input.value = '';
      this._sendFollowUp(v);
    };
    sendBtn?.addEventListener('click', doSend);
    input?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); } });

    this._makeDraggable(popup);
  }

  _appendMessage(role, innerHtml, id) {
    const thread = this.popup?.querySelector('#lens-chat-thread');
    if (!thread) return;
    // If this is the very first AI message, fill the placeholder
    if (role === 'ai' && thread.querySelector('.lens-loading')) {
      const bubble = thread.querySelector('.lens-msg-bubble');
      if (bubble && !id) { bubble.innerHTML = innerHtml; return; }
    }
    const wrap = document.createElement('div');
    wrap.className = `lens-msg lens-msg-${role}`;
    if (id) wrap.id = id;
    wrap.innerHTML = role === 'ai'
      ? `<span class="lens-msg-label">AI</span><div class="lens-msg-bubble">${innerHtml}</div>`
      : `<div class="lens-msg-bubble lens-msg-bubble-user">${innerHtml}</div>`;
    thread.appendChild(wrap);
    this._scrollThread();
  }

  _scrollThread() {
    const thread = this.popup?.querySelector('#lens-chat-thread');
    if (thread) thread.scrollTop = thread.scrollHeight;
  }

  _setInputEnabled(on) {
    const input   = this.popup?.querySelector('.lens-chat-input');
    const sendBtn = this.popup?.querySelector('.lens-chat-send');
    if (input)   { input.disabled   = !on; if (on) input.focus(); }
    if (sendBtn) { sendBtn.disabled = !on; }
  }

  _textToHtml(text) {
    return text.split('\n\n').filter(Boolean)
      .map(p => `<p>${this._escapeHtml(p)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g,     '<em>$1</em>')
      }</p>`)
      .join('');
  }

  _escapeHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  _makeDraggable(popup) {
    const header = popup.querySelector('.lens-popup-header');
    header.addEventListener('mousedown', e => {
      if (e.button !== 0 || e.target.closest('.lens-popup-close, .lens-save-note')) return;
      e.preventDefault();
      const r = popup.getBoundingClientRect();
      let ox = e.clientX - r.left, oy = e.clientY - r.top;
      const onMove = ev => {
        popup.style.left = `${Math.max(0, Math.min(ev.clientX - ox, window.innerWidth  - 300))}px`;
        popup.style.top  = `${Math.max(0, Math.min(ev.clientY - oy, window.innerHeight - 50))}px`;
      };
      const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  }

  _dismissPopup() {
    if (this.popup?.parentNode) this.popup.parentNode.removeChild(this.popup);
    this.popup      = null;
    this.activeChat = null;
  }

  // ── Styles ────────────────────────────────────────────────────────────────────

  _injectStyles() {
    if (document.getElementById('lens-module-styles')) return;
    const s = document.createElement('style');
    s.id = 'lens-module-styles';
    s.textContent = `
      .lens-popup {
        position: fixed; width: 320px; max-height: 480px;
        display: flex; flex-direction: column;
        background: rgba(14,14,20,0.98);
        border: 1px solid rgba(59,130,246,0.42);
        border-radius: 10px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(59,130,246,0.06);
        z-index: 1000; overflow: hidden;
        font-family: 'Outfit', system-ui, sans-serif;
        color: #ccc8c2; user-select: text;
      }
      .lens-popup-header {
        display: flex; align-items: center; gap: 0.35rem;
        padding: 0.44rem 0.6rem;
        background: rgba(59,130,246,0.1);
        border-bottom: 1px solid rgba(59,130,246,0.18);
        cursor: move; user-select: none; flex-shrink: 0;
      }
      .lens-popup-icon { font-size: 0.8rem; flex-shrink: 0; }
      .lens-popup-title {
        flex: 1; font-size: 0.68rem; font-weight: 700; color: #93c5fd;
        letter-spacing: 0.01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .lens-save-note, .lens-popup-close {
        flex-shrink: 0; background: none; border: none; color: #555;
        font-size: 0.85rem; cursor: pointer; padding: 0; line-height: 1;
        width: 20px; height: 20px; display: flex; align-items: center;
        justify-content: center; border-radius: 4px; transition: background 0.15s, color 0.15s;
      }
      .lens-save-note:hover { background: rgba(255,255,255,0.08); color: #fbbf24; }
      .lens-popup-close:hover { background: rgba(255,255,255,0.08); color: #fff; }
      .lens-tags-row {
        padding: 0.35rem 0.6rem 0.1rem; flex-shrink: 0;
        display: flex; flex-wrap: wrap; gap: 0.15rem;
      }
      .lens-tag {
        display: inline-block; background: rgba(59,130,246,0.18); color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.35); border-radius: 4px;
        padding: 0.06rem 0.32rem; font-size: 0.62rem; font-weight: 700;
      }
      .lens-metric-block {
        padding: 0.25rem 0.6rem 0.3rem; border-bottom: 1px solid rgba(255,255,255,0.05);
        flex-shrink: 0;
      }
      .lens-metric-row {
        display: flex; justify-content: space-between; align-items: baseline;
        font-size: 0.63rem; padding: 0.08rem 0; gap: 0.5rem;
      }
      .lens-metric-name { color: #6b9eff; font-weight: 600; flex-shrink: 0; }
      .lens-metric-val  { color: #888; text-align: right; }
      .lens-chat-thread {
        flex: 1; overflow-y: auto; padding: 0.5rem 0.6rem;
        display: flex; flex-direction: column; gap: 0.55rem;
        min-height: 80px;
      }
      .lens-chat-thread::-webkit-scrollbar { width: 3px; }
      .lens-chat-thread::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
      .lens-msg { display: flex; gap: 0.3rem; align-items: flex-start; }
      .lens-msg-ai  { flex-direction: row; }
      .lens-msg-user { flex-direction: row-reverse; }
      .lens-msg-label {
        flex-shrink: 0; font-size: 0.58rem; font-weight: 800; color: #3b82f6;
        background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
        border-radius: 3px; padding: 0.08rem 0.28rem; margin-top: 0.05rem;
        letter-spacing: 0.04em;
      }
      .lens-msg-bubble {
        background: rgba(255,255,255,0.04); border-radius: 6px;
        padding: 0.4rem 0.5rem; flex: 1; min-width: 0;
      }
      .lens-msg-bubble-user {
        background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.2);
      }
      .lens-msg-bubble p {
        margin: 0 0 0.4rem; line-height: 1.58; color: #b4b0aa; font-size: 0.69rem;
      }
      .lens-msg-bubble p:last-child { margin-bottom: 0; }
      .lens-msg-bubble-user p { color: #c8d8f8; }
      .lens-loading { color: #555; font-size: 0.68rem; font-style: italic; margin: 0 !important; }
      .lens-no-hit  { color: #555; font-size: 0.7rem;  margin: 0; line-height: 1.55; }
      .lens-chat-input-row {
        display: flex; gap: 0.3rem; padding: 0.4rem 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.06); flex-shrink: 0;
        background: rgba(0,0,0,0.2);
      }
      .lens-chat-input {
        flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 5px; color: #d0cec9; font-family: inherit; font-size: 0.69rem;
        padding: 0.35rem 0.5rem; outline: none; transition: border-color 0.15s;
      }
      .lens-chat-input:focus { border-color: rgba(59,130,246,0.5); }
      .lens-chat-input:disabled { opacity: 0.4; cursor: not-allowed; }
      .lens-chat-input::placeholder { color: #444; }
      .lens-chat-send {
        flex-shrink: 0; width: 28px; height: 28px; border-radius: 5px; border: none;
        background: #2563eb; color: #fff; cursor: pointer; display: flex;
        align-items: center; justify-content: center; transition: background 0.15s;
      }
      .lens-chat-send:hover:not(:disabled) { background: #3b82f6; }
      .lens-chat-send:disabled { opacity: 0.35; cursor: not-allowed; }
    `;
    document.head.appendChild(s);
  }
}
