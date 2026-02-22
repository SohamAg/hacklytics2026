/**
 * Google Lens-style feature for Three.js Canvas
 * Allows selecting parts of the canvas and getting AI-powered insights
 */

import * as THREE from 'three';

export class LensModule {
  constructor(renderer, camera, scene, pickableMeshes) {
    this.renderer = renderer;
    this.camera = camera;
    this.scene = scene;
    this.pickableMeshes = pickableMeshes;
    
    this.active = false;
    this.isSelecting = false;
    this.startX = 0;
    this.startY = 0;
    this.endX = 0;
    this.endY = 0;
    
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    
    this.selectionCanvas = null;
    this.selectionCtx = null;
    this.selectedMeshes = [];
    this.ragData = null;
    this.geminiKey = null;
    this.lensBtn = null;
    
    this.init();
  }

  init() {
    // Create overlay canvas for selection visualization
    this.selectionCanvas = document.createElement('canvas');
    this.selectionCanvas.id = 'lens-selection-canvas';
    this.selectionCanvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      background: transparent;
      cursor: crosshair;
      z-index: 15;
      display: none;
    `;
    
    const container = document.getElementById('canvas-container');
    if (container) {
      container.appendChild(this.selectionCanvas);
      this.resizeSelectionCanvas();
    }
    
    this.selectionCtx = this.selectionCanvas.getContext('2d');
    
    // Event listeners
    this.selectionCanvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.selectionCanvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.selectionCanvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
    
    window.addEventListener('resize', () => this.resizeSelectionCanvas());
    
    // ESC key to exit lens mode
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.active) {
        this.deactivate();
      }
    });
  }

  resizeSelectionCanvas() {
    if (!this.selectionCanvas) return;
    const container = this.selectionCanvas.parentElement;
    this.selectionCanvas.width = container.clientWidth;
    this.selectionCanvas.height = container.clientHeight;
  }

  async setGeminiKey(key) {
    this.geminiKey = key;
  }

  async loadRAGData(csvUrls) {
    try {
      this.ragData = {};
      for (const url of csvUrls) {
        const response = await fetch(url);
        const csvText = await response.text();
        const fileName = url.split('/').pop();
        this.ragData[fileName] = this.parseCSV(csvText);
      }
      console.log('RAG data loaded:', this.ragData);
    } catch (error) {
      console.error('Failed to load RAG data:', error);
    }
  }

  parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
      const row = {};
      const values = lines[i].split(',');
      headers.forEach((header, idx) => {
        row[header] = values[idx]?.trim() || '';
      });
      data.push(row);
    }
    
    return data;
  }

  toggle() {
    this.active = !this.active;
    if (this.selectionCanvas) {
      this.selectionCanvas.style.display = this.active ? 'block' : 'none';
    }
    return this.active;
  }

  deactivate() {
    if (!this.active) return;
    this.active = false;
    if (this.selectionCanvas) {
      this.selectionCanvas.style.display = 'none';
      // Clear any ongoing selection
      this.isSelecting = false;
      this.selectionCtx.clearRect(0, 0, this.selectionCanvas.width, this.selectionCanvas.height);
    }
    // Update button state
    if (this.lensBtn) {
      this.lensBtn.classList.remove('active');
      this.lensBtn.setAttribute('aria-pressed', 'false');
      this.lensBtn.textContent = '🔍 Lens';
    }
  }

  onMouseDown(e) {
    if (!this.active) return;
    
    this.isSelecting = true;
    const rect = this.selectionCanvas.getBoundingClientRect();
    this.startX = e.clientX - rect.left;
    this.startY = e.clientY - rect.top;
  }

  onMouseMove(e) {
    if (!this.isSelecting) return;
    
    const rect = this.selectionCanvas.getBoundingClientRect();
    this.endX = e.clientX - rect.left;
    this.endY = e.clientY - rect.top;
    
    this.drawSelection();
  }

  drawSelection() {
    if (!this.selectionCtx) return;
    
    // Clear canvas
    this.selectionCtx.clearRect(0, 0, this.selectionCanvas.width, this.selectionCanvas.height);
    
    // Draw selection rectangle
    const minX = Math.min(this.startX, this.endX);
    const minY = Math.min(this.startY, this.endY);
    const width = Math.abs(this.endX - this.startX);
    const height = Math.abs(this.endY - this.startY);
    
    this.selectionCtx.strokeStyle = 'rgba(126, 184, 255, 0.8)';
    this.selectionCtx.lineWidth = 2;
    this.selectionCtx.strokeRect(minX, minY, width, height);
    
    // Draw semi-transparent fill
    this.selectionCtx.fillStyle = 'rgba(126, 184, 255, 0.1)';
    this.selectionCtx.fillRect(minX, minY, width, height);
    
    // Draw corner dots
    this.selectionCtx.fillStyle = 'rgba(126, 184, 255, 0.8)';
    this.selectionCtx.beginPath();
    this.selectionCtx.arc(minX, minY, 3, 0, Math.PI * 2);
    this.selectionCtx.fill();
    this.selectionCtx.beginPath();
    this.selectionCtx.arc(minX + width, minY, 3, 0, Math.PI * 2);
    this.selectionCtx.fill();
    this.selectionCtx.beginPath();
    this.selectionCtx.arc(minX, minY + height, 3, 0, Math.PI * 2);
    this.selectionCtx.fill();
    this.selectionCtx.beginPath();
    this.selectionCtx.arc(minX + width, minY + height, 3, 0, Math.PI * 2);
    this.selectionCtx.fill();
  }

  async onMouseUp(e) {
    if (!this.isSelecting) return;
    
    this.isSelecting = false;
    this.selectionCtx.clearRect(0, 0, this.selectionCanvas.width, this.selectionCanvas.height);
    
    // Get selected meshes via raycasting
    const minX = Math.min(this.startX, this.endX);
    const minY = Math.min(this.startY, this.endY);
    const maxX = Math.max(this.startX, this.endX);
    const maxY = Math.max(this.startY, this.endY);
    
    this.selectedMeshes = this.pickMeshesByArea(minX, minY, maxX, maxY);
    
    if (this.selectedMeshes.length > 0) {
      const info = this.getSelectedMeshesInfo();
      await this.showResultsWithAI(info, e.clientX, e.clientY);
    }
  }

  pickMeshesByArea(minX, minY, maxX, maxY) {
    const selected = [];
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    // Cast rays from multiple points in the selection area
    const points = [
      [centerX, centerY],
      [minX, minY],
      [maxX, maxY],
      [minX, maxY],
      [maxX, minY],
    ];
    
    const hitMeshes = new Set();
    
    for (const [x, y] of points) {
      this.mouse.x = (x / this.selectionCanvas.width) * 2 - 1;
      this.mouse.y = -(y / this.selectionCanvas.height) * 2 + 1;
      
      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects(this.pickableMeshes);
      
      for (const hit of intersects) {
        hitMeshes.add(hit.object);
      }
    }
    
    return Array.from(hitMeshes).slice(0, 5); // Limit to 5 meshes
  }

  getSelectedMeshesInfo() {
    const info = [];
    for (const mesh of this.selectedMeshes) {
      const name = mesh.name?.trim() || 'Unknown part';
      const geometry = mesh.geometry;
      const vertices = geometry?.attributes?.position?.count || 0;
      
      info.push({
        name,
        vertices,
        material: mesh.material?.color?.getHexString?.() || 'default',
      });
    }
    return {
      count: this.selectedMeshes.length,
      parts: info,
    };
  }

  async showResultsWithAI(info, cursorX, cursorY) {
    const popup = this.createPopup(cursorX, cursorY);
    popup.classList.add('loading');
    popup.querySelector('.popup-content p').textContent = 'Analyzing...';
    
    document.body.appendChild(popup);
    
    try {
      const analysis = await this.analyzeWithGemini(info);
      popup.classList.remove('loading');
      popup.querySelector('.popup-content p').innerHTML = this.formatAnalysis(analysis);
    } catch (error) {
      console.error('AI analysis failed:', error);
      popup.classList.remove('loading');
      popup.querySelector('.popup-content p').innerHTML = `
        <strong>Selection Analysis</strong><br>
        ${info.parts.map(p => `<strong>${p.name}</strong><br>Vertices: ${p.vertices}`).join('<br>')}
        <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);">
        <em>AI analysis unavailable. Make sure GEMINI_API_KEY is set.</em>
      `;
    }
  }

  async analyzeWithGemini(selectionInfo) {
    if (!this.geminiKey) {
      // Fallback analysis
      return this.generateBasicAnalysis(selectionInfo);
    }
    
    const context = this.buildRAGContext();
    const prompt = this.buildPrompt(selectionInfo, context);
    
    try {
      const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=' + this.geminiKey, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: prompt }]
          }],
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 256,
          }
        })
      });
      
      const data = await response.json();
      if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
        return data.candidates[0].content.parts[0].text;
      }
      return this.generateBasicAnalysis(selectionInfo);
    } catch (error) {
      console.error('Gemini API error:', error);
      return this.generateBasicAnalysis(selectionInfo);
    }
  }

  buildRAGContext() {
    if (!this.ragData) return '';
    
    let context = 'Available data:\n';
    for (const [fileName, data] of Object.entries(this.ragData)) {
      context += `\n${fileName}:\n`;
      if (Array.isArray(data)) {
        context += `Rows: ${data.length}\n`;
        if (data.length > 0) {
          const firstRow = data[0];
          context += `Columns: ${Object.keys(firstRow).join(', ')}\n`;
          // Add sample data points
          for (let i = 0; i < Math.min(2, data.length); i++) {
            context += `Sample: ${JSON.stringify(data[i])}\n`;
          }
        }
      }
    }
    
    return context;
  }

  buildPrompt(selectionInfo, ragContext) {
    const partsDesc = selectionInfo.parts
      .map(p => `- ${p.name} (${p.vertices} vertices)`)
      .join('\n');
    
    return `You are a cardiac anatomy expert analyzing a 3D heart model visualization.

The user has selected the following parts:
${partsDesc}

${ragContext}

Provide a brief (2-3 sentences), informative analysis about the selected heart parts. Include:
1. What anatomical structure this is
2. Its clinical significance
3. Any relevant observations from the available data

Keep the response concise and educational.`;
  }

  generateBasicAnalysis(selectionInfo) {
    const partsText = selectionInfo.parts
      .map(p => `\n<strong>${p.name}</strong>\nVertices: ${p.vertices}`)
      .join('\n');
    
    const cardiacParts = {
      'atrium': 'Upper chamber that receives blood',
      'ventricle': 'Lower chamber that pumps blood',
      'valve': 'Controls blood flow through chambers',
      'vessel': 'Blood vessel carrying blood',
      'apex': 'Pointed bottom tip of the heart',
      'septum': 'Wall dividing left and right chambers',
    };
    
    let description = '<strong>Selected Parts:</strong>';
    for (const part of selectionInfo.parts) {
      description += `\n<strong>${part.name}</strong>`;
      for (const [key, val] of Object.entries(cardiacParts)) {
        if (part.name.toLowerCase().includes(key)) {
          description += `\n${val}`;
          break;
        }
      }
      description += `\nGeometry: ${part.vertices} vertices`;
    }
    
    return description;
  }

  formatAnalysis(analysis) {
    return `
      <strong>AI Analysis</strong>
      <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);">
      <p style="margin: 0.5rem 0;">${analysis}</p>
    `;
  }

  createPopup(cursorX, cursorY) {
    const popup = document.createElement('div');
    popup.className = 'lens-popup';
    popup.innerHTML = `
      <div class="popup-content">
        <div class="popup-header">
          <h3>Selection Analysis</h3>
          <button class="popup-close">&times;</button>
        </div>
        <p>Loading...</p>
      </div>
    `;
    
    const closeBtn = popup.querySelector('.popup-close');
    closeBtn.addEventListener('click', () => popup.remove());
    
    // Position near cursor with some offset
    const offsetX = 15;
    const offsetY = 15;
    popup.style.left = (cursorX + offsetX) + 'px';
    popup.style.top = (cursorY + offsetY) + 'px';
    
    return popup;
  }
}
