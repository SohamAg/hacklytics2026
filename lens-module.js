/**
 * Google Lens-style feature for Three.js Canvas
 * Allows selecting parts of the canvas and getting AI-powered insights
 */

import * as THREE from 'three';
import { GoogleGenerativeAI } from '@google/generative-ai';

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
    this.geminiModel = null;
    this.lensBtn = null;

    // Rate limit / queue management
    this.requestQueue = Promise.resolve();
    this.analysisCache = new Map();
    
    this.initCanvas();
    this.initGemini();
  }

  initCanvas() {
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
    
    this.selectionCanvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.selectionCanvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.selectionCanvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
    
    window.addEventListener('resize', () => this.resizeSelectionCanvas());
    
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.active) {
        this.deactivate();
      }
    });
  }

  initGemini() {
    const apiKey = prompt('Enter your Gemini API key (get one from https://aistudio.google.com/apikey):');
    if (apiKey) {
      this.setGeminiKey(apiKey);
    }
  }

  setGeminiKey(key) {
    if (!key || typeof key !== 'string' || key.trim().length === 0) {
      console.warn('Invalid API key');
      return false;
    }
    
    try {
      const genAI = new GoogleGenerativeAI(key);
      // Using gemini-2.0-flash-lite for higher free-tier rate limits
      this.geminiModel = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
      console.log('Gemini API initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize Gemini API:', error);
      this.geminiModel = null;
      return false;
    }
  }

  resizeSelectionCanvas() {
    if (!this.selectionCanvas) return;
    const container = this.selectionCanvas.parentElement;
    this.selectionCanvas.width = container.clientWidth;
    this.selectionCanvas.height = container.clientHeight;
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
      this.isSelecting = false;
      this.selectionCtx.clearRect(0, 0, this.selectionCanvas.width, this.selectionCanvas.height);
    }
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
    
    this.selectionCtx.clearRect(0, 0, this.selectionCanvas.width, this.selectionCanvas.height);
    
    const minX = Math.min(this.startX, this.endX);
    const minY = Math.min(this.startY, this.endY);
    const width = Math.abs(this.endX - this.startX);
    const height = Math.abs(this.endY - this.startY);
    
    this.selectionCtx.strokeStyle = 'rgba(126, 184, 255, 0.8)';
    this.selectionCtx.lineWidth = 2;
    this.selectionCtx.strokeRect(minX, minY, width, height);
    
    this.selectionCtx.fillStyle = 'rgba(126, 184, 255, 0.1)';
    this.selectionCtx.fillRect(minX, minY, width, height);
    
    this.selectionCtx.fillStyle = 'rgba(126, 184, 255, 0.8)';
    [
      [minX, minY],
      [minX + width, minY],
      [minX, minY + height],
      [minX + width, minY + height],
    ].forEach(([x, y]) => {
      this.selectionCtx.beginPath();
      this.selectionCtx.arc(x, y, 3, 0, Math.PI * 2);
      this.selectionCtx.fill();
    });
  }

  async onMouseUp(e) {
    if (!this.isSelecting) return;
    
    this.isSelecting = false;
    this.selectionCtx.clearRect(0, 0, this.selectionCanvas.width, this.selectionCanvas.height);
    
    const minX = Math.min(this.startX, this.endX);
    const minY = Math.min(this.startY, this.endY);
    const maxX = Math.max(this.startX, this.endX);
    const maxY = Math.max(this.startY, this.endY);
    
    this.selectedMeshes = this.pickMeshesByArea(minX, minY, maxX, maxY);
    
    if (this.selectedMeshes.length > 0) {
      const info = this.getSelectedMeshesInfo();
      this.highlightSelectedMeshes();
      await this.showResults(info, e.clientX, e.clientY);
    }
  }

  pickMeshesByArea(minX, minY, maxX, maxY) {
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
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
    
    return Array.from(hitMeshes).slice(0, 5);
  }

  getSelectedMeshesInfo() {
    return {
      count: this.selectedMeshes.length,
      parts: this.selectedMeshes.map(mesh => ({
        name: mesh.name?.trim() || 'Unknown part',
        vertices: mesh.geometry?.attributes?.position?.count || 0,
      })),
    };
  }

  highlightSelectedMeshes() {
    // Store original materials
    const originalMaterials = new Map();
    
    for (const mesh of this.selectedMeshes) {
      originalMaterials.set(mesh, mesh.material);
      
      // Create highlight material
      const highlightMaterial = new THREE.MeshPhongMaterial({
        color: 0x7eb8ff,
        emissive: 0x4da6ff,
        emissiveIntensity: 0.5,
        wireframe: false,
      });
      
      mesh.material = highlightMaterial;
    }
    
    // Restore original materials after 2 seconds
    setTimeout(() => {
      for (const mesh of this.selectedMeshes) {
        if (originalMaterials.has(mesh)) {
          mesh.material = originalMaterials.get(mesh);
        }
      }
    }, 2000);
    
    // Request render
    this.renderer.render(this.scene, this.camera);
  }

  async showResults(info, cursorX, cursorY) {
    const popup = this.createPopup(cursorX, cursorY);
    popup.classList.add('loading');
    popup.querySelector('.popup-content p').textContent = 'Analyzing...';
    document.body.appendChild(popup);
    
    try {
      let analysis;
      if (this.geminiModel) {
        analysis = await this.queueAnalysis(info);
      } else {
        analysis = this.generateFallbackAnalysis(info);
      }
      popup.classList.remove('loading');
      popup.querySelector('.popup-content p').innerHTML = analysis;
    } catch (error) {
      console.error('Analysis error:', error);
      popup.classList.remove('loading');
      popup.querySelector('.popup-content p').innerHTML = `
        <strong>Selection Analysis</strong><br>
        ${info.parts.map(p => `<strong>${p.name}</strong><br>Vertices: ${p.vertices}`).join('<br>')}
        <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);">
        <em>AI analysis unavailable. Check your API key.</em>
      `;
    }
  }

  /**
   * Queues requests so they run one at a time,
   * preventing bursts that trigger 429 rate limit errors.
   */
  queueAnalysis(info) {
    this.requestQueue = this.requestQueue.then(() => this.analyzeWithGemini(info));
    return this.requestQueue;
  }

  /**
   * Calls Gemini with retry + exponential backoff on 429 errors.
   * Also caches results so identical selections skip the API entirely.
   */
  async analyzeWithGemini(info, retries = 3, baseDelay = 1000) {
    // Build a stable cache key from sorted part names
    const cacheKey = info.parts.map(p => p.name).sort().join('|');
    if (this.analysisCache.has(cacheKey)) {
      return this.analysisCache.get(cacheKey);
    }

    const partsDesc = info.parts
      .map(p => `- ${p.name} (${p.vertices} vertices)`)
      .join('\n');

    const prompt = `Analyze these selected heart parts:
${partsDesc}

Provide a 2 sentence analysis about what these structures are and their clinical significance.`;

    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        const result = await this.geminiModel.generateContent({
          contents: [{ parts: [{ text: prompt }] }]
        });

        const text = result.response.text();
        const html = `<strong>AI Analysis</strong>
          <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);">
          <p style="margin: 0.5rem 0;">${text}</p>`;

        // Cache the successful response
        this.analysisCache.set(cacheKey, html);
        return html;

      } catch (error) {
        const isRateLimit = error?.status === 429 || error?.message?.includes('429');
        const isLastAttempt = attempt === retries - 1;

        if (isRateLimit && !isLastAttempt) {
          const waitMs = baseDelay * Math.pow(2, attempt); // 1s, 2s, 4s
          console.warn(`Rate limit hit. Retrying in ${waitMs}ms... (attempt ${attempt + 1}/${retries})`);
          await new Promise(res => setTimeout(res, waitMs));
          continue;
        }

        console.error('Gemini error:', error);
        return this.generateFallbackAnalysis(info);
      }
    }
  }

  generateFallbackAnalysis(info) {
    const cardiacParts = {
      'atrium': 'Upper chamber that receives blood',
      'ventricle': 'Lower chamber that pumps blood',
      'valve': 'Controls blood flow between chambers',
      'vessel': 'Blood vessel carrying blood',
      'apex': 'Pointed bottom tip of the heart',
      'septum': 'Wall dividing left and right chambers',
    };
    
    let description = '<strong>Selected Parts:</strong>';
    for (const part of info.parts) {
      description += `<br><strong>${part.name}</strong>`;
      for (const [key, val] of Object.entries(cardiacParts)) {
        if (part.name.toLowerCase().includes(key)) {
          description += `<br>${val}`;
          break;
        }
      }
      description += `<br>Vertices: ${part.vertices}`;
    }
    return description;
  }

  createPopup(cursorX, cursorY) {
    const popup = document.createElement('div');
    popup.className = 'lens-popup';
    popup.innerHTML = `
      <div class="popup-content">
        <div class="popup-header">
          <h3>Analysis</h3>
          <button class="popup-close">&times;</button>
        </div>
        <p></p>
      </div>
    `;
    
    popup.querySelector('.popup-close').addEventListener('click', () => popup.remove());
    popup.style.left = (cursorX + 15) + 'px';
    popup.style.top = (cursorY + 15) + 'px';
    
    // Add drag functionality
    const header = popup.querySelector('.popup-header');
    let offsetX = 0;
    let offsetY = 0;
    let isDragging = false;
    
    header.addEventListener('mousedown', (e) => {
      isDragging = true;
      const rect = popup.getBoundingClientRect();
      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;
      header.style.cursor = 'grabbing';
    });
    
    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      popup.style.left = (e.clientX - offsetX) + 'px';
      popup.style.top = (e.clientY - offsetY) + 'px';
    });
    
    document.addEventListener('mouseup', () => {
      isDragging = false;
      header.style.cursor = 'grab';
    });
    
    header.style.cursor = 'grab';
    
    return popup;
  }
}