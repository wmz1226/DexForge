'use strict';

class MotionWall {
  constructor(videoURL, onClose) {
    this.videoURL = videoURL;
    this.onClose = onClose;
    this.active = false;
  }
  open(source, opener, paused) {
    if (this.active || !source) return;
    this.active = true;
    this.opener = opener;
    this.source = source;
    this.paused = paused;
    this.overlay = document.createElement('div');
    this.overlay.className = 'motion-wall';
    this.overlay.setAttribute('role', 'dialog');
    this.overlay.setAttribute('aria-modal', 'true');
    this.overlay.setAttribute('aria-label', 'Video panorama');
    this.canvas = document.createElement('canvas');
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('aria-label', 'Drag to pan; pinch or Control-scroll to zoom');
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    const close = document.createElement('button');
    close.className = 'wall-close';
    close.type = 'button';
    close.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M8 3v5H3m9-5v5h5M8 17v-5H3m9 5v-5h5"/></svg>Minimize';
    close.addEventListener('click', () => this.close());
    this.video = document.createElement('video');
    this.video.className = 'wall-source';
    this.video.hidden = true;
    this.video.muted = this.video.loop = this.video.playsInline = true;
    this.video.preload = 'auto';
    this.video.src = this.videoURL(source.video);
    this.poster = new Image();
    this.poster.onload = () => { if (this.active) this.draw(); };
    this.poster.src = source.poster;
    this.overlay.append(this.canvas, this.video, close);
    document.body.append(this.overlay);
    document.querySelector('main').inert = true;
    document.querySelector('#motion-toggle').inert = true;
    document.body.classList.add('wall-open');
    opener.setAttribute('aria-expanded', 'true');
    this.resize = () => {
      this.width = innerWidth;
      this.height = innerHeight;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(this.width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.minimum = Math.min(this.width / source.width, this.height / source.height);
      this.scale = this.minimum;
      this.x = (this.width - source.width * this.scale) / 2;
      this.y = (this.height - source.height * this.scale) / 2;
      this.draw();
    };
    this.resize();
    window.addEventListener('resize', this.resize);
    this.overlay.addEventListener('keydown', event => {
      if (event.key === 'Escape') this.close();
      if (event.key === 'Tab') { event.preventDefault(); close.focus(); }
      if (event.key === ' ') { event.preventDefault(); this.setPaused(!this.paused); }
      const moves = { ArrowLeft: [60, 0], ArrowRight: [-60, 0], ArrowUp: [0, 60], ArrowDown: [0, -60] };
      if (moves[event.key]) { event.preventDefault(); this.x += moves[event.key][0]; this.y += moves[event.key][1]; this.draw(); }
      if (['+', '=', '-'].includes(event.key)) { event.preventDefault(); this.zoom(event.key === '-' ? .85 : 1.15, this.width / 2, this.height / 2); }
    });
    const pointers = new Map();
    const gesture = () => {
      const points = [...pointers.values()];
      return { x: points.reduce((s,p) => s+p.x, 0)/points.length, y: points.reduce((s,p) => s+p.y, 0)/points.length, distance: points.length > 1 ? Math.hypot(points[1].x-points[0].x, points[1].y-points[0].y) : 0 };
    };
    this.canvas.addEventListener('pointerdown', event => {
      if (event.button > 0) return;
      this.canvas.focus();
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.classList.add('dragging');
    });
    this.canvas.addEventListener('pointermove', event => {
      if (!pointers.has(event.pointerId)) return;
      const before = gesture();
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      const after = gesture();
      this.x += after.x - before.x;
      this.y += after.y - before.y;
      if (before.distance && after.distance) this.zoom(after.distance / before.distance, after.x, after.y);
      else this.draw();
    });
    const release = event => { pointers.delete(event.pointerId); if (!pointers.size) this.canvas.classList.remove('dragging'); };
    this.canvas.addEventListener('pointerup', release);
    this.canvas.addEventListener('pointercancel', release);
    this.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) this.zoom(Math.exp(-event.deltaY * .004), event.clientX, event.clientY);
      else { this.x -= event.deltaX; this.y -= event.deltaY; this.draw(); }
    }, { passive: false });
    this.video.addEventListener('loadeddata', () => this.draw());
    this.tick = () => {
      if (!this.active) return;
      if (!document.hidden && !this.video.paused) this.draw();
      this.animation = requestAnimationFrame(this.tick);
    };
    this.tick();
    this.setPaused(paused);
    close.focus();
  }
  zoom(factor, x, y) {
    const next = Math.max(this.minimum * .7, Math.min(this.minimum * 4, this.scale * factor));
    this.x = x - (x - this.x) * next / this.scale;
    this.y = y - (y - this.y) * next / this.scale;
    this.scale = next;
    this.draw();
  }
  draw() {
    if (!this.active) return;
    this.ctx.fillStyle = '#eae5eb';
    this.ctx.fillRect(0, 0, this.width, this.height);
    const image = this.video.readyState >= 2 ? this.video : this.poster.complete && this.poster.naturalWidth ? this.poster : null;
    if (!image) return;
    const width = this.source.width * this.scale, height = this.source.height * this.scale;
    this.x = ((this.x % width) + width) % width - width;
    this.y = ((this.y % height) + height) % height - height;
    for (let y = this.y; y < this.height; y += height) {
      for (let x = this.x; x < this.width; x += width) this.ctx.drawImage(image, x, y, width + .5, height + .5);
    }
  }
  setPaused(paused) {
    if (!this.active) return;
    this.paused = paused;
    this.syncPlayback();
  }
  syncPlayback() {
    if (!this.active) return;
    if (this.paused || document.hidden) this.video.pause();
    else this.video.play().catch(() => {});
  }
  close() {
    if (!this.active) return;
    this.active = false;
    cancelAnimationFrame(this.animation);
    window.removeEventListener('resize', this.resize);
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    this.overlay.remove();
    document.body.classList.remove('wall-open');
    document.querySelector('main').inert = false;
    document.querySelector('#motion-toggle').inert = false;
    this.opener.setAttribute('aria-expanded', 'false');
    this.opener.focus({ preventScroll: true });
    this.onClose();
  }
}
