'use strict';

const { hands, clips } = window.RESULTS;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const motionButton = document.querySelector('#motion-toggle');
const supportsMP4 = Boolean(document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"'));
const videoURL = url => supportsMP4 ? url : url.replace(/\.mp4$/, '.webm');
const visibleVideos = new Set();
let paused = reducedMotion.matches;
let zoom;
let warmedHD;
const warmQueue = [];
const queued = new WeakSet();
let warming = 0;
let warmTimer;
const wall = new MotionWall(videoURL, () => refreshPlayback());

function prepareVideo(video) {
  if (!video.isConnected || video.getAttribute('src')) return;
  video.preload = 'auto';
  video.src = video.dataset.src;
  if (!video.poster) video.poster = video.dataset.poster;
}
function pumpWarmQueue() {
  clearTimeout(warmTimer);
  if (document.hidden || wall.active || [...visibleVideos].some(v => v.isConnected && v.readyState < 3)) {
    warmTimer = setTimeout(pumpWarmQueue, 160);
    return;
  }
  while (warming < 3 && warmQueue.length) {
    const video = warmQueue.shift();
    queued.delete(video);
    if (!video.isConnected || video.getAttribute('src')) continue;
    warming++;
    let finished = false;
    const done = () => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      video.removeEventListener('canplaythrough', done);
      video.removeEventListener('error', done);
      warming--;
      pumpWarmQueue();
    };
    const timeout = setTimeout(done, 5000);
    video.addEventListener('canplaythrough', done, {once:true});
    video.addEventListener('error', done, {once:true});
    prepareVideo(video);
  }
}
function warmVideo(video) {
  if (video.getAttribute('src') || queued.has(video)) return;
  queued.add(video);
  warmQueue.push(video);
  clearTimeout(warmTimer);
  warmTimer = setTimeout(pumpWarmQueue, 120);
}
function warmHD(clip) {
  const src = videoURL(clip.hd);
  if (warmedHD?.dataset.src === src || zoom) return;
  if (warmedHD) { warmedHD.removeAttribute('src'); warmedHD.load(); }
  warmedHD = document.createElement('video');
  warmedHD.muted = warmedHD.loop = warmedHD.playsInline = true;
  warmedHD.preload = 'auto';
  warmedHD.dataset.src = src;
  warmedHD.src = src;
}

function updatePlayback(video) {
  if (paused || document.hidden || wall.active || !video.isConnected || (zoom?.preview === video && zoom.hdReady) || (!visibleVideos.has(video) && zoom?.preview !== video && zoom?.hd !== video)) {
    video.pause();
  } else {
    prepareVideo(video);
    if (video.paused) video.play().catch(() => {});
  }
}
const visibility = new IntersectionObserver(entries => {
  entries.forEach(({ target, isIntersecting, intersectionRatio }) => {
    if (isIntersecting && intersectionRatio >= .01 && target.isConnected) {
      visibleVideos.add(target);
      if (!target.poster) target.poster = target.dataset.poster;
    }
    else visibleVideos.delete(target);
  });
  refreshPlayback();
}, { threshold: .01 });
const approaching = new IntersectionObserver(entries => {
  entries.forEach(({target,isIntersecting}) => { if (isIntersecting) warmVideo(target); });
}, {rootMargin:'360px 0px',threshold:0});
function refreshPlayback() {
  document.querySelectorAll('video[data-src]').forEach(updatePlayback);
  wall.syncPlayback();
  motionButton.innerHTML = paused ? '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m7 4 9 6-9 6Z"/></svg>' : '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 4v12M13 4v12"/></svg>';
  motionButton.setAttribute('aria-label', paused ? 'Play videos' : 'Pause videos');
  motionButton.setAttribute('aria-pressed', String(paused));
}
motionButton.addEventListener('click', () => { paused = !paused; refreshPlayback(); });
document.addEventListener('visibilitychange', refreshPlayback);
reducedMotion.addEventListener('change', event => { paused = event.matches; wall.setPaused(paused); refreshPlayback(); });

function closeZoom() {
  if (!zoom) return;
  const current = zoom;
  zoom = null;
  if (current.hdReady) current.preview.currentTime = current.hd.currentTime;
  current.hd.pause();
  current.hd.removeAttribute('src');
  current.hd.load();
  current.card.prepend(current.preview);
  current.wrapper.remove();
  document.body.classList.remove('zoom-open');
  refreshPlayback();
  current.opener.focus({ preventScroll: true });
}
function openViewer(clip, card, opener) {
  closeZoom();
  const preview = card.querySelector('video');
  const box = card.getBoundingClientRect();
  const wrapper = document.createElement('div');
  wrapper.className = 'clip-zoom';
  wrapper.dataset.clipId = clip.id;
  wrapper.tabIndex = 0;
  wrapper.setAttribute('role', 'button');
  wrapper.setAttribute('aria-label', 'Close enlarged video');
  const scale = Math.min(2, (innerWidth - 24) / box.width, (innerHeight - 24) / box.height);
  const extra = box.width * (scale - 1) / 2;
  const extraY = box.height * (scale - 1) / 2;
  const left = Math.max(extra + 12, Math.min(box.left, innerWidth - box.width - extra - 12));
  const top = Math.max(extraY + 12, Math.min(box.top, innerHeight - box.height - extraY - 12));
  Object.assign(wrapper.style, { left: `${left + scrollX}px`, top: `${top + scrollY}px`, width: `${box.width}px`, height: `${box.height}px` });
  const hd = warmedHD?.dataset.src === videoURL(clip.hd) ? warmedHD : document.createElement('video');
  if (hd === warmedHD) warmedHD = null;
  hd.className = 'zoom-hd';
  hd.muted = hd.loop = hd.playsInline = true;
  hd.preload = 'auto';
  hd.dataset.src = videoURL(clip.hd);
  const current = { wrapper, card, preview, hd, opener, hdReady: false };
  zoom = current;
  const showHD = () => {
    if (zoom !== current) return;
    current.hdReady = true;
    wrapper.classList.add('hd-ready');
    updatePlayback(preview);
    updatePlayback(hd);
  };
  const synchronize = () => {
    if (zoom !== current) return;
    hd.currentTime = preview.currentTime;
  };
  hd.addEventListener('seeked', () => {
    if (zoom !== current || current.hdReady) return;
    if (paused) { showHD(); return; }
    hd.play().catch(() => {});
    if (hd.requestVideoFrameCallback) hd.requestVideoFrameCallback(() => {
      if (zoom !== current || current.hdReady) return;
      const difference = Math.abs(hd.currentTime - preview.currentTime);
      if (Math.min(difference, hd.duration - difference) > .12) synchronize();
      else showHD();
    });
    else showHD();
  });
  wrapper.append(preview, hd);
  document.body.append(wrapper);
  document.body.classList.add('zoom-open');
  if (!hd.getAttribute('src')) hd.src = hd.dataset.src;
  if (hd.readyState >= 1) synchronize();
  else hd.addEventListener('loadedmetadata', synchronize, {once:true});
  wrapper.addEventListener('click', closeZoom);
  wrapper.addEventListener('pointerleave', event => { if (event.pointerType === 'mouse') closeZoom(); });
  wrapper.addEventListener('keydown', event => {
    if (['Escape', 'Enter', ' '].includes(event.key)) { event.preventDefault(); closeZoom(); }
  });
  wrapper.focus({ preventScroll: true });
  refreshPlayback();
  requestAnimationFrame(() => { wrapper.style.transform = `scale(${scale})`; });
}
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeZoom(); });
document.addEventListener('pointerdown', event => { if (zoom && !zoom.wrapper.contains(event.target)) closeZoom(); });
window.addEventListener('resize', closeZoom);
window.addEventListener('scroll', () => {
  if (!zoom) return;
  const box = zoom.card.getBoundingClientRect();
  if (box.bottom < 0 || box.top > innerHeight) closeZoom();
}, { passive: true });

function makeCard(clip) {
  const card = document.createElement('article');
  card.className = 'video-card';
  card.dataset.clipId = clip.id;
  const video = document.createElement('video');
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = 'none';
  video.dataset.poster = clip.poster;
  video.dataset.src = videoURL(clip.video);
  video.tabIndex = -1;
  video.setAttribute('aria-hidden', 'true');
  const open = document.createElement('button');
  open.type = 'button';
  open.className = 'video-open';
  open.setAttribute('aria-label', 'Open video');
  open.addEventListener('click', () => openViewer(clip, card, open));
  open.addEventListener('pointerenter', () => warmHD(clip));
  open.addEventListener('focus', () => warmHD(clip));
  card.append(video, open);
  visibility.observe(video);
  approaching.observe(video);
  return card;
}

document.querySelectorAll('[data-stage]').forEach(section => {
  const stage = section.dataset.stage;
  const tabs = section.querySelector('.hand-tabs');
  const panel = section.querySelector('.gallery-panel');
  const track = section.querySelector('.video-track');
  const previous = section.querySelector('[data-slide=previous]');
  const next = section.querySelector('[data-slide=next]');
  const range = section.querySelector('.slide-range');
  const expand = section.querySelector('.expand-gallery');
  expand.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4"/></svg><span>Expand</span>';
  let hand = hands[0].id;
  let warmPageTimer;
  function warmNextPage() {
    clearTimeout(warmPageTimer);
    warmPageTimer = setTimeout(() => {
      const bounds = section.getBoundingClientRect();
      if (bounds.bottom < -360 || bounds.top > innerHeight + 360) return;
      const page = Math.round(track.scrollLeft / track.clientWidth);
      track.children[page + 1]?.querySelectorAll('video').forEach(warmVideo);
    }, 300);
  }
  function updateArrows() {
    previous.disabled = track.scrollLeft < 2;
    next.disabled = track.scrollLeft >= track.scrollWidth - track.clientWidth - 2;
    range.value = track.scrollLeft / Math.max(1, track.scrollWidth - track.clientWidth);
  }
  function slide(direction) {
    closeZoom();
    track.scrollBy({ left: direction * track.clientWidth, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
  }
  function render() {
    closeZoom();
    track.querySelectorAll('video').forEach(video => {
      visibility.unobserve(video);
      approaching.unobserve(video);
      visibleVideos.delete(video);
      video.pause();
      video.removeAttribute('src');
      video.load();
    });
    const selected = clips.filter(clip => clip.stage === stage && clip.hand === hand);
    track.replaceChildren();
    for (let start = 0; start < selected.length; start += 20) {
      const slide = document.createElement('div');
      slide.className = 'gallery-slide';
      selected.slice(start, start + 20).forEach(clip => slide.append(makeCard(clip)));
      track.append(slide);
    }
    track.scrollLeft = 0;
    panel.setAttribute('aria-labelledby', `${stage}-tab-${hand}`);
    tabs.querySelectorAll('button').forEach(tab => {
      const active = tab.dataset.hand === hand;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    requestAnimationFrame(updateArrows);
    warmNextPage();
  }
  hands.forEach((item, index) => {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.textContent = item.name;
    tab.id = `${stage}-tab-${item.id}`;
    tab.dataset.hand = item.id;
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', panel.id);
    tab.addEventListener('click', () => { hand = item.id; render(); });
    tab.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (index + 1) % hands.length;
      else if (event.key === 'ArrowLeft') target = (index + hands.length - 1) % hands.length;
      else if (event.key === 'Home') target = 0;
      else if (event.key === 'End') target = hands.length - 1;
      else return;
      event.preventDefault();
      hand = hands[target].id;
      render();
      tabs.children[target].focus();
    });
    tabs.append(tab);
  });
  range.addEventListener('input', () => { track.scrollTo({ left: Number(range.value) * (track.scrollWidth - track.clientWidth), behavior: 'instant' }); });
  range.addEventListener('pointerdown', () => track.classList.add('scrubbing'));
  function finishScrub() {
    if (!track.classList.contains('scrubbing')) return;
    const left = Math.round(track.scrollLeft / track.clientWidth) * track.clientWidth;
    track.classList.remove('scrubbing');
    track.scrollTo({ left, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
  }
  window.addEventListener('pointerup', finishScrub);
  window.addEventListener('pointercancel', finishScrub);
  range.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowDown', 'ArrowRight', 'ArrowUp'].includes(event.key)) {
      event.preventDefault();
      slide(['ArrowLeft', 'ArrowDown'].includes(event.key) ? -1 : 1);
    }
    if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      track.scrollLeft = event.key === 'Home' ? 0 : track.scrollWidth;
    }
  });
  expand.addEventListener('click', () => {
    closeZoom();
    wall.open(window.RESULTS.mosaics[`${stage}-${hand}`], expand, paused);
    refreshPlayback();
  });
  previous.addEventListener('click', () => slide(-1));
  next.addEventListener('click', () => slide(1));
  track.addEventListener('scroll', () => { closeZoom(); updateArrows(); warmNextPage(); }, { passive: true });
  window.addEventListener('scroll', warmNextPage, {passive:true});
  new ResizeObserver(updateArrows).observe(track);
  track.addEventListener('keydown', event => {
    if (event.target !== track) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); slide(event.key === 'ArrowLeft' ? -1 : 1); }
    if (event.key === 'Home' || event.key === 'End') { event.preventDefault(); track.scrollLeft = event.key === 'Home' ? 0 : track.scrollWidth; }
  });
  let drag;
  let suppressClick = false;
  track.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    drag = { id: event.pointerId, x: event.clientX, left: track.scrollLeft, moved: false };
    suppressClick = false;
  });
  track.addEventListener('pointermove', event => {
    if (!drag) return;
    const distance = event.clientX - drag.x;
    if (Math.abs(distance) > 5) {
      drag.moved = true;
      track.classList.add('dragging');
      track.setPointerCapture(drag.id);
      track.scrollLeft = drag.left - distance;
    }
  });
  function endDrag() {
    if (!drag) return;
    suppressClick = drag.moved;
    if (track.hasPointerCapture(drag.id)) track.releasePointerCapture(drag.id);
    track.classList.remove('dragging');
    drag = null;
    setTimeout(() => { suppressClick = false; }, 0);
  }
  track.addEventListener('pointerup', endDrag);
  track.addEventListener('pointercancel', endDrag);
  track.addEventListener('click', event => { if (suppressClick) { event.preventDefault(); event.stopPropagation(); } }, true);
  render();
});
refreshPlayback();
