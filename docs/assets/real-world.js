'use strict';

(() => {
  const section = document.querySelector('#real-world');
  const track = section.querySelector('.video-track');
  const previous = section.querySelector('[data-slide=previous]');
  const next = section.querySelector('[data-slide=next]');
  const range = section.querySelector('.slide-range');
  const behavior = () => reducedMotion.matches ? 'instant' : 'smooth';
  const count = 5;
  const cards = [];
  let step = 0;
  let currentPosition = 0;
  let settleTimer;
  let drag;
  let suppressClick = false;

  for (let slot = 0; slot < count * 3; slot++) {
    const index = slot % count + 1;
    const name = `clip-${String(index).padStart(2, '0')}`;
    const card = makeCard({
      id: `real-world-${index}`,
      video: `media/real-world-preview/${name}.mp4`,
      hd: `media/real-world-cut/${name}.mp4`,
      poster: `media/real-world-cut/${name}.webp`,
    });
    card.querySelector('.video-open').setAttribute('aria-label', `Open real-world video ${index}`);
    cards.push(card);
    track.append(card);
  }

  const position = () => ((track.scrollLeft / step - count) % count + count) % count;
  const measureStep = () => cards[0].getBoundingClientRect().width + parseFloat(getComputedStyle(track).gap);
  function updateControls() {
    if (!step || Math.abs(measureStep() - step) > .1) return;
    currentPosition = position();
    const index = Math.round(currentPosition) % count;
    range.value = Math.min(1, currentPosition / (count - 1));
    range.setAttribute('aria-valuetext', `Video ${index + 1} of ${count}`);
    cards.forEach((card, slot) => {
      const left = slot * step - track.scrollLeft;
      card.querySelector('.video-open').tabIndex = left > -step + 2 && left < track.clientWidth - 2 ? 0 : -1;
    });
  }
  function recenter() {
    if (!step || Math.abs(measureStep() - step) > .1) return;
    const cycle = count * step;
    const shift = track.scrollLeft < cycle - 1 ? count : track.scrollLeft >= 2 * cycle - 1 ? -count : 0;
    if (!shift) return;
    closeZoom();
    // Move the playing elements with the visible cards, preserving playback time.
    cards.forEach((card, slot) => {
      const left = slot * step - track.scrollLeft;
      if (left <= -step || left >= track.clientWidth) return;
      const destination = cards[slot + shift];
      if (!destination) return;
      const video = card.querySelector('video');
      const other = destination.querySelector('video');
      card.prepend(other);
      destination.prepend(video);
    });
    track.style.scrollSnapType = 'none';
    track.scrollTo({left: track.scrollLeft + shift * step, behavior: 'instant'});
    if (drag) drag.left += shift * step;
    requestAnimationFrame(() => {
      track.style.scrollSnapType = '';
      updateControls();
      refreshPlayback();
    });
  }
  function slide(direction) {
    closeZoom();
    recenter();
    track.scrollBy({left: direction * step, behavior: behavior()});
  }
  previous.addEventListener('click', () => slide(-1));
  next.addEventListener('click', () => slide(1));
  range.addEventListener('input', () => {
    closeZoom();
    track.scrollTo({left: (count + Number(range.value) * (count - 1)) * step, behavior: 'instant'});
  });
  range.addEventListener('pointerdown', () => track.classList.add('scrubbing'));
  const finishScrub = () => track.classList.remove('scrubbing');
  window.addEventListener('pointerup', finishScrub);
  window.addEventListener('pointercancel', finishScrub);
  [track, range].forEach(element => element.addEventListener('keydown', event => {
    if (event.target !== element) return;
    if (['ArrowLeft', 'ArrowRight', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (event.key === 'Home' || event.key === 'End') {
        closeZoom();
        track.scrollTo({left: (count + (event.key === 'Home' ? 0 : count - 1)) * step, behavior: behavior()});
      } else slide(['ArrowLeft', 'ArrowDown'].includes(event.key) ? -1 : 1);
    }
  }));
  track.addEventListener('scroll', () => {
    closeZoom();
    updateControls();
    if (drag && (track.scrollLeft < step || track.scrollLeft > track.scrollWidth - track.clientWidth - step)) recenter();
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => { if (!drag) recenter(); }, 160);
  }, {passive: true});
  track.addEventListener('scrollend', () => { if (!drag) recenter(); });
  new ResizeObserver(() => {
    step = measureStep();
    track.scrollTo({left: (count + currentPosition) * step, behavior: 'instant'});
    updateControls();
  }).observe(track);

  track.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    drag = {id: event.pointerId, x: event.clientX, left: track.scrollLeft, moved: false};
    suppressClick = false;
  });
  track.addEventListener('pointermove', event => {
    if (!drag) return;
    const distance = event.clientX - drag.x;
    if (Math.abs(distance) <= 5) return;
    drag.moved = true;
    track.classList.add('dragging');
    track.setPointerCapture(drag.id);
    track.scrollLeft = drag.left - distance;
  });
  const endDrag = () => {
    if (!drag) return;
    suppressClick = drag.moved;
    if (track.hasPointerCapture(drag.id)) track.releasePointerCapture(drag.id);
    track.classList.remove('dragging');
    drag = null;
    setTimeout(() => { suppressClick = false; }, 0);
  };
  track.addEventListener('pointerup', endDrag);
  track.addEventListener('pointercancel', endDrag);
  track.addEventListener('click', event => {
    if (suppressClick) { event.preventDefault(); event.stopPropagation(); }
  }, true);
  requestAnimationFrame(updateControls);
})();
