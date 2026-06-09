/* ══════════════════════════════════════════════════════════════════════════
   particles.js — Lightweight canvas particle system + CSS floating elements
   - Canvas: floating dots connected by lines (medical network feel)
   - CSS: glowing orbs, floating hearts, subtle pulse rings
   All animations are RAF-based and pause when tab is hidden (perf-safe)
══════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── Config ──────────────────────────────────────────────────────────────
  const CFG = {
    dots:        55,          // number of canvas particles
    dotRadius:   2.2,
    maxSpeed:    0.38,
    connectDist: 130,         // px — draw line when closer than this
    dotColors:   ['rgba(230,57,70,', 'rgba(69,123,157,', 'rgba(29,53,87,'],
    orbs:        6,           // CSS blur orbs
    hearts:      12,          // floating emoji hearts
  };

  // ═══════════════════════════════════════════════════════════════════════
  // 1. CANVAS PARTICLE NETWORK  (covers full page, behind everything)
  // ═══════════════════════════════════════════════════════════════════════

  function initCanvasParticles() {
    const canvas = document.createElement('canvas');
    canvas.id = 'particle-canvas';
    document.body.prepend(canvas);
    const ctx = canvas.getContext('2d');

    let W, H, particles = [];

    function resize() {
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    // Create particles
    function makeParticle() {
      const colorBase = CFG.dotColors[Math.floor(Math.random() * CFG.dotColors.length)];
      return {
        x:  Math.random() * W,
        y:  Math.random() * H,
        vx: (Math.random() - 0.5) * CFG.maxSpeed * 2,
        vy: (Math.random() - 0.5) * CFG.maxSpeed * 2,
        r:  CFG.dotRadius * (0.7 + Math.random() * 0.6),
        color: colorBase,
        alpha: 0.4 + Math.random() * 0.5,
      };
    }

    for (let i = 0; i < CFG.dots; i++) particles.push(makeParticle());

    let raf;
    function draw() {
      ctx.clearRect(0, 0, W, H);

      // Update positions
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -10) p.x = W + 10;
        if (p.x > W + 10) p.x = -10;
        if (p.y < -10) p.y = H + 10;
        if (p.y > H + 10) p.y = -10;
      });

      // Draw connecting lines
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CFG.connectDist) {
            const opacity = (1 - dist / CFG.connectDist) * 0.18;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(69,123,157,${opacity})`;
            ctx.lineWidth   = 1;
            ctx.stroke();
          }
        }
      }

      // Draw dots
      particles.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `${p.color}${p.alpha})`;
        ctx.fill();
      });

      raf = requestAnimationFrame(draw);
    }

    // Pause when tab is hidden to save CPU
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) { cancelAnimationFrame(raf); }
      else { raf = requestAnimationFrame(draw); }
    });

    raf = requestAnimationFrame(draw);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 2. CSS GLOWING ORBS (large blurred color blobs)
  // ═══════════════════════════════════════════════════════════════════════

  const ORB_PALETTES = [
    ['rgba(230,57,70,0.12)',  'rgba(193,18,31,0.08)'],
    ['rgba(69,123,157,0.12)', 'rgba(29,53,87,0.08)'],
    ['rgba(168,218,220,0.10)','rgba(69,123,157,0.06)'],
    ['rgba(230,57,70,0.08)',  'rgba(255,184,0,0.06)'],
  ];

  function spawnOrbs() {
    for (let i = 0; i < CFG.orbs; i++) {
      const el = document.createElement('div');
      el.className = 'bg-orb';

      const size  = 280 + Math.random() * 320;
      const pal   = ORB_PALETTES[i % ORB_PALETTES.length];
      const dur   = 18 + Math.random() * 22;   // seconds
      const delay = -(Math.random() * dur);     // start mid-animation
      const dx    = (Math.random() - 0.5) * 400;
      const dy    = (Math.random() - 0.5) * 300;

      el.style.cssText = `
        width: ${size}px; height: ${size}px;
        left: ${Math.random() * 90}%; top: ${Math.random() * 90}%;
        background: radial-gradient(circle, ${pal[0]} 0%, ${pal[1]} 60%, transparent 100%);
        --dx: ${dx}px; --dy: ${dy}px;
        animation-duration: ${dur}s;
        animation-delay: ${delay}s;
      `;
      document.body.appendChild(el);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 3. PULSE RINGS (in hero banner)
  // ═══════════════════════════════════════════════════════════════════════

  function spawnPulseRings() {
    const hero = document.querySelector('.hero-banner');
    if (!hero) return;

    // Make hero relatively positioned
    if (getComputedStyle(hero).position === 'static') {
      hero.style.position = 'relative';
      hero.style.overflow = 'hidden';
    }

    for (let i = 0; i < 3; i++) {
      const ring = document.createElement('div');
      ring.className = 'pulse-ring';
      const size = 120 + i * 80;
      ring.style.cssText = `
        width: ${size}px; height: ${size}px;
        right: ${40 + i * 30}px; bottom: ${-size / 2}px;
        animation-duration: ${2.5 + i * 0.8}s;
        animation-delay: ${i * 0.7}s;
      `;
      hero.appendChild(ring);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // 5. HERO CANVAS — DNA/molecule dots bouncing in the banner
  // ═══════════════════════════════════════════════════════════════════════

  function initHeroCanvas() {
    const hero = document.querySelector('.hero-banner');
    if (!hero) return;

    const c = document.createElement('canvas');
    c.id = 'hero-canvas';
    hero.prepend(c);
    const ctx = c.getContext('2d');

    let particles = [];
    function resize() {
      c.width  = hero.offsetWidth;
      c.height = hero.offsetHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 28; i++) {
      particles.push({
        x:  Math.random() * c.width,
        y:  Math.random() * c.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        r:  1.5 + Math.random() * 2,
      });
    }

    function draw() {
      ctx.clearRect(0, 0, c.width, c.height);

      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > c.width)  p.vx *= -1;
        if (p.y < 0 || p.y > c.height) p.vy *= -1;

        // Connections
        particles.forEach(q => {
          const d = Math.hypot(p.x - q.x, p.y - q.y);
          if (d < 80) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = `rgba(255,255,255,${(1 - d / 80) * 0.25})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        });

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.fill();
      });

      requestAnimationFrame(draw);
    }
    draw();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // INIT — run after DOM ready
  // ═══════════════════════════════════════════════════════════════════════

  function init() {
    initCanvasParticles();
    spawnOrbs();
    spawnPulseRings();
    initHeroCanvas();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
