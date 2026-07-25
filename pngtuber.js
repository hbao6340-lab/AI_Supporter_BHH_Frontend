// pngtuber.js - PNGTuber character controller for idle and mouth sync
// Works alongside app.js for model switching and lip sync

(function() {
    'use strict';

    function ready(fn) {
        if (document.readyState !== 'loading') fn();
        else document.addEventListener('DOMContentLoaded', fn);
    }

    ready(() => {
        const img = document.getElementById('pngtuber-img');
        const container = document.getElementById('pngtuber-container');
        if (!img || !container) return;

        const base = './PNGTuber/';
        const textures = {
            normal: base + 'normal.png',
            blink: base + 'normal-blink.png',
            talkOpen: base + 'talk-open.png',
            talkClosed: base + 'talk-closed.png'
        };

        // Preload textures
        Object.values(textures).forEach(src => {
            const i = new Image();
            i.src = src;
        });

        let currentState = 'normal';
        let blinkTimer = null;
        let talkTimer = null;
        let lastVolume = 0;
        let isInitialized = false;

        const BLINK_MIN = 2500;
        const BLINK_MAX = 5000;
        const BLINK_DURATION = 150;
        const VOLUME_SMOOTH = 0.3;
        const OPEN_THRESHOLD = 0.12;
        const CLOSED_THRESHOLD = 0.05;

        function scheduleBlink() {
            clearTimeout(blinkTimer);
            const delay = BLINK_MIN + Math.random() * (BLINK_MAX - BLINK_MIN);
            blinkTimer = setTimeout(() => {
                if (currentModelType === 'pngtuber' && !isTalking()) {
                    doBlink();
                }
                scheduleBlink();
            }, delay);
        }

        function doBlink() {
            const prev = currentState;
            if (prev === 'blink') return;
            setState('blink');
            setTimeout(() => {
                setState(prev === 'blink' ? 'normal' : prev);
            }, BLINK_DURATION);
        }

        function isTalking() {
            return currentState === 'talkOpen' || currentState === 'talkClosed';
        }

        function setState(s) {
            if (!textures[s]) return;
            if (currentState === s) return;
            currentState = s;
            img.src = textures[s];
        }

        function applyVolume(volume) {
            const v = Math.max(0, Math.min(1, volume));
            lastVolume = lastVolume * (1 - VOLUME_SMOOTH) + v * VOLUME_SMOOTH;

            if (lastVolume > OPEN_THRESHOLD) {
                setState('talkOpen');
                clearTimeout(talkTimer);
                talkTimer = setTimeout(() => {
                    if (lastVolume <= OPEN_THRESHOLD) {
                        setState('normal');
                    }
                }, 100);
            } else if (lastVolume > CLOSED_THRESHOLD) {
                setState('talkClosed');
                clearTimeout(talkTimer);
                talkTimer = setTimeout(() => {
                    if (lastVolume <= CLOSED_THRESHOLD) {
                        setState('normal');
                    }
                }, 100);
            } else {
                clearTimeout(talkTimer);
                if (isTalking()) {
                    setState('normal');
                }
            }
        }

        function onMouseMove(e) {
            if (currentModelType !== 'pngtuber') return;
            const rect = img.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;
            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            const dx = (e.clientX - cx) / rect.width;
            const dy = (e.clientY - cy) / rect.height;
            const rotateY = dx * 6;
            const rotateX = -dy * 4;
            const translateX = dx * 4;
            const translateY = dy * 3;
            img.style.transform = `translate(${translateX}px, ${translateY}px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
            img.style.transition = 'transform 0.08s linear';
            img.style.transformStyle = 'preserve-3d';
        }

        function init() {
            if (isInitialized) return;
            isInitialized = true;

            setState('normal');
            container.classList.add('active');
            scheduleBlink();

            window.addEventListener('mousemove', onMouseMove);

            window.addEventListener('resize', () => {
                if (currentModelType === 'pngtuber') {
                    onMouseMove({ clientX: window.innerWidth / 2, clientY: window.innerHeight / 2 });
                }
            });
        }

        function stop() {
            clearTimeout(blinkTimer);
            clearTimeout(talkTimer);
            container.classList.remove('active');
            isInitialized = false;
            window.removeEventListener('mousemove', onMouseMove);
        }

        window.pngtuber = {
            init,
            stop,
            setState,
            applyVolume,
            onMouseMove,
            getState: () => currentState,
            textures,
            img,
            container
        };

        window.initPNGTuber = () => window.pngtuber.init();
        window.stopPNGTuber = () => window.pngtuber.stop();
        window.setPNGTuberVolume = (v) => window.pngtuber.applyVolume(v);
        window.setPNGTuberViseme = (viseme) => {
            if (viseme === 'A') setState('normal');
            else if (viseme === 'B') { setState('talkClosed'); setTimeout(() => setState('normal'), 120); }
            else if (viseme === 'C') { setState('talkOpen'); setTimeout(() => setState('normal'), 120); }
        };
        window.forcePNGTuberBlink = () => {
            if (!window.pngtuber) return;
            const prev = window.pngtuber.getState();
            setState('blink');
            setTimeout(() => setState(prev === 'blink' ? 'normal' : prev), BLINK_DURATION);
        };

        try {
            if (typeof currentModelType !== 'undefined' && currentModelType === 'pngtuber') {
                init();
            }
        } catch(e) {}

        setTimeout(() => {
            const origApply = window.applyVolumeToMouth;
            const origAnimate = window.animateViseme;

            window.applyVolumeToMouth = function(volume) {
                try {
                    if (typeof currentModelType !== 'undefined' && currentModelType === 'pngtuber' && window.setPNGTuberVolume) {
                        window.setPNGTuberVolume(volume);
                        return;
                    }
                } catch(e){}
                if (typeof origApply === 'function') return origApply.call(this, volume);
            };

            window.animateViseme = function(viseme) {
                try {
                    if (typeof currentModelType !== 'undefined' && currentModelType === 'pngtuber') {
                        if (window.setPNGTuberViseme) {
                            window.setPNGTuberViseme(viseme);
                            return;
                        }
                    }
                } catch(e){}
                if (typeof origAnimate === 'function') return origAnimate.call(this, viseme);
            };

            if (typeof window.switchModelType === 'function') {
                const origSwitch = window.switchModelType;
                window.switchModelType = async function(type) {
                    const res = await origSwitch.call(this, type);
                    try {
                        if (type === 'pngtuber') {
                            init();
                        } else {
                            stop();
                        }
                    } catch(e){}
                    return res;
                };
            }
        }, 400);
    });
})();
