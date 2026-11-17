"""Animated food-themed background for the Streamlit app (pure CSS, no libraries).

Look: a deep purple-black night sky with slow glowing "spice" colour blobs, and food emoji
that float gently upwards while swaying. The page content sits on a white card above it.
Call inject_background() once, right after st.set_page_config().
"""
import random

import streamlit as st

FOODS = ["🍕", "🍔", "🍟", "🌮", "🍜", "🍣", "🍩", "🥗", "🍗", "🍛", "🍦", "🥘", "🌯", "🥟", "🍰", "☕", "🍅", "🌶️"]

CSS = """
<style>
/* ---- dark base, everything Streamlit paints becomes see-through ---- */
html, body { background: #0b0416 !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: transparent !important; }

/* ---- the content sits on a soft white card above the animation ---- */
[data-testid="stMainBlockContainer"], .block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 18px;
    padding: 1.6rem 2.4rem 2.6rem 2.4rem !important;
    margin-top: 1.2rem;
    margin-bottom: 2rem;
    box-shadow: 0 12px 50px rgba(0, 0, 0, 0.55);
}

/* ---- animated layer (behind everything, never blocks clicks) ---- */
.fx-bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none;
         background: radial-gradient(120% 90% at 50% 0%, #2a0f4d 0%, #14072a 45%, #0b0416 100%); }

.fx-orb { position: absolute; border-radius: 50%; filter: blur(70px); opacity: .55;
          animation: fx-drift 26s ease-in-out infinite alternate; }
.fx-orb.o1 { width: 46vw; height: 46vw; left: -12vw; top: -10vw;  background: #e63946; animation-duration: 30s; }   /* chilli  */
.fx-orb.o2 { width: 40vw; height: 40vw; right: -10vw; top: 8vh;   background: #fb8500; animation-duration: 24s; animation-delay: -6s; }  /* saffron */
.fx-orb.o3 { width: 36vw; height: 36vw; left: 18vw; bottom: -14vw; background: #7b2cbf; animation-duration: 34s; animation-delay: -11s; } /* beetroot*/
.fx-orb.o4 { width: 26vw; height: 26vw; right: 12vw; bottom: -8vw; background: #ffb703; opacity: .38; animation-duration: 28s; animation-delay: -3s; } /* turmeric */

@keyframes fx-drift { from { transform: translate3d(0, 0, 0) scale(1); }
                      to   { transform: translate3d(7vw, 6vh, 0) scale(1.18); } }

.fx-food { position: absolute; bottom: -90px; line-height: 1; will-change: transform;
           animation-name: fx-rise, fx-sway; animation-timing-function: linear, ease-in-out;
           animation-iteration-count: infinite, infinite; animation-direction: normal, alternate;
           filter: drop-shadow(0 0 10px rgba(255, 183, 3, .35)); }

@keyframes fx-rise { from { transform: translateY(0); }       to { transform: translateY(-125vh); } }
@keyframes fx-sway { from { translate: -26px 0; rotate: -18deg; } to { translate: 26px 0; rotate: 18deg; } }

@media (prefers-reduced-motion: reduce) { .fx-orb, .fx-food { animation: none !important; } }
</style>
"""


def inject_background(n_items: int = 18, seed: int = 7) -> None:
    rnd = random.Random(seed)                     # same layout every run
    items = []
    for i in range(n_items):
        rise, sway = rnd.randint(24, 46), rnd.randint(5, 11)
        items.append(
            f'<span class="fx-food" style="left:{rnd.randint(2, 96)}%;font-size:{rnd.randint(26, 60)}px;'
            f'opacity:{rnd.uniform(0.16, 0.34):.2f};animation-duration:{rise}s,{sway}s;'
            f'animation-delay:-{rnd.randint(0, rise)}s,-{rnd.randint(0, sway)}s">{FOODS[i % len(FOODS)]}</span>')
    orbs = "".join(f'<div class="fx-orb o{k}"></div>' for k in range(1, 5))
    st.markdown(CSS + f'<div class="fx-bg" aria-hidden="true">{orbs}{"".join(items)}</div>',
                unsafe_allow_html=True)
