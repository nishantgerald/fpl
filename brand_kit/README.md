# Gameweek Zero — brand kit v1.0

The mark is a top-down centre circle crossed by the halfway line. It reads as a slashed zero — the null set, the state before any data exists. Pitch and number in one shape.

Every asset is hand-authored SVG. All text is converted to outlined paths, so nothing depends on a font being installed.

## Contents

### `01-core/`

| File | Use |
|---|---|
| `brand-sheet.svg` | One-page reference: construction, clearspace, palette, type, variants |
| `mark-full-*.svg` | Full mark, bar runs past the ring. Transparent background, mint / white / black |
| `mark-glyph-*.svg` | Zero-proportioned mark, bar stops at the ring. Use inline with text |
| `icon-primary.svg` | Default app tile — purple ground, mint mark, pitch linework |
| `icon-inverse.svg` | Mint ground, purple mark |
| `icon-ink.svg` | Near-black ground for dark UI |
| `icon-mono-black/white.svg` | Single-colour, for print, embroidery, sponsor walls |
| `icon-simplified.svg` | Heavier strokes, no pitch lines — for anything under 48 px |
| `lockup-horizontal-*.svg` | Primary lockup. `dark` on purple, `light` on white, `mono` for one colour |
| `lockup-stacked-*.svg` | Square-ish contexts: profile pictures, splash, merch |
| `lockup-compact-*.svg` | `GW` + the mark as the zero. Nav bars, tight headers |

### `02-platform/`

| File | Use |
|---|---|
| `favicon.svg` | 64 px, simplified mark |
| `app-icon-ios-1024.svg` | Full-bleed square, **no rounded corners** — iOS applies its own mask |
| `android-adaptive-foreground-432.svg` + `-background-432.svg` | Android adaptive icon pair; art sits inside the 264 px safe circle |
| `maskable-icon-512.svg` | PWA maskable; content inside the 80% safe zone |
| `splash-1242x2688.svg` | Launch screen |
| `og-card-1200x630.svg` | Open Graph / Twitter card |

### `03-ui/`

`spinner-mint.svg` (animated via SMIL — the arc rotates, the halfway line stays put), `empty-state.svg`, `avatar-placeholder.svg`, `pitch-formation-442.svg`, and four chip icons: `chip-wildcard`, `chip-bench-boost`, `chip-triple-captain`, `chip-free-hit`.

### `04-marketing/`

`x-header-1500x500`, `linkedin-banner-1584x396`, `github-hero-1280x640`, `appstore-feature-1024x500`.

### `tokens.css`

CSS custom properties for the palette, type, and radii.

## Rules

**Clearspace.** Leave `x` on all sides, where `x` is the ring stroke weight. Nothing enters that zone.

**Minimum size.** 16 px for `mark-glyph`, 24 px for the pitch version. Below 48 px switch to `icon-simplified.svg` — the pitch linework closes up.

**Colour.** Mint (`#00FF87`) is the only accent that touches the mark. Cyan (`#04F5FF`) is reserved for one highlight per screen — the captain ring, a live state. On white, use `#00B85F` instead of mint; raw mint fails contrast on light backgrounds.

**Type.** Poppins — Bold for the wordmark and headings, Medium for labels, Regular for body. Geometric sans, so its circular bowls sit naturally next to the mark.

**Don't.** Re-draw the mark, rotate it, tint it, add effects, stretch the lockups non-uniformly, or place the wordmark on a busy photograph without a solid plate behind it.

## Rasterising

```bash
# PNG at any size
rsvg-convert -w 1024 01-core/icon-primary.svg -o icon-1024.png

# favicon.ico
rsvg-convert -w 32 02-platform/favicon.svg -o f32.png
rsvg-convert -w 16 02-platform/favicon.svg -o f16.png
convert f16.png f32.png favicon.ico
```

The animated spinner uses SMIL, which works in every current browser but not in most static rasterisers. If you need it in React Native or as a Lottie file, the arc is a single `stroke-dasharray` rotation — trivial to reimplement.
