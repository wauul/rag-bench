---
name: RAG Bench
description: A precise workspace for measured retrieval comparisons.
colors:
  lavender: "#655ce0"
  control-lavender: "#5b54df"
  teal: "#169b8a"
  apricot: "#ed9760"
  rose: "#bc69a9"
  canvas: "#f7f8fc"
  sidebar: "#eef0f8"
  panel: "#ffffff"
  ink: "#20243b"
  muted-ink: "#626a83"
  caption-ink: "#636b84"
  panel-border: "#e2e5f0"
  grid: "#eceef5"
  lavender-wash: "#eeecff"
  lavender-deep: "#5147b8"
  lavender-mid: "#9b94eb"
  nav-hover: "#e3e6f3"
  score-surface: "#f3f4fa"
  ready-wash: "#e0f3ec"
  ready-ink: "#21765a"
  pending-wash: "#e9ebf3"
  pending-ink: "#777f96"
  neutral-series: "#858ca3"
typography:
  headline:
    fontSize: "2rem"
    fontWeight: 720
    letterSpacing: "-0.025em"
  section:
    fontSize: "1.35rem"
    fontWeight: 680
    letterSpacing: "-0.025em"
  title:
    fontSize: "1.08rem"
    fontWeight: 650
    letterSpacing: "-0.025em"
  body:
    fontFamily: "sans-serif"
  metric:
    fontSize: "1.8rem"
    letterSpacing: "-0.03em"
  chart-label:
    fontFamily: "sans-serif"
    fontSize: "12px"
  button:
    fontWeight: 600
  chip:
    fontSize: "0.75rem"
rounded:
  chip: "6px"
  control: "9px"
  navigation: "10px"
  utility: "12px"
  panel: "14px"
  sample: "16px"
spacing:
  chip-block: "0.25rem"
  chip-inline: "0.5rem"
  compact-gap: "0.7rem"
  tab-gap: "1.4rem"
components:
  button-primary:
    backgroundColor: "{colors.control-lavender}"
    textColor: "{colors.panel}"
    rounded: "{rounded.control}"
    typography: "{typography.button}"
  button-secondary:
    rounded: "{rounded.control}"
    typography: "{typography.button}"
  panel:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.panel}"
  navigation-selected:
    backgroundColor: "{colors.lavender}"
    textColor: "{colors.panel}"
    rounded: "{rounded.navigation}"
    padding: "0.65rem 0.7rem"
  sample-chip:
    backgroundColor: "#ffffffad"
    textColor: "{colors.lavender-deep}"
    rounded: "{rounded.chip}"
    padding: "0.25rem 0.5rem"
    typography: "{typography.chip}"
---

# Design System: RAG Bench

## Overview

**Creative North Star: "Comparison Canvas"**

A quiet, compact analytical workspace: cool surfaces, precise borders, one sans family, and a retained lavender, teal, apricot, and rose palette. Charts and tabular numbers carry the evidence; native Streamlit controls carry interaction and accessibility.

The visual system supports comparisons without claiming certainty beyond the saved measurements. Missing data stays visibly absent, partial scores stay qualified, and measurement scope stays close to its chart. Page composition belongs in `.impeccable/surfaces/dashboard.md`.

**Key Characteristics:**
- Cool, flat surfaces with restrained borders.
- Stable configuration colors and tabular figures.
- Native controls with clear focus and compact panels.
- Visible uncertainty and explicit measurement scope.

## Colors

### Primary
Lavender identifies the brand, selected workspace item, configuration names, and the first chart series. Native primary controls use the separately extracted control lavender from the Streamlit theme; preserve this observed distinction. Lavender wash, midpoint, and deep tones form the score heatmap and sample panel treatments.

### Secondary
Teal identifies the second configuration and completed outcomes. Apricot identifies the third configuration and retry outcomes. Rose identifies the fourth configuration and failed outcomes. These roles are contextual; a configuration color does not itself indicate success or failure.

### Neutral
Canvas and sidebar tones separate the workspace from white panels. Ink carries primary text; muted and caption ink support annotations. Panel border and grid tones organize evidence. Neutral series denotes queued or cancelled outcomes. Ready and pending washes support setup readiness.

**The Stable Series Rule.** Keep configuration colors in lavender, teal, apricot, rose order even when a measurement is missing; skip the absent mark, not its color slot.

## Typography

The Streamlit theme supplies one native sans family; charts explicitly use sans-serif. Do not infer a bundled named font or a modular ratio. Headings follow the extracted headline, section, and title roles; document, question, and current-run panels use a compact section override (1.18rem). The mobile headline is smaller (1.65rem at 720px and below).

Metric values and score tables use tabular figures. Chart labels are compact; native body/control sizes and line heights remain framework-owned where the source does not set them.

**The Numeric Alignment Rule.** Use tabular figures for comparable measurements and retain their units and unavailable-value markers.

## Layout

The main container is bounded (1380px), with desktop padding (3.5rem 2.8rem 4rem). At 900px and below its padding becomes (4rem 1.25rem 3rem). This later rule overrides the earlier 720px container padding; the 720px heading and sample-panel adjustments still apply.

At 640px and below, horizontal column groups stack. Overview metrics remain two per row. Native Streamlit owns the collapsible sidebar. Tabs keep their labels on one line, never shrink, and scroll horizontally; their gap reduces from the tab-gap token to 1rem on narrow screens. Login content is bounded (420px).

Use the established compact spacing steps inside panels and keep comparison graphics stretched to the available column. Avoid treating the overview's specific column ratios as a rule for every surface.

**The Narrow Evidence Rule.** Stack evidence panels on narrow screens and allow tab navigation to scroll without breaking tab labels.

## Elevation & Depth

Custom workspace surfaces use tonal separation and thin borders; the application stylesheet adds no panel shadows. Native popovers and controls retain their framework treatments. Selection is a filled lavender surface, hover a cool tinted surface, and focus a visible lavender outline (2px, with 3px offset for buttons and links; 2px offset for sidebar navigation).

**The Flat Panel Rule.** Separate evidence panels with tone and borders rather than introducing decorative elevation.

## Shapes

Panels and forms use the panel radius. Buttons use the control radius; sidebar choices use the navigation radius; upload zones and expanders use the utility radius. Sample panels are softer, with the sample radius; metadata chips are compact, with the chip radius. Readiness indicators are circular. The brand diamond and readiness check are exact CSS geometry because inline SVG is removed by Streamlit in this rendering path.

## Components

**Buttons.** Native Streamlit buttons retain their semantics, disabled states, and framework hover behavior. Custom styling sets the control radius, medium weight, and minimum height (44px). Primary actions use the theme's control lavender. Focus uses the shared visible outline.

**Inputs.** Keep native text, select, upload, and password controls, including labels, validation, keyboard behavior, and framework focus states. Text caret color is lavender; upload zones have a pale cool background and utility radius. Unspecified native input borders and radii are not invented tokens.

**Navigation.** Sidebar radio labels have minimum height (44px), compact padding, the navigation radius, cool hover fill, and filled lavender selection. Their color/background transition is brief (160ms ease-out). Tabs retain native selection semantics and their scrollable narrow-screen treatment.

**Panels.** White bordered containers group comparisons, answers, history, setup, and run state. Internal padding remains native except where source explicitly styles a signature panel. The lavender sample panel uses padding (1.35rem 1.5rem), reduced to 1rem below 720px.

**Chips and statuses.** Sample metadata chips use translucent white over lavender wash. Native status badges retain text and their Streamlit status colors; chart outcome colors use the contextual mapping above. Readiness indicators pair exact geometry with adjacent labels and details.

**Charts and scores.** Plotly shares transparent canvas backgrounds, muted sans labels, subtle grids, horizontal legends, and dark hover labels. Heatmaps leave unavailable cells empty. Score strips use paired cool surfaces and tabular numbers. Motion is limited to interaction feedback; reduced-motion preference disables transitions and animation.

## Do's and Don'ts

### Do:
- Do preserve the retained four-color palette and stable configuration order.
- Do keep native control semantics, labels, focus, and disabled states.
- Do display missing measurements, partial-score qualifiers, and measurement scope explicitly.
- Do keep comparison figures aligned with visible units and tabular scores.

### Don't:
- Don't invent scores, winners from incomplete measurements, quota remaining, or dollar spend from HTTP counts.
- Don't replace native controls with decorative imitations in the application.
- Don't add shadows to evidence panels or break narrow-screen tab labels.

Extraction evidence: `dashboard/styles.css`, `dashboard/charts.py`, `dashboard/ui.py`, `dashboard/overview.py`, sampled controls in `dashboard/app.py`, `dashboard/performance.py`, `dashboard/debugger.py`, and `dashboard/optimization.py`, plus `.streamlit/config.toml`. Sidecar samples translate visual treatments into standalone HTML; they are previews, not substitute application controls. Synthesized tonal ramps in the sidecar are preview metadata, not additional shipped tokens.

Not canonized: the existing uppercase "PRIVATE WORKSPACE" sidebar kicker remains a build defect, not a reusable typography role. No source repair is performed by this documentation pass.
