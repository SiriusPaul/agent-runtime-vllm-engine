# osh26_agent_runtime_ppt

- Canvas format: ppt169
- Created: 20260609
- Refined: 20260701
- Current deck: 24 pages
- Main PPTX: `exports/osh26_agent_runtime_ppt_refined_20260701.pptx`

## Directories

- `svg_output/`: raw SVG output
- `svg_final/`: finalized SVG output
- `images/`: presentation assets
- `notes/`: speaker notes
- `templates/`: project templates
- `sources/`: source materials and normalized markdown
- `exports/`: main native pptx (timestamped); `_svg.pptx` sibling added when exported with `--svg-snapshot`
- `backup/<timestamp>/`: svg_output/ archive (always written in default-flow mode; safe to delete old timestamps)

## Build

```powershell
python .\rebuild_svg_deck.py
python .\create_pptx_from_svg.py
python .\verify_generated_pptx.py
```

The refined PPTX is assembled as native OpenXML with full-slide SVG images, so it does not depend on npm packages.
