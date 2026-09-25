# Changelog

A formátum a [Keep a Changelog](https://keepachangelog.com/hu/1.1.0/) ajánlását követi, a verziózás
[SemVer](https://semver.org/lang/hu/).

## [Unreleased]

### Added
- Projektváz: `pyproject.toml`, `bookpecker` CLI (`run`, `layout`, `assemble`, `info`, `new-book`).
- Könyvenkénti konfiguráció (`books/<slug>/book.yaml`) projekt szintű alapértékekkel és
  oldaltartományonkénti `page_overrides`-szal; oldalparitás (`first_page_side`), `skip_pages`.
- Lépcsőzetes, hash-alapú cache-elt pipeline: rasterize → preprocess (normalizálás, deskew) → layout → OCR.
- Layout: PP-DocLayout (PaddleOCR) és modell nélküli heurisztikus motor; oldalparitás szerint tükrözött
  margómaszk; régiótípus szerinti eldobás (élőfej, oldalszám, ornamentum, kép).
- Olvasási sorrend vegyes (teljes szélességű + többhasábos) oldalakra, hasábköz-detektálással.
- Sidebar-felismerés keretvonalak és színezett háttér alapján; Markdown idézetblokként, helyben vagy az
  oldal végén.
- OCR motorok: Tesseract (sorgeometriával) és kísérleti PaddleOCR-VL; régiótípusonként választható.
- Markdown összeállítás: sortörés csak bekezdéshatáron (mondat közben soha), elválasztás-összevonás magyar
  szabályokkal, bekezdés-folytatás hasáb- és oldalhatáron át, egységes `##` címsorok (vagy `auto` szintek),
  HTML → Markdown táblák, `<!-- page N -->` jelölők.
- Debug overlay PNG oldalanként a beállítások hangolásához.
- Unit tesztek és szintetikus oldalas layout teszt.
