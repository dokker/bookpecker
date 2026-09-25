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
- Lapos könyvkonfig: `books/<slug>.yaml` (a `books/<slug>/book.yaml` is működik); a `new-book` a sablon
  `title:`/`pdf:` sorait tölti ki.
- `edge_touch`: a lap szélét érintő régiók (belógó szomszéd oldal, szegélyek) eldobása.
- `replacements`: könyvspecifikus regex-javítások (pl. Tesseract `%` → `96` tévesztés).
- Megvilágítás-kiegyenlítés (`preprocess.flatten`) a gerincárnyék ellen.
- OCR-ráhagyás a gerinc felőli oldalon, a ráhagyásba lógó képek/ornamentumok kifehérítése.
- Tesseract zajszűrés: alacsony konfidenciájú szavak, sorvégi töredékek, ornamentumból olvasott „szemét sorok”.
- Egymásba ágyazott duplikált layout-régiók (pl. kétsoros cím háromszor) kiszűrése.

### Changed
- A sidebar-keretjelöltek közül kiesnek a képekre/ornamentumokra eső és a tartalomdoboz szélét érintő
  jelöltek (díszes oldalszegély nem lesz „sidebar”).
- Heurisztikus layout: soralapú csoportosítás (sorfragmensek → hasábos/teljes szélességű sorok → blokkok),
  adaptív szóköz-becslés.
- `paddle` extra: `paddleocr[doc-parser]`; a PaddleOCR-VL motor nem tölti be a pipeline saját
  layout/orientáció almodelljeit.
