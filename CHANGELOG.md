# Changelog

A formátum a [Keep a Changelog](https://keepachangelog.com/hu/1.1.0/) ajánlását követi, a verziózás
[SemVer](https://semver.org/lang/hu/).

## [Unreleased]

### Fixed
- Sor végén elválasztott rag szám vagy rövidítés után: `10-⏎zel` → `10-zel` (eddig `10- zel`),
  `VÉ-⏎je` → `VÉ-je` (eddig `VÉje`).
- Gerinc melletti pöttyök: sor eleji `*`/írásjel-szemét kisbetű előtt törölve (eddig a `*`
  felsorolásnak számított, és szó közepén is új bekezdést nyitott); sor végi `--`, `-–`, `- ,`
  egy elválasztójellé; ponttá olvasott elválasztójel (`fegy.` + `verek`) szótár alapján összevonva.
- Minőségi riport: számok ragjai (`5%-kal`, `3-nál`) és a pont nélkül látott rövidítések (`stb`,
  `ill`, `max`…) nem számítanak ismeretlen szónak.

### Added
- `margin_notes` layout-opció (alapból ki): a fő szövegtömbtől oldalra eső keskeny szövegblokkok
  (külső margós széljegyzetek) egy sidebar-csoportba kerülnek, alapból az oldal végére – nem
  keverednek a főszöveg bekezdései közé. Kikapcsolva a layout cache-kulcs változatlan.
- `books/grimoire.yaml`: Grimoire – A mágia könyve (108 DPI a 72 pt/px oldalképekhez, szűk margó +
  `edge_touch` a szabálytalanul vágott szkennekhez, széljegyzetek, borító/tartalomjegyzék/karakterlap
  kihagyva).
- Párhuzamos pipeline (`run`/`layout`, `-j/--workers`, alap: CPU-szám − 2): előfeldolgozás külön
  folyamatokban előre, Tesseract szálkészletben a GPU-s PP-OCR-rel egy időben, összefésülés és mentés
  külön szálon. A kimenet bájtra azonos a sorossal; 20 oldalon 303 s → 125 s.

### Changed
- Hunspell: szóellenőrzés közvetlenül a libhunspell-lel (ctypes), a `hunspell -a` pipe csak
  tartalék. A pipe minden ismeretlen szóhoz javaslatokat is számolt (magyar szótárral ~30–120 ms);
  a Codex teljes Markdown-összeállítása ~1 óráról 14 s-ra gyorsult, azonos szöveggel.
- A Tesseract-folyamatok `OMP_THREAD_LIMIT=1`-et kapnak párhuzamos futáskor – csak ők, a saját
  folyamatunk (Paddle) nem.
- `books/codex.yaml`: a teljes könyv feldolgozásához kihagyva a borító, előszó, a „A Szerepjátékról…”
  bevezető (PDF 1–9), a tartalomjegyzék (259) és a karakterlap + térkép (262–264).

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

- `paddle_ocr` OCR motor: PP-OCRv5 sordetektálás + latin felismerő (GPU-n ~1–2 s/régió),
  sorgeometriával és sorkonfidenciával; a gerinc melletti összenyomott sorokat és a sorvégi
  elválasztójeleket sokkal jobban olvassa, mint a Tesseract.
- `ensemble` OCR motor: PP-OCRv5 szöveg + Tesseract szavanként összeigazítva; a Tesseract csak
  javít (gondolatjel, amit a PP-OCR karakterkészlete nem ismer; ékezet és egybeolvadt szavak
  szótári döntéssel), új szót sosem ad hozzá.
- Hunspell-alapú utófeldolgozás (`spell`): csak egyértelmű magánhangzó-tévesztést javít
  (ú/ű, ó/ő…, legfeljebb 2 betű, pontosan egy szótári jelölt); szótárral ellenőrzött
  elválasztás-összevonás; könyvspecifikus szólista (`spell.words`).
- Minőségi riport: `out/<slug>/quality.md` (oldalak az ismeretlen szavak aránya és a bizonytalan
  sorok szerint, leggyakoribb ismeretlen szavak) és `quality` mező az oldalankénti JSON-ban.

### Changed
- OCR-kivágás: a `crop_pad_x` ráhagyás mindkét oldalra kerül (vetemedett szkenneken a külső
  margónál is levágódott a sorvégi kötőjel); a ráhagyásba lógó *minden* más régió (szomszéd hasáb,
  kép, oldalszám) kifehérítve.
- Sorösszevonás: a kötőjel utáni kósza írásjel eldobva (`tudá-,`); szótárral a lap szélén elveszett
  kötőjel pótolva (`jelentke` + `zik` → `jelentkezik`).
- Codex: `ocr.default: ensemble`, könyvspecifikus `spell.words`.
- Függőség: `opencv-python-headless` helyett `opencv-contrib-python` – a PaddleX ezt rögzíti, és a két
  OpenCV-csomag egymás `cv2/` fájljait írta felül.
- A sidebar-keretjelöltek közül kiesnek a képekre/ornamentumokra eső és a tartalomdoboz szélét érintő
  jelöltek (díszes oldalszegély nem lesz „sidebar”).
- Heurisztikus layout: soralapú csoportosítás (sorfragmensek → hasábos/teljes szélességű sorok → blokkok),
  adaptív szóköz-becslés.
- `paddle` extra: `paddleocr[doc-parser]`; a PaddleOCR-VL motor nem tölti be a pipeline saját
  layout/orientáció almodelljeit.
