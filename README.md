# bookpecker

Szkennelt könyv-PDF-ekből (elsősorban magyar nyelvű RPG-könyvekből) tiszta Markdown, **teljesen lokálisan**
futó OCR-rel.

- 1, 2, 3… hasábos és **vegyes** oldalak (teljes szélességű cím/bekezdés + hasábok) helyes olvasási sorrendben
- lapszéli **ornamentumok**, élőfej, oldalszám, a szomszéd oldalról belógó szöveg eldobása (oldalparitás szerint
  tükrözött margómaszk + layout-modell címkék)
- keretes / színezett hátterű **sidebarok** → Markdown idézetblokk
- **táblázatok** (VLM motorral Markdown/HTML táblaként)
- képek, illusztrációk: csak felismerjük, hogy ne legyen belőlük szemét-szöveg – nem mentjük ki őket
- sortörés **csak bekezdéshatáron**; mondat közben soha. Az elválasztott szavak összeforrasztva
  (`var-⏎ázsló` → `varázsló`, de `kis- és nagybetű`, `Kard-Mágia` marad)
- címsorok külön blokkban, alapból egységesen `##` (opcionálisan betűméret alapján `#`/`##`/`###`)
- több könyv, könyvenként (és oldaltartományonként) külön beállításokkal
- lépcsőzetes, cache-elt pipeline: beállítás módosítása után csak az érintett lépések futnak újra

## Pipeline

```
PDF ─ rasterize ─ preprocess ─ layout ─ filter ─ reading order ─ OCR ─ assemble ─ Markdown
        300 DPI    deskew,      PP-DocLayout    margómaszk,   hasábok,     régiónként   bekezdések,
                   normalizálás vagy heurisztika drop_kinds    sidebarok    motorral     elválasztás
                                                 └──── debug overlay PNG ────┘
```

| Könyvtár | Tartalom |
|---|---|
| `books/<slug>.yaml` (vagy `books/<slug>/book.yaml`) | könyv beállításai (verziókezelt) |
| `work/<slug>/raw, prep` | raszterizált és előfeldolgozott oldalak |
| `work/<slug>/layout/NNNN.json` | régiók (típus, bbox, sorrend, eldobás oka) |
| `work/<slug>/debug/NNNN.png` | **overlay a hangoláshoz** |
| `work/<slug>/ocr/NNNN.json` | OCR eredmény régiónként, sorgeometriával |
| `out/<slug>/book.md` | a teljes könyv |
| `out/<slug>/pages/NNNN.md, .json` | oldalanként (a JSON-ban régiók + szöveg) |

A `work/`, `out/` és a PDF-ek nincsenek a repóban.

## Telepítés

Rendszercsomagok (Arch/Manjaro):

```sh
sudo pacman -S --needed uv tesseract tesseract-data-hun
```

Virtualenv a projektben (Python 3.12 – a PaddlePaddle a legújabb Pythont még nem támogatja):

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"             # alap: heurisztikus layout + Tesseract
```

Opcionális, modell alapú layout (PP-DocLayout) és VLM OCR (PaddleOCR-VL):

```sh
# GPU (CUDA 12.6 build; GTX 1650-en is fut) – vagy CPU: uv pip install paddlepaddle
uv pip install "paddlepaddle-gpu==3.3.1" --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install -e ".[paddle]"
```

A modelleket a PaddleOCR első használatkor tölti le (`~/.paddlex`).

## Gyors kezdés

```sh
bookpecker new-book kalandok --pdf ~/Scans/kalandok.pdf --title "Kalandok könyve"
$EDITOR books/kalandok.yaml              # margók, hasábok, kihagyott oldalak…
bookpecker layout kalandok -p 10-15      # csak layout → work/kalandok/debug/*.png megnézése
bookpecker run kalandok -p 10-15         # OCR + Markdown ezekre az oldalakra
bookpecker run kalandok                  # az egész könyv
bookpecker run --all                     # minden könyv a books/ alatt
```

Egyéb parancsok:

| Parancs | Mit csinál |
|---|---|
| `bookpecker info <slug>` | összefoglaló (oldalszám, kihagyott oldalak) |
| `bookpecker info <slug> --page 37` | az adott oldal teljesen feloldott beállításai |
| `bookpecker run <slug> --from layout` | újrafuttatás a megadott lépéstől (`rasterize`, `preprocess`, `layout`, `ocr`) |
| `bookpecker run <slug> --force` | minden lépés újra |
| `bookpecker run <slug> --ocr-engine paddle_vl` | OCR motor felülírása (összehasonlításhoz) |
| `bookpecker assemble <slug>` | csak a Markdown újraépítése a cache-elt OCR-ből |

## Könyv beállítása (`books/<slug>.yaml`)

Minta: [`books/_template.yaml`](books/_template.yaml). Projekt szintű alapértékek: `config/defaults.yaml`.
Oldalszámok mindig **1-alapú PDF-oldalindexek**.

| Kulcs | Alapérték | Leírás |
|---|---|---|
| `pdf` | – | a szkennelt PDF (a könyv mappájához, a projekt gyökeréhez képest, vagy abszolút) |
| `title` | slug | a `book.md` főcíme |
| `first_page_side` | `right` | az 1. PDF-oldal jobb (recto) vagy bal (verso) oldal; ebből jön a paritás |
| `skip_pages` | `[]` | kihagyott oldalak: `[1, "3-5", "250-"]` – a paritást nem tolják el |
| `page_number_offset` | `0` | nyomtatott oldalszám = PDF-oldal + offset (csak a `<!-- page -->` jelölőkhöz) |
| `dpi` | `300` | raszterizálás felbontása |
| `margins` | `top .05, bottom .05, inner .04, outer .05` | eldobott lapszélek az oldalméret arányában; `inner` = gerinc felőli, oldalparitás szerint tükrözve |
| `columns` | `auto` | hasábszám; `auto` = a hasábközök detektálása |
| `full_width_ratio` | `0.6` | ennél szélesebb (vagy hasábközt átérő) régió teljes szélességűnek számít |
| `layout_engine` | `doclayout` | `doclayout` (PP-DocLayout modell) vagy `heuristic` (modell nélkül) |
| `drop_kinds` | `[header, footer, page_number, ornament, figure]` | eldobott régiótípusok |
| `sidebars.lines / shading` | `true / true` | keretes és színezett hátterű dobozok keresése |
| `sidebars.placement` | `inline` | `inline` (a helyén) vagy `page_end` (az oldal szövege után) |
| `ocr.default` | `tesseract` | OCR motor; típusonként felülírható: `ocr.text`, `ocr.title`, `ocr.table`, `ocr.caption`, `ocr.sidebar` |
| `languages` | `[hun]` | Tesseract nyelvek |
| `heading_levels` | `flat` | `flat`: minden cím `##`; `auto`: betűméret-klaszterek (`#`–`###`), bizonytalan esetben `flat` |
| `edge_touch` | `0.01` | a lap szélét (ennyi arányon belül) érintő régiók eldobása – a szomszéd oldalról belógó szöveg, szegélyek |
| `crop_pad_x` | `0.012` | OCR-ráhagyás a régiók gerinc felőli oldalán (a layout-dobozok ott gyakran levágják a sűrített sorokat) |
| `replacements` | `[]` | könyvspecifikus regex-javítások bekezdésekre/címekre: `[{pattern: '…', repl: '…'}]` (multiline) |
| `tesseract.min_word_conf` | `30` | számjegy nélküli szavak ez alatti konfidenciával kiesnek (ornamentum-zaj) |
| `preprocess.flatten` | `true` | megvilágítás kiegyenlítése (gerincárnyék) |
| `preprocess.deskew` | `true` | ferdeség-korrekció (±`max_skew_deg`) |
| `page_overrides` | `{}` | oldaltartományonkénti felülírás, bármely fenti oldal szintű kulccsal, pl. `"37-42": {columns: 1}`, vagy `"88": {side: left}` ha egy hiányzó lap elrontja a paritást |

### Hangolás

1. `bookpecker layout <slug> -p <néhány jellemző oldal>`, majd nézd meg a `work/<slug>/debug/*.png` képeket:
   - piros keret: margómaszk (ami kívül esik, az kiesik); piros áthúzás: eldobott régió + ok
     (`margin`, `kind:header`, `tiny`, `empty`…)
   - színes keretek: megtartott régiók `sorrend:típus` felirattal, `gN` = sidebar-csoport; lila: sidebar keret
2. Állítsd a `margins` értékeket úgy, hogy az ornamentumok és oldalszámok kívül essenek, de a szöveg ne.
3. Ha egyes oldalak eltérnek (pl. egyhasábos fejezetnyitó), használd a `page_overrides`-t.
4. `bookpecker run <slug> -p …` → `out/<slug>/pages/*.md` ellenőrzése.

## OCR motorok

| Motor | Erősség | Megjegyzés |
|---|---|---|
| `tesseract` | gyors, CPU, jó magyar modell, sorgeometriát ad (bekezdés-felismeréshez) | táblázatot csak soronként |
| `paddle_vl` | PaddleOCR-VL (0.9B VLM): **táblázat → Markdown/HTML**, sorvégi elválasztást maga összevonja | **kísérleti**; nincs sorgeometria, a bekezdéseket a modell üres sorai és az írásjelek alapján bontjuk |

Mért tapasztalat (Codex alapkönyv, GTX 1650 4 GB):

- A PaddleOCR-VL bf16 súlyokkal jön; a Turing GPU nem tud bf16-ot, fp32-ben pedig nem fér el 4 GB-ban
  (fp16 kényszerítéssel sem) → csak **CPU-n** fut, ~20–40 s régiónként (~15 perc/oldal).
  Magyar ékezetekben gyengébb a Tesseractnál („mozgatörugókkal”), a `%` jelet viszont jól olvassa.
- Ezért a javasolt beállítás: `ocr.default: tesseract`; VLM legfeljebb táblákra (`ocr: {table: paddle_vl}`,
  `paddle_vl: {device: cpu}`).
- A PP-DocLayout layout-modell GPU-n ~1–2 s/oldal, a teljes pipeline Tesseracttal ~15 s/oldal.

## Fejlesztés

```sh
source .venv/bin/activate
pytest
```

A tesztek (config-feloldás, paritás/margótükrözés, olvasási sorrend, bekezdés- és elválasztáskezelés,
Markdown összeállítás, szintetikus oldal heurisztikus layouttal) nem igényelnek modellt vagy Tesseractot.

Változások: [CHANGELOG.md](CHANGELOG.md).
