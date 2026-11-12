# bookpecker

Szkennelt könyv-PDF-ekből (elsősorban magyar nyelvű RPG-könyvekből) tiszta Markdown, **teljesen lokálisan**
futó OCR-rel.

- 1, 2, 3… hasábos és **vegyes** oldalak (teljes szélességű cím/bekezdés + hasábok) helyes olvasási sorrendben
- lapszéli **ornamentumok**, élőfej, oldalszám, a szomszéd oldalról belógó szöveg eldobása (oldalparitás szerint
  tükrözött margómaszk + layout-modell címkék)
- keretes / színezett hátterű **sidebarok** → Markdown idézetblokk
- **táblázatok** (VLM motorral Markdown/HTML táblaként)
- **két OCR motor együtt** (`ensemble`): PP-OCRv5 + Tesseract szavanként összeigazítva, a vitás
  szavakat a magyar Hunspell-szótár dönti el; szótári ékezetjavítás és **minőségi riport**
  (`out/<slug>/quality.md`: mely oldalakat érdemes átnézni)
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
| `out/<slug>/quality.md` | OCR-minőség oldalanként (ismeretlen szavak aránya, bizonytalan sorok) |
| `out/<slug>/pages/NNNN.md, .json` | oldalanként (a JSON-ban régiók + szöveg) |

A `work/`, `out/` és a PDF-ek nincsenek a repóban.

## Telepítés

Rendszercsomagok (Arch/Manjaro):

```sh
sudo pacman -S --needed uv tesseract tesseract-data-hun hunspell hunspell-hu
```

Virtualenv a projektben (Python 3.12 – a PaddlePaddle a legújabb Pythont még nem támogatja):

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"             # alap: heurisztikus layout + Tesseract
```

Opcionális, de **ajánlott**: modell alapú layout (PP-DocLayout), PP-OCRv5 (`paddle_ocr`, `ensemble`)
és VLM OCR (PaddleOCR-VL):

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
| `bookpecker run <slug> -j 6` | párhuzamos CPU-szálak száma (alap: CPU-szám − 2; `-j 1` = soros) |

### Párhuzamos futás

A `run` és a `layout` alapból párhuzamosan dolgozik, hogy a GPU és a CPU-magok egyszerre legyenek
kihasználva:

- a következő oldalak raszterizálása + előfeldolgozása külön folyamatokban előre fut;
- a layout és a PP-OCRv5 a fő szálon, oldalanként fut a GPU-n (a modell egyszer van a VRAM-ban);
- a Tesseract régiónként egy szálkészletben fut (egyszálú tesseract-folyamatok), a GPU-munkával
  egy időben;
- a két motor összefésülése és az oldal mentése külön szálon történik, közben a GPU már a következő
  oldalon dolgozik.

Az eredmény bájtra azonos a soros futáséval. Mérés (Codex, 20 oldal, layout → Markdown, i7-9750H +
GTX 1650): soros 303 s, párhuzamos 125 s. A GPU-kihasználtság ettől sem lesz magas: a modellek kicsik,
a kártya oldalanként csak néhány másodpercig dolgozik.

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
| `margin_notes.enabled` | `false` | széljegyzetek: a fő szövegtömb mellett álló keskeny (`max_width`, a tartalom szélességének 30%-a alatti) szövegblokkok egy idézetblokkba, alapból az oldal végére (`placement`) |
| `ocr.default` | `tesseract` | OCR motor (`ensemble`, `paddle_ocr`, `tesseract`, `paddle_vl`); típusonként felülírható: `ocr.text`, `ocr.title`, `ocr.table`, `ocr.caption`, `ocr.sidebar` |
| `paddle_ocr.device` | paddle alap | `gpu:0` / `cpu`; `min_score: 0.5` alatti sorok kiesnek |
| `spell.enabled` | `true` | Hunspell-utófeldolgozás és minőségi riport (`hunspell` + `hunspell-hu` kell) |
| `spell.words` | `[]` | nevek, játékkifejezések; ragozott alakjaik is ismertnek számítanak (`mentalior` → `mentaliorban`) |
| `spell.fix_diacritics` / `spell.dehyphenate` | `true` / `true` | ékezetjavítás; szótárral ellenőrzött elválasztás-összevonás |
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
5. `out/<slug>/quality.md`: a legrosszabb oldalak elöl; a gyakori ismeretlen neveket/kifejezéseket
   vedd fel a `spell.words` listába (utána elég a `bookpecker assemble <slug>`).

## OCR motorok

| Motor | Erősség | Megjegyzés |
|---|---|---|
| `ensemble` | **a legmegbízhatóbb**: PP-OCRv5 szöveg + Tesseract javítások | ~25–35 s/oldal (GTX 1650) |
| `paddle_ocr` | PP-OCRv5 (sordetektálás + latin felismerő), GPU-n gyors; a gerinc melletti összenyomott sorokat és a sorvégi elválasztást jól olvassa | nincs en dash (–) a karakterkészletében; ű/ő helyett néha ú/ó (a `spell` lépés javítja) |
| `tesseract` | gyors, CPU, jó magyar modell, sorgeometriát ad | a gerinc felé összenyomódó sorokon gyenge („fel2”, „Veszé: lyérzet”); táblázatot csak soronként |
| `paddle_vl` | PaddleOCR-VL (0.9B VLM): **táblázat → Markdown/HTML** | **kísérleti**; nincs sorgeometria |

### Az `ensemble` működése

A PP-OCRv5 adja a szöveget és a sorgeometriát; a Tesseract ugyanannak a régiónak ugyanarra a
vizuális sorára eső szavait ékezet- és írásjel-független kulcsok alapján összeigazítjuk vele. A
Tesseract **csak javít, új szót nem tesz hozzá**:

- gondolatjel, amelyet csak a Tesseract látott (`fejfájást – gyengébb`, `8-9`), sorvégi kötőjel,
  amelyet a PP-OCR pontnak olvasott vagy elvesztett;
- ugyanaz a szó más ékezettel: a szótárban szereplő alak nyer (`túnik` → `tűnik`), a magyarban nem
  létező betű (`ý`, `ā`) helyett a Tesseracté;
- egybeolvadt szavak (`nagyhatásúellenmérgek`) szétválasztása, ha a Tesseract betűre ugyanazt
  olvasta szóközzel;
- félreolvasott szó (`lőfegóver`) helyett a hasonló, szótárban szereplő Tesseract-szó; ha a
  PP-OCR maga is bizonytalan volt a sorban (konfidencia < 0,85), lazább hasonlósági küszöbbel;
- `l`/`1`, `O`/`0` tévesztés, ha a Tesseract számjegyet olvasott.

Utána a `spell` lépés: ékezetjavítás csak akkor, ha legfeljebb két magánhangzó cseréje a saját
csoportján belül (a/á, e/é, i/í, o/ó/ö/ő, u/ú/ü/ű) **pontosan egy** szótári szót ad; elválasztás
összevonásakor a szótár dönt (`kard-` + `mágia` összetétel marad; a lap szélén elveszett kötőjel
esetén `jelentke` + `zik` → `jelentkezik`).

Mért eredmény (Codex, 6 vegyes oldal: 20–21, 50–51, 120–121; szótárban nem szereplő szavak,
könyvspecifikus szólista nélkül): Tesseract **97**, ensemble **73**; a riport „bizonytalan sor”
oszlopa az ensemble-nél szinte mindenhol 0.

Tapasztalat (GTX 1650 4 GB):

- A PaddleOCR-VL bf16 súlyokkal jön; a Turing GPU nem tud bf16-ot, fp32-ben pedig nem fér el 4 GB-ban
  → csak **CPU-n** fut (~15 perc/oldal), ezért legfeljebb táblákra javasolt
  (`ocr: {table: paddle_vl}`, `paddle_vl: {device: cpu}`).
- A PP-DocLayout layout-modell GPU-n ~1–2 s/oldal; a PP-OCRv5 modelleket a PaddleOCR első
  használatkor tölti le (~/.paddlex). `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` kihagyja az
  indításkori hálózati ellenőrzést.

## Fejlesztés

```sh
source .venv/bin/activate
pytest
```

A tesztek (config-feloldás, paritás/margótükrözés, olvasási sorrend, bekezdés- és elválasztáskezelés,
Markdown összeállítás, ensemble-összeigazítás, szintetikus oldal heurisztikus layouttal) nem igényelnek
modellt vagy Tesseractot; a Hunspell-tesztek kimaradnak, ha nincs telepítve.

Változások: [CHANGELOG.md](CHANGELOG.md).
