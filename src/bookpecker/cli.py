"""Command line interface."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import config_path, find_root, list_books, load_book
from .ocr.tesseract import limit_threads
from .pipeline import STAGES, assemble_book, process_page

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Scanned book PDFs → Markdown with local OCR.")

RootOpt = typer.Option(None, "--root", help="Project root (dir containing books/). Default: auto.")
PagesOpt = typer.Option(None, "--pages", "-p", help='PDF pages, e.g. "10-20,33".')
WorkersOpt = typer.Option(0, "--workers", "-j",
                          help="Parallel CPU workers (Tesseract, preprocessing); 0 = CPU count − 2, 1 = serial.")


def _root(root: Optional[Path]) -> Path:
    return root.resolve() if root else find_root()


def _books(root: Path, slugs: list[str], all_: bool) -> list[str]:
    if all_:
        return list_books(root)
    if not slugs:
        raise typer.BadParameter("give book slug(s) or --all")
    return slugs


def _failed(page: int, e: BaseException) -> None:
    typer.secho(f"  p{page:04d} FAILED: {e}", fg=typer.colors.RED, err=True)


def _process_serial(book, todo: list[int], until: str, force_from: Optional[str],
                    ocr_engine: Optional[str]) -> None:
    for page in todo:
        try:
            process_page(book, page, force_from=force_from, ocr_engine=ocr_engine, until=until,
                         log=typer.echo)
        except Exception as e:  # keep going; one bad page must not stop a whole book
            _failed(page, e)


def _process_parallel(book, todo: list[int], until: str, force_from: Optional[str],
                      ocr_engine: Optional[str], workers: int) -> None:
    """Keep the GPU busy and the CPU cores working at the same time.

    - rasterize + preprocess of upcoming pages: separate processes (spawned – the main process
      holds CUDA state that must not be forked);
    - layout and GPU OCR: this thread, page by page;
    - Tesseract per region: a thread pool (each call is its own tesseract process);
    - waiting for Tesseract, merging, writing the page: one finisher thread, so the next page's
      GPU work starts meanwhile.
    """
    early = [s for s in STAGES if STAGES.index(s) <= STAGES.index("preprocess")]
    # stages the prefetch already recomputed must not be forced again here
    main_force = force_from if force_from not in early else ("layout" if until not in early else None)
    prep_workers = max(1, min(4, workers // 3))
    in_flight = max(2, workers // 4)  # pages whose Tesseract work may overlap
    limit_threads(1)  # many single-threaded tesseracts scale better than a few multi-threaded ones
    try:
        _pipeline_pages(book, todo, until, force_from, main_force, ocr_engine, workers, prep_workers, in_flight)
    finally:
        limit_threads(None)


def _pipeline_pages(book, todo, until, force_from, main_force, ocr_engine, workers, prep_workers,
                    in_flight) -> None:
    early = [s for s in STAGES if STAGES.index(s) <= STAGES.index("preprocess")]
    with ProcessPoolExecutor(prep_workers, mp_context=mp.get_context("spawn")) as prep, \
            ThreadPoolExecutor(workers, thread_name_prefix="tess") as cpu, \
            ThreadPoolExecutor(1, thread_name_prefix="finish") as finisher:
        prefetch = {page: prep.submit(process_page, book, page, force_from, None,
                                      "preprocess" if until not in early else until)
                    for page in todo}
        pending: deque[tuple[int, Future]] = deque()

        def drain(limit: int) -> None:
            while len(pending) > limit:
                page, fut = pending.popleft()
                try:
                    fut.result()
                except Exception as e:
                    _failed(page, e)

        for page in todo:
            try:
                prefetch.pop(page).result()
                if until in early:
                    continue
                complete = process_page(book, page, force_from=main_force, ocr_engine=ocr_engine,
                                        until=until, log=typer.echo, pool=cpu)
            except Exception as e:
                _failed(page, e)
                continue
            if complete is not None:
                pending.append((page, finisher.submit(complete)))
            drain(in_flight)
        drain(0)


def _run(slugs: list[str], all_: bool, pages: Optional[str], root: Optional[Path],
         until: str, force_from: Optional[str], ocr_engine: Optional[str], assemble: bool,
         workers: int = 0) -> None:
    r = _root(root)
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 2)
    for slug in _books(r, slugs, all_):
        book = load_book(r, slug)
        todo = book.pages(pages)
        typer.echo(f"[{slug}] {book.title}: {len(todo)} page(s), {workers} worker(s)")
        t0 = time.time()
        if workers > 1 and len(todo) > 1:
            _process_parallel(book, todo, until, force_from, ocr_engine, workers)
        else:
            _process_serial(book, todo, until, force_from, ocr_engine)
        if assemble:
            out = assemble_book(book, book.pages(pages), log=typer.echo)
            typer.echo(f"[{slug}] → {out}")
        typer.echo(f"[{slug}] done in {time.time() - t0:.1f}s")


@app.command()
def run(
    slugs: list[str] = typer.Argument(None, help="Book slug(s) under books/"),
    all_: bool = typer.Option(False, "--all", help="All books"),
    pages: Optional[str] = PagesOpt,
    force: bool = typer.Option(False, "--force", help="Recompute every stage"),
    from_stage: Optional[str] = typer.Option(None, "--from", help=f"Recompute from stage: {STAGES}"),
    ocr_engine: Optional[str] = typer.Option(None, "--ocr-engine", help="Override OCR engine for all kinds"),
    workers: int = WorkersOpt,
    root: Optional[Path] = RootOpt,
):
    """Full pipeline: rasterize → preprocess → layout → OCR → Markdown."""
    if from_stage and from_stage not in STAGES:
        raise typer.BadParameter(f"--from must be one of {STAGES}")
    _run(slugs, all_, pages, root, "ocr", "rasterize" if force else from_stage, ocr_engine, True, workers)


@app.command()
def layout(
    slugs: list[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all"),
    pages: Optional[str] = PagesOpt,
    force: bool = typer.Option(False, "--force", help="Recompute layout even if cached"),
    workers: int = WorkersOpt,
    root: Optional[Path] = RootOpt,
):
    """Stop after layout: writes work/<book>/debug/NNNN.png overlays for tuning book.yaml."""
    _run(slugs, all_, pages, root, "layout", "layout" if force else None, None, False, workers)


@app.command()
def assemble(
    slugs: list[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all"),
    pages: Optional[str] = PagesOpt,
    root: Optional[Path] = RootOpt,
):
    """Only rebuild Markdown from cached OCR results."""
    r = _root(root)
    for slug in _books(r, slugs, all_):
        book = load_book(r, slug)
        typer.echo(f"[{slug}] → {assemble_book(book, book.pages(pages), log=typer.echo)}")


@app.command()
def info(
    slug: str,
    page: Optional[int] = typer.Option(None, "--page", help="Show resolved config for this page"),
    root: Optional[Path] = RootOpt,
):
    """Book summary, or the fully resolved config of one page."""
    book = load_book(_root(root), slug)
    if page is not None:
        cfg = book.page_config(page)
        typer.echo(yaml.safe_dump(json.loads(cfg.model_dump_json()), allow_unicode=True, sort_keys=False))
        return
    skipped = sorted(book.skipped())
    typer.echo(f"title:   {book.title}\npdf:     {book.pdf}\npages:   {book.page_count} "
               f"({len(skipped)} skipped)\nfirst page side: {book.first_page_side}")


def _set_yaml_value(text: str, key: str, value: str) -> str:
    """Replace a top-level `key:` line of the template, keeping its trailing comment."""
    line = re.compile(rf'^{key}:[ \t]*("[^"\n]*"|[^#\n]*?)([ \t]*#.*)?$', re.M)
    quoted = json.dumps(value, ensure_ascii=False)
    if line.search(text):
        return line.sub(lambda m: f"{key}: {quoted}{m.group(2) or ''}", text, count=1)
    return f"{key}: {quoted}\n{text}"


@app.command("new-book")
def new_book(
    slug: str,
    pdf: Path = typer.Option(..., "--pdf", exists=True, dir_okay=False, help="Scanned PDF"),
    title: Optional[str] = typer.Option(None, "--title"),
    root: Optional[Path] = RootOpt,
):
    """Create books/<slug>.yaml from books/_template.yaml."""
    r = _root(root)
    existing = config_path(r, slug)
    if existing:
        raise typer.BadParameter(f"{existing} already exists")
    target = r / "books" / f"{slug}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    template = r / "books" / "_template.yaml"
    text = template.read_text(encoding="utf-8") if template.exists() else ""
    text = _set_yaml_value(text, "pdf", str(pdf.resolve()))
    text = _set_yaml_value(text, "title", title or slug)
    target.write_text(text, encoding="utf-8")
    typer.echo(f"created {target}")


if __name__ == "__main__":
    app()
