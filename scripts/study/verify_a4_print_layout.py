#!/usr/bin/env python3
from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path('notes/study/interactive').resolve()
CHROME = Path(os.environ.get('GM_CHROME_PATH', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))
ROUTE = '#/a4-master-sheet'
A4_WIDTH_PT = 595
A4_HEIGHT_PT = 842
TOLERANCE_PT = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.settimeout(0.25)
                sock.connect(('127.0.0.1', port))
                return
            except OSError:
                time.sleep(0.05)
    raise RuntimeError(f'local HTTP server did not start on port {port}')


def start_server(port: int) -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(ROOT), **kwargs)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    wait_for_server(port)
    return server, thread


def run_chrome_print(url: str, output_pdf: Path) -> None:
    require(CHROME.is_file(), f'Google Chrome not found at {CHROME}')
    cmd = [
        str(CHROME),
        '--headless=new',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--no-first-run',
        '--no-default-browser-check',
        '--run-all-compositor-stages-before-draw',
        '--virtual-time-budget=2500',
        f'--print-to-pdf={output_pdf}',
        '--print-to-pdf-no-header',
        url,
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    require(result.returncode == 0, f'Chrome print failed with exit {result.returncode}:\n{result.stdout}')
    require(output_pdf.is_file() and output_pdf.stat().st_size > 1000, f'Chrome did not create a usable PDF: {output_pdf}')


def assert_print_content_fits(url: str, work_dir: Path) -> dict:
    """Use Chrome DevTools to emulate print media and fail on hidden overflow."""
    require(shutil.which('node'), 'Node.js is required for Chrome DevTools print fit verification')
    debug_port = free_port()
    profile_dir = work_dir / 'chrome-profile'
    cmd = [
        str(CHROME),
        '--headless=new',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--no-first-run',
        '--no-default-browser-check',
        f'--remote-debugging-port={debug_port}',
        f'--user-data-dir={profile_dir}',
        url,
    ]
    chrome = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    node_script = r'''
const debugPort = Number(process.argv[1]);

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(url, attempts = 80) {
  let lastError;
  for (let index = 0; index < attempts; index += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(100);
  }
  throw lastError || new Error(`Could not fetch ${url}`);
}

async function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const pending = new Map();
  let nextId = 1;
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    }
  };
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error('Chrome DevTools websocket failed'));
  });
  return {
    send(method, params = {}) {
      const id = nextId++;
      ws.send(JSON.stringify({ id, method, params }));
      return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
    },
    close() {
      ws.close();
    },
  };
}

const tabs = await fetchJson(`http://127.0.0.1:${debugPort}/json`);
const tab = tabs.find((entry) => entry.type === 'page' && entry.webSocketDebuggerUrl);
if (!tab) throw new Error('No debuggable page target found');
const client = await connect(tab.webSocketDebuggerUrl);
try {
  await client.send('Page.enable');
  await client.send('Runtime.enable');
  await client.send('Emulation.setEmulatedMedia', { media: 'print' });
  for (let index = 0; index < 80; index += 1) {
    const ready = await client.send('Runtime.evaluate', {
      expression: 'document.readyState === "complete" && document.querySelectorAll(".a4-side").length === 2',
      returnByValue: true,
    });
    if (ready.result.value) break;
    await sleep(100);
  }
  await client.send('Runtime.evaluate', {
    expression: 'new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))',
    awaitPromise: true,
  });
  const fit = await client.send('Runtime.evaluate', {
    returnByValue: true,
    expression: `(() => {
      const tolerance = 2;
      const sideData = [...document.querySelectorAll('.a4-side')].map((side, index) => {
        const overflowingCells = [...side.querySelectorAll('.a4-cell')]
          .filter((cell) => cell.scrollHeight > cell.clientHeight + tolerance || cell.scrollWidth > cell.clientWidth + tolerance)
          .map((cell) => ({
            title: cell.querySelector('h3')?.textContent || cell.className,
            overflowY: Math.max(0, cell.scrollHeight - cell.clientHeight),
            overflowX: Math.max(0, cell.scrollWidth - cell.clientWidth),
          }));
        return {
          index,
          title: side.querySelector('h2')?.textContent || side.getAttribute('aria-label') || String(index + 1),
          clientHeight: side.clientHeight,
          scrollHeight: side.scrollHeight,
          clientWidth: side.clientWidth,
          scrollWidth: side.scrollWidth,
          overflowY: Math.max(0, side.scrollHeight - side.clientHeight),
          overflowX: Math.max(0, side.scrollWidth - side.clientWidth),
          overflowingCells,
        };
      });
      return {
        route: location.hash,
        readyState: document.readyState,
        sideCount: sideData.length,
        sides: sideData,
        ok: sideData.length === 2 && sideData.every((side) =>
          side.overflowY <= tolerance &&
          side.overflowX <= tolerance &&
          side.overflowingCells.length === 0
        ),
      };
    })()`,
  });
  console.log(JSON.stringify(fit.result.value));
} finally {
  client.close();
}
'''
    try:
        result = subprocess.run(
            ['node', '--input-type=module', '-e', node_script, str(debug_port)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        require(result.returncode == 0, f'print fit inspection failed with exit {result.returncode}:\n{result.stdout}')
        fit = json.loads(result.stdout.strip().splitlines()[-1])
        require(fit.get('sideCount') == 2, f'expected 2 printable A4 sides in DOM, got {fit.get("sideCount")}: {fit}')
        require(fit.get('ok'), f'print content overflows A4 side box: {json.dumps(fit, indent=2)}')
        return fit
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
            chrome.wait(timeout=5)


def pdf_info_with_pdfinfo(pdf_path: Path) -> tuple[int, float, float] | None:
    pdfinfo = shutil.which('pdfinfo') or '/opt/homebrew/bin/pdfinfo'
    if not pdfinfo or not Path(pdfinfo).is_file():
        return None
    result = subprocess.run([pdfinfo, str(pdf_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    require(result.returncode == 0, f'pdfinfo failed with exit {result.returncode}:\n{result.stdout}')
    pages_match = re.search(r'^Pages:\s+(\d+)$', result.stdout, flags=re.M)
    size_match = re.search(r'^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts', result.stdout, flags=re.M)
    require(pages_match is not None, f'pdfinfo output missing page count:\n{result.stdout}')
    require(size_match is not None, f'pdfinfo output missing page size:\n{result.stdout}')
    return int(pages_match.group(1)), float(size_match.group(1)), float(size_match.group(2))


def pdf_info_with_pypdf(pdf_path: Path) -> tuple[int, float, float]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError('Neither pdfinfo nor pypdf is available for PDF inspection') from exc
    reader = PdfReader(str(pdf_path))
    first = reader.pages[0].mediabox
    return len(reader.pages), float(first.width), float(first.height)


def assert_a4_two_pages(pdf_path: Path) -> tuple[int, float, float]:
    info = pdf_info_with_pdfinfo(pdf_path) or pdf_info_with_pypdf(pdf_path)
    pages, width, height = info
    require(pages == 2, f'expected exactly 2 PDF pages for double-sided A4, got {pages}')
    require(abs(width - A4_WIDTH_PT) <= TOLERANCE_PT, f'expected A4 width around {A4_WIDTH_PT}pt, got {width:.2f}pt')
    require(abs(height - A4_HEIGHT_PT) <= TOLERANCE_PT, f'expected A4 height around {A4_HEIGHT_PT}pt, got {height:.2f}pt')
    return info


def main() -> None:
    require(ROOT.is_dir(), f'missing static site root: {ROOT}')
    port = free_port()
    server, _thread = start_server(port)
    try:
        with tempfile.TemporaryDirectory(prefix='gm-a4-print-') as tmp:
            tmp_path = Path(tmp)
            pdf_path = Path(tmp) / 'a4-master-sheet.pdf'
            url = f'http://127.0.0.1:{port}/{ROUTE}'
            fit = assert_print_content_fits(url, tmp_path)
            run_chrome_print(url, pdf_path)
            pages, width, height = assert_a4_two_pages(pdf_path)
            side_heights = ','.join(f'{side["scrollHeight"]}/{side["clientHeight"]}' for side in fit.get('sides', []))
            print(f'A4 PRINT VALIDATION PASSED pages={pages} size={width:.2f}x{height:.2f}pt side_scroll/client={side_heights} bytes={pdf_path.stat().st_size}')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'A4 PRINT VALIDATION FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
