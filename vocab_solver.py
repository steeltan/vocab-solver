# -*- coding: utf-8 -*-
"""
词汇测试秒答助手  v2
====================
英文单词 -> 中文意思。按 F9 识别，只给意思，不猜选项。

    F9   识别一次：截屏 -> OCR -> 查词 -> 悬浮窗显示意思
    F7   框选识别区域（强烈建议，区域越小越快；会保存到 vocab_solver.crop.json）
    F8   退出

速度策略：
  1. 词库内的词：字典直接命中，目标 < 1s（实测通常 0.25~0.4s）。
  2. 词库外的词：立刻显示「查词中…」，后台联网查词，总耗时硬性 <= 3s；
     结果写入 vocab_solver_cache.json，同一个词第二次直接命中缓存。
  3. 悬浮窗在独立线程里常驻复用，显示时不会卡住热键（v1 会阻塞 6 秒）。

依赖：pip install rapidocr-onnxruntime keyboard
"""

import csv
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher, get_close_matches

import keyboard
import numpy as np
from PIL import Image, ImageGrab

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------- 配置 ----------------
BASE = os.path.dirname(os.path.abspath(__file__))

VOCAB_CSV = os.path.join(BASE, "vocab_bank_大学以上.csv")
CACHE_FILE = os.path.join(BASE, "vocab_solver_cache.json")
CROP_FILE = os.path.join(BASE, "vocab_solver.crop.json")
LOG_FILE = os.path.join(BASE, "vocab_solver.log")
OCR_CONFIG = os.path.join(BASE, "ocr_dml.yaml")

HOTKEY_TRIGGER = "f9"
HOTKEY_REGION = "f7"
HOTKEY_QUIT = "f8"

# 识别区域 (left, top, right, bottom)，None 表示全屏（全屏很慢，建议 F7 框选）
CROP = None

# 时间预算（秒）
LOCAL_BUDGET = 1.0          # 词库命中目标
ONLINE_BUDGET = 3.0         # 从按下 F9 起，最晚出结果（含截图 + OCR）
ONLINE_CALL_TIMEOUT = 1.3   # 单个词典接口超时；两个接口加起来仍 <= 3s

POPUP_SECONDS = 5.0
DEBUG = False               # True 时每次保存 debug_last.png / debug_ocr.txt

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    try:
        print(line)
    except Exception:
        pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------- 词库 ----------------
def load_vocab(path):
    """返回 word -> 中文释义。用 csv 解析，兼容带引号的 "舒适的,舒服的"。"""
    word2cn = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)          # 跳过表头
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0].strip().lower()
            cn = row[1].strip()
            if word and cn:
                word2cn[word] = cn
    return word2cn


def load_cache():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_cache():
    try:
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(CACHE, f, ensure_ascii=False, indent=0, sort_keys=True)
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        log("保存缓存失败：%s" % e)


WORD2CN = load_vocab(VOCAB_CSV)
VOCAB_KEYS = list(WORD2CN)
CACHE = load_cache()

# ---------------- 文本处理 ----------------
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*[A-Za-z]|[A-Za-z]")
_POS_RE = re.compile(r"^\s*(?:[a-z]{1,6}\.\s*)+", re.I)
_BRACKET_RE = re.compile(r"[（(][^（()）]*[）)]")
_TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "－": "-"})


def norm_word(tok):
    """OCR 文本规整成词库键：小写、去弯引号、去首尾连字符。"""
    t = (tok or "").translate(_TRANS).strip().lower()
    return t.strip("-' ")


def clean_cn(s):
    """把在线词典的长释义压成一句中文。"""
    if not s:
        return ""
    s = s.replace("...", "").replace("…", "").strip()
    s = _POS_RE.sub("", s)                 # 去掉 n. / adj. / int. 之类词性前缀
    s = re.split(r"[;；]", s)[0]           # 只取第一组义项
    s = _BRACKET_RE.sub("", s)             # 去掉括号补充说明
    s = re.sub(r"\s+", " ", s).strip(" ，,。.;；:：\"'")
    if len(s) > 30:
        s = s[:30] + "…"
    return s


def _box_height(box):
    try:
        ys = [p[1] for p in box]
        return max(ys) - min(ys)
    except Exception:
        return 0


def extract_tokens(ocr_items):
    """OCR 结果 -> [(word, height, score)]，只保留英文 token。"""
    out = []
    for text, score, box in ocr_items:
        for m in _TOKEN_RE.finditer(text or ""):
            w = m.group(0)
            if len(w) >= 2:
                out.append((w, _box_height(box), float(score or 0)))
    return out


def pick_word(ocr_items):
    """挑出被测单词。

    返回 (word, cn, 是否词库命中, tokens)。
    优先：词库精确命中 -> 字号最大的未知词。
    """
    tokens = extract_tokens(ocr_items)

    hits = []
    for w, h, sc in tokens:
        key = norm_word(w)
        if key and key in WORD2CN:
            hits.append((h, sc, len(key), key))
    if hits:
        # 题目单词通常是字最大的那块；同大小时取 OCR 置信度高的
        hits.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        key = hits[0][3]
        return key, WORD2CN[key], True, tokens

    if tokens:
        # 未知词：同样按字号 > 置信度 > 长度排序，取最像题目单词的那个
        tokens.sort(key=lambda x: (-x[1], -x[2], -len(x[0])))
        return norm_word(tokens[0][0]), None, False, tokens

    return None, None, False, tokens


def fuzzy_local(word, min_sim=0.84):
    """未知词在本地找近似（多半是 OCR 少/错一个字母）。"""
    if not word or len(word) < 3:
        return None
    try:
        cand = get_close_matches(word, VOCAB_KEYS, n=1, cutoff=min_sim)
    except Exception:
        return None
    if not cand:
        return None
    w = cand[0]
    if abs(len(w) - len(word)) > 2:
        return None
    s = SequenceMatcher(None, word, w).ratio()
    if s < min_sim:
        return None
    return w, WORD2CN[w], s


# ---------------- 在线查词 ----------------
def _http_json(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json,text/plain,*/*"})
    with urllib.request.urlopen(req, timeout=max(0.3, timeout)) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _youdao(word, timeout):
    url = ("https://dict.youdao.com/suggest?num=3&doctype=json&q="
           + urllib.parse.quote(word))
    data = _http_json(url, timeout)
    for e in (data.get("data") or {}).get("entries") or []:
        if norm_word(e.get("entry", "")) == word:
            return clean_cn(e.get("explain", ""))
    return None


def _iciba(word, timeout):
    url = ("https://dict.iciba.com/dictionary/word/suggestion?nums=3&word="
           + urllib.parse.quote(word))
    data = _http_json(url, timeout)
    for m in data.get("message") or []:
        if norm_word(m.get("key", "")) == word:
            return clean_cn(m.get("paraphrase", ""))
    return None


PROVIDERS = (("有道", _youdao), ("金山", _iciba))


def online_lookup(word, deadline):
    """在 deadline（time.time() 绝对值）之前拿到中文释义，返回 (cn, 来源)。"""
    for name, fn in PROVIDERS:
        remain = deadline - time.time()
        if remain <= 0.2:
            break
        try:
            cn = fn(word, min(remain, ONLINE_CALL_TIMEOUT))
        except Exception:
            cn = None
        if cn:
            return cn, name
    return None, None


# ---------------- OCR ----------------
_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR
            log("加载 OCR 模型 (GPU/DirectML)...")
            _engine = RapidOCR(config_path=OCR_CONFIG)
            log("预热 GPU ...")
            _warm = Image.new("RGB", (420, 110), "white")
            _engine(np.array(_warm)[:, :, ::-1])
            log("OCR 就绪")
        return _engine


def do_ocr(img):
    engine = get_engine()
    arr = np.array(img)                 # RGB
    arr = arr[:, :, ::-1]               # -> BGR
    result, _ = engine(arr)
    out = []
    if result:
        for box, text, score in result:
            out.append((text.strip(), score, box))
    return out


def grab():
    if CROP:
        return ImageGrab.grab(bbox=CROP, all_screens=True)
    return ImageGrab.grab(all_screens=True)


# ---------------- 悬浮窗（常驻线程，绝不阻塞热键） ----------------
class Popup(object):
    def __init__(self):
        self.q = queue.Queue()
        self.ready = threading.Event()
        self.root = None
        self._hide_id = None
        threading.Thread(target=self._loop, daemon=True).start()
        self.ready.wait(6)

    def _loop(self):
        try:
            import tkinter as tk
        except Exception as e:
            log("tkinter 不可用：%s" % e)
            self.ready.set()
            return

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="#0e1216")

        frame = tk.Frame(root, bg="#0e1216", highlightthickness=2,
                         highlightbackground="#00e676")
        frame.pack(fill="both", expand=True)

        self.lbl_word = tk.Label(frame, text="", font=("Microsoft YaHei", 13),
                                 fg="#8fa3b0", bg="#0e1216", anchor="w",
                                 padx=22, pady=0)
        self.lbl_cn = tk.Label(frame, text="", font=("Microsoft YaHei", 22, "bold"),
                               fg="#00e676", bg="#0e1216", anchor="w",
                               padx=22, pady=2, justify="left",
                               wraplength=520)
        self.lbl_note = tk.Label(frame, text="", font=("Microsoft YaHei", 10),
                                 fg="#5c6b76", bg="#0e1216", anchor="w",
                                 padx=22, pady=0)
        self.lbl_word.pack(fill="x", pady=(10, 0))
        self.lbl_cn.pack(fill="x")
        self.lbl_note.pack(fill="x", pady=(0, 10))

        root.bind("<Button-1>", lambda e: root.withdraw())
        root.bind("<Escape>", lambda e: root.withdraw())

        self.root = root
        self.ready.set()
        root.after(16, self._poll)
        root.mainloop()

    def _poll(self):
        try:
            while True:
                self._apply(self.q.get_nowait())
        except queue.Empty:
            pass
        except Exception:
            pass
        try:
            self.root.after(16, self._poll)
        except Exception:
            pass

    def _apply(self, msg):
        _, word, cn, note, secs = msg
        root = self.root
        self.lbl_word.config(text=word or "")
        self.lbl_cn.config(text=cn or "")
        self.lbl_note.config(text=note or "")
        root.update_idletasks()
        w = max(root.winfo_reqwidth(), 200)
        h = root.winfo_reqheight()
        sw = root.winfo_screenwidth()
        root.geometry("%dx%d+%d+%d" % (w, h, sw - w - 28, 76))
        root.deiconify()
        root.lift()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        if self._hide_id is not None:
            try:
                root.after_cancel(self._hide_id)
            except Exception:
                pass
        self._hide_id = root.after(int(secs * 1000), root.withdraw)

    def show(self, word="", cn="", note="", secs=POPUP_SECONDS):
        self.q.put(("show", word, cn, note, secs))

    def hide(self):
        self.q.put(("show", "", "", "", 0.01))


POPUP = None

# ---------------- 主流程 ----------------
_busy = threading.Lock()
_gen = 0


def _show(gen, word, cn, note, secs=POPUP_SECONDS):
    if gen == _gen and POPUP is not None:
        POPUP.show(word, cn, note, secs)


def _lookup_worker(gen, word, deadline, t0):
    """后台联网查词；总耗时不超过 deadline。"""
    try:
        cn, src = online_lookup(word, deadline)
        if cn:
            CACHE[word] = cn
            save_cache()
            _show(gen, word, cn, "%s · 联网 · %.2fs" % (src, time.time() - t0))
            return

        hit = fuzzy_local(word)
        if hit:
            w, c, s = hit
            _show(gen, word, c, "词库近似 %s（%.0f%%）· %.2fs"
                  % (w, s * 100, time.time() - t0))
            return

        _show(gen, word, "没查到释义", "词库未收录，联网也超时/无结果", 4)
    except Exception as e:
        log("联网查词异常：%s" % e)
        _show(gen, word, "查词出错", str(e)[:60], 4)


def run_once():
    global _gen
    if not _busy.acquire(blocking=False):
        log("上一次识别还没完成，忽略")
        return
    _gen += 1
    gen = _gen
    t0 = time.time()
    try:
        img = grab()
        ocr_items = do_ocr(img)
        word, cn, in_vocab, tokens = pick_word(ocr_items)

        if DEBUG:
            try:
                img.save(os.path.join(BASE, "debug_last.png"))
                with open(os.path.join(BASE, "debug_ocr.txt"), "w", encoding="utf-8") as df:
                    for text, score, box in ocr_items:
                        df.write("%s\t%.3f\t%s\n" % (text, score, box))
            except Exception:
                pass

        if not word:
            log("没识别到英文：%s" % [t[0] for t in tokens][:6])
            _show(gen, "没识别到英文单词", "按 F7 重新框选识别区域", "OCR %d 条" % len(ocr_items), 3)
            return

        elapsed = time.time() - t0

        if in_vocab:
            log("词库命中：%s = %s  (%.2fs)" % (word, cn, elapsed))
            note = "词库 · %.2fs" % elapsed
            if elapsed > LOCAL_BUDGET:
                note += " · 区域偏大，按 F7 缩小更快"
            _show(gen, word, cn, note)
            return

        cached = CACHE.get(word)
        if cached:
            log("缓存命中：%s = %s  (%.2fs)" % (word, cached, elapsed))
            _show(gen, word, cached, "缓存 · %.2fs" % elapsed)
            return

        # 词库没有：先把词显示出来，再后台联网，保证 3s 内一定有结果
        log("词库未收录，联网查 %s  (已用 %.2fs)" % (word, elapsed))
        _show(gen, word, "查词中…", "词库未收录 · 联网查询", 8)

        deadline = t0 + ONLINE_BUDGET
        if deadline - time.time() < 0.3:
            deadline = time.time() + 0.3
        threading.Thread(target=_lookup_worker,
                         args=(gen, word, deadline, t0), daemon=True).start()
    except Exception as e:
        log("异常：%s" % e)
        _show(gen, "出错了", str(e)[:80], 4)
    finally:
        _busy.release()


# ---------------- 识别区域 ----------------
def load_crop():
    global CROP
    if os.path.exists(CROP_FILE):
        try:
            with open(CROP_FILE, encoding="utf-8") as f:
                c = json.load(f)
            if isinstance(c, list) and len(c) == 4:
                CROP = tuple(int(x) for x in c)
                log("已加载识别区域：%s" % (CROP,))
        except Exception as e:
            log("读取区域配置失败：%s" % e)


def select_region():
    import tkinter as tk
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.25)
    root.geometry("%dx%d+0+0" % (root.winfo_screenwidth(), root.winfo_screenheight()))
    cv = tk.Canvas(root, bg="black", highlightthickness=0)
    cv.pack(fill="both", expand=True)
    start = {}
    rect = [None]

    def on_press(e):
        start["x"], start["y"] = e.x, e.y
        cv.delete("all")
        rect[0] = cv.create_rectangle(e.x, e.y, e.x, e.y, outline="#00e676", width=2)

    def on_drag(e):
        if rect[0] is not None:
            cv.coords(rect[0], start["x"], start["y"], e.x, e.y)

    def on_release(e):
        x1, y1 = start.get("x", 0), start.get("y", 0)
        left, top = min(x1, e.x), min(y1, e.y)
        right, bottom = max(x1, e.x), max(y1, e.y)
        if right - left < 20 or bottom - top < 20:
            root.destroy()
            return
        global CROP
        CROP = (left, top, right, bottom)
        with open(CROP_FILE, "w", encoding="utf-8") as f:
            json.dump(list(CROP), f)
        log("已设置识别区域：%s" % (CROP,))
        root.destroy()

    cv.bind("<ButtonPress-1>", on_press)
    cv.bind("<B1-Motion>", on_drag)
    cv.bind("<ButtonRelease-1>", on_release)
    root.mainloop()


def _rotate_log():
    try:
        if os.path.getsize(LOG_FILE) > 2 * 1024 * 1024:
            os.replace(LOG_FILE, LOG_FILE + ".old")
    except Exception:
        pass


# ---------------- 入口 ----------------
def main():
    global POPUP
    _rotate_log()
    log("词库加载完成：%d 词（缓存 %d 条）" % (len(WORD2CN), len(CACHE)))
    load_crop()

    POPUP = Popup()          # 先把悬浮窗线程拉起来，避免首次显示卡顿
    get_engine()             # 预加载 + 预热 OCR

    log("热键：%s=识别  %s=框选  %s=退出"
        % (HOTKEY_TRIGGER.upper(), HOTKEY_REGION.upper(), HOTKEY_QUIT.upper()))
    keyboard.add_hotkey(HOTKEY_TRIGGER,
                        lambda: threading.Thread(target=run_once, daemon=True).start())
    keyboard.add_hotkey(HOTKEY_REGION,
                        lambda: threading.Thread(target=select_region, daemon=True).start())
    keyboard.add_hotkey(HOTKEY_QUIT, lambda: os._exit(0))

    if CROP:
        POPUP.show("助手就绪", "F9 识别", "F7 框选区域 · F8 退出", 2.5)
        log("就绪。识别区域 %s，按 %s 秒答" % (CROP, HOTKEY_TRIGGER.upper()))
    else:
        POPUP.show("还没框选识别区域", "先按 F7 框选题目区域",
                   "不框选会全屏识别，慢到没法用", 8)
        log("就绪，但没有识别区域：请按 %s 框选题目区域" % HOTKEY_REGION.upper())
    keyboard.wait()


def _selftest(path):
    """python vocab_solver.py --test 图片.png  仅测识别的耗时与结果。"""
    t0 = time.time()
    img = Image.open(path).convert("RGB")
    t_img = time.time() - t0
    t1 = time.time()
    items = do_ocr(img)
    t_ocr = time.time() - t1
    t2 = time.time()
    word, cn, in_vocab, tokens = pick_word(items)
    t_pick = time.time() - t2
    print("截图/读图 %.3fs  OCR %.3fs  选词 %.4fs  合计 %.3fs"
          % (t_img, t_ocr, t_pick, time.time() - t0))
    print("tokens:", [(w, round(s, 3), int(h)) for w, h, s in tokens])
    print("word=%r  in_vocab=%s  cn=%r" % (word, in_vocab, cn))
    if word and not in_vocab:
        t3 = time.time()
        r, src = online_lookup(word, time.time() + ONLINE_CALL_TIMEOUT)
        print("在线(%s) %.3fs -> %r" % (src, time.time() - t3, r))
    for text, score, box in items:
        print("  OCR: %r  %.3f" % (text, score))


if __name__ == "__main__":
    if "--test" in sys.argv and len(sys.argv) > sys.argv.index("--test") + 1:
        _selftest(sys.argv[sys.argv.index("--test") + 1])
    else:
        main()
