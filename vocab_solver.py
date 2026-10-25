# -*- coding: utf-8 -*-
"""
词汇测试秒答助手  v3
====================
英文单词 -> 中文意思。屏幕上常驻一个小面板，用鼠标点它一下就识别，只给意思。

    左键点面板   识别一次：截屏 -> OCR -> 查词 -> 面板显示意思
    按住拖动     把面板挪到不挡题目的位置
    右键         重新框选题目区域
    关黑窗口      退出（也可以按 F8）
    F9 / F7 / F8 键盘的等价操作，可选

速度策略：
  1. 词库内的词：字典直接命中，目标 < 1s（实测通常 0.2~0.4s）。
  2. 词库外的词：立刻显示「查词中…」，后台联网查词，总耗时硬性 <= 3s；
     结果写入 vocab_solver_cache.json，同一个词第二次直接命中缓存。
  3. 面板跑在独立线程里常驻，点它不会抢走考试窗口的焦点。

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
ONLINE_BUDGET = 3.0         # 从点击/按键起，最晚出结果（含截图 + OCR）
ONLINE_CALL_TIMEOUT = 1.3   # 单个词典接口超时；两个接口加起来仍 <= 3s

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


# ---------------- 常驻悬浮窗（点一下识别） ----------------
class Panel(object):
    """一直显示的小面板：左键点一下识别，按住可拖动，右键出菜单。

    点它不会把焦点从考试窗口抢走（WS_EX_NOACTIVATE），所以点完可以继续答题。
    """

    IDLE_CN = "点我识别"
    IDLE_NOTE = "右键：重新框选区域"
    NO_CROP_CN = "点我先框选区域"
    NO_CROP_NOTE = "按住拖出题目所在的小块"

    def __init__(self):
        self.q = queue.Queue()
        self.ready = threading.Event()
        self.root = None
        self.rect = (0, 0, 0, 0)      # 屏幕坐标，用来避开识别区域
        self._press = None
        self._moved = False
        self._win_xy = (0, 0)
        threading.Thread(target=self._loop, daemon=True).start()
        self.ready.wait(6)

    # ---------- Tk 线程 ----------
    def _loop(self):
        try:
            import tkinter as tk
        except Exception as e:
            log("tkinter 不可用：%s" % e)
            self.ready.set()
            return

        prev_fg = self._foreground()      # 记下当前前台窗口，面板显示完还给它

        root = tk.Tk()
        root.withdraw()                   # 先不显示，避免创建时抢焦点
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="#0e1216")

        frame = tk.Frame(root, bg="#0e1216", highlightthickness=2,
                         highlightbackground="#00e676")
        frame.pack(fill="both", expand=True)

        self.lbl_word = tk.Label(frame, text="", font=("Microsoft YaHei", 11),
                                 fg="#8fa3b0", bg="#0e1216", anchor="w",
                                 padx=16, pady=0)
        self.lbl_cn = tk.Label(frame, text=self.IDLE_CN,
                               font=("Microsoft YaHei", 19, "bold"),
                               fg="#00e676", bg="#0e1216", anchor="w",
                               padx=16, pady=2, justify="left", wraplength=300)
        self.lbl_note = tk.Label(frame, text=self.IDLE_NOTE,
                                 font=("Microsoft YaHei", 9),
                                 fg="#5c6b76", bg="#0e1216", anchor="w",
                                 padx=16, pady=0)
        self.lbl_word.pack(fill="x", pady=(8, 0))
        self.lbl_cn.pack(fill="x")
        self.lbl_note.pack(fill="x", pady=(0, 8))

        # 只在顶层绑定：子控件的 bindtags 会冒泡到顶层，绑多次会触发多次
        root.bind("<ButtonPress-1>", self._on_press)
        root.bind("<B1-Motion>", self._on_drag)
        root.bind("<ButtonRelease-1>", self._on_release)
        root.bind("<Button-3>", self._on_right_click)

        self.root = root
        self._place()
        self._no_activate()               # 显示之前先设成「不激活」

        root.bind("<Configure>", lambda e: self._update_rect())
        root.deiconify()
        root.after(60, self._update_rect)
        root.after(80, lambda: self._restore_focus(prev_fg))

        self.ready.set()
        root.after(16, self._poll)
        root.mainloop()

    def _place(self):
        root = self.root
        root.update_idletasks()
        w = max(root.winfo_reqwidth(), 150)
        h = root.winfo_reqheight()
        sw = root.winfo_screenwidth()
        root.geometry("%dx%d+%d+%d" % (w, h, sw - w - 28, 76))

    @staticmethod
    def _foreground():
        try:
            import ctypes
            return ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return 0

    @staticmethod
    def _restore_focus(hwnd):
        try:
            import ctypes
            if hwnd:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _no_activate(self):
        """可点击但不抢焦点，点完考试窗口仍是活动窗口。"""
        try:
            import ctypes
            u = ctypes.windll.user32
            hwnd = u.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            style = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
            u.SetWindowLongW(hwnd, GWL_EXSTYLE,
                             style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        except Exception as e:
            log("设置不抢焦点失败（不影响点按）：%s" % e)

    def _update_rect(self):
        try:
            r = self.root
            x, y = r.winfo_rootx(), r.winfo_rooty()
            self.rect = (x, y, x + r.winfo_width(), y + r.winfo_height())
        except Exception:
            pass

    # ---------- 鼠标 ----------
    def _on_press(self, e):
        self._press = (e.x_root, e.y_root)
        self._win_xy = (self.root.winfo_x(), self.root.winfo_y())
        self._moved = False

    def _on_drag(self, e):
        if not self._press:
            return
        dx = e.x_root - self._press[0]
        dy = e.y_root - self._press[1]
        if abs(dx) > 4 or abs(dy) > 4:
            self._moved = True
        if self._moved:
            self.root.geometry("+%d+%d" % (self._win_xy[0] + dx, self._win_xy[1] + dy))

    def _on_release(self, e):
        moved = self._moved
        self._press = None
        self._moved = False
        self._update_rect()
        if not moved:
            self._click()

    def _on_right_click(self, e):
        """右键 = 重新框选识别区域。"""
        self.show("", "重新框选区域", self.NO_CROP_NOTE)
        threading.Thread(target=select_region, daemon=True).start()

    def _click(self):
        global CROP
        if not CROP:
            self.show("", "先框选区域", self.NO_CROP_NOTE)
            threading.Thread(target=select_region, daemon=True).start()
            return
        threading.Thread(target=run_once, daemon=True).start()

    # ---------- 消息队列 ----------
    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "hide":
                    self.root.withdraw()
                    if item[1] is not None:
                        item[1].set()
                elif item[0] == "restore":
                    self.root.deiconify()
                    self.root.lift()
                    try:
                        self.root.attributes("-topmost", True)
                    except Exception:
                        pass
                else:
                    self._set(*item[1:])
        except queue.Empty:
            pass
        except Exception:
            pass
        try:
            self.root.after(16, self._poll)
        except Exception:
            pass

    def _set(self, word, cn, note):
        root = self.root
        self.lbl_word.config(text=word or "")
        self.lbl_cn.config(text=cn or "")
        self.lbl_note.config(text=note or "")
        root.update_idletasks()
        w = max(root.winfo_reqwidth(), 150)
        h = root.winfo_reqheight()
        sw = root.winfo_screenwidth()
        x, y = root.winfo_x(), root.winfo_y()
        if x + w > sw - 8:
            x = max(8, sw - w - 28)
        root.geometry("%dx%d+%d+%d" % (w, h, x, y))
        self._update_rect()

    # ---------- 对外 ----------
    def show(self, word="", cn="", note=""):
        self.q.put(("set", word, cn, note))

    def idle(self):
        self.q.put(("set", "", self.IDLE_CN, self.IDLE_NOTE))

    def hide_for_shot(self, timeout=0.6):
        """截图前临时藏起来（仅当它压住识别区域时），返回是否藏了。"""
        ev = threading.Event()
        self.q.put(("hide", ev))
        return ev.wait(timeout)

    def restore(self):
        self.q.put(("restore",))


PANEL = None

# ---------------- 主流程 ----------------
_busy = threading.Lock()
_gen = 0


def _show(gen, word, cn, note):
    if gen == _gen and PANEL is not None:
        PANEL.show(word, cn, note)


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

        _show(gen, word, "没查到释义", "词库未收录，联网也超时/无结果")
    except Exception as e:
        log("联网查词异常：%s" % e)
        _show(gen, word, "查词出错", str(e)[:60])


def _rects_overlap(a, b):
    """两个 (left, top, right, bottom) 是否相交。"""
    if not a or not b or a[2] <= a[0] or b[2] <= b[0]:
        return False
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def run_once():
    global _gen
    if not _busy.acquire(blocking=False):
        log("上一次识别还没完成，忽略")
        return
    _gen += 1
    gen = _gen
    t0 = time.time()
    try:
        _show(gen, "", "识别中…", "")

        # 面板自己压住识别区域时先藏起来，免得把自己的字识别进去
        hidden = False
        if PANEL is not None and CROP and _rects_overlap(PANEL.rect, CROP):
            hidden = PANEL.hide_for_shot()
        try:
            img = grab()
        finally:
            if hidden:
                PANEL.restore()

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
            _show(gen, "没识别到英文单词", "右键 → 重新框选区域",
                  "OCR %d 条" % len(ocr_items))
            return

        elapsed = time.time() - t0

        if in_vocab:
            log("词库命中：%s = %s  (%.2fs)" % (word, cn, elapsed))
            note = "词库 · %.2fs" % elapsed
            if elapsed > LOCAL_BUDGET:
                note += " · 区域偏大，重框小一点更快"
            _show(gen, word, cn, note)
            return

        cached = CACHE.get(word)
        if cached:
            log("缓存命中：%s = %s  (%.2fs)" % (word, cached, elapsed))
            _show(gen, word, cached, "缓存 · %.2fs" % elapsed)
            return

        # 词库没有：先把词显示出来，再后台联网，保证 3s 内一定有结果
        log("词库未收录，联网查 %s  (已用 %.2fs)" % (word, elapsed))
        _show(gen, word, "查词中…", "词库未收录 · 联网查询")

        deadline = t0 + ONLINE_BUDGET
        if deadline - time.time() < 0.3:
            deadline = time.time() + 0.3
        threading.Thread(target=_lookup_worker,
                         args=(gen, word, deadline, t0), daemon=True).start()
    except Exception as e:
        log("异常：%s" % e)
        _show(gen, "出错了", str(e)[:80], "")
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
            if PANEL is not None:
                PANEL.idle()
            root.destroy()
            return
        global CROP
        CROP = (left, top, right, bottom)
        with open(CROP_FILE, "w", encoding="utf-8") as f:
            json.dump(list(CROP), f)
        log("已设置识别区域：%s" % (CROP,))
        if PANEL is not None:
            PANEL.idle()
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
    global PANEL
    _rotate_log()
    log("词库加载完成：%d 词（缓存 %d 条）" % (len(WORD2CN), len(CACHE)))
    load_crop()

    PANEL = Panel()          # 常驻悬浮面板，程序活着它就在
    get_engine()             # 预加载 + 预热 OCR

    log("热键（可选）：%s=识别  %s=框选  %s=退出"
        % (HOTKEY_TRIGGER.upper(), HOTKEY_REGION.upper(), HOTKEY_QUIT.upper()))
    keyboard.add_hotkey(HOTKEY_TRIGGER,
                        lambda: threading.Thread(target=run_once, daemon=True).start())
    keyboard.add_hotkey(HOTKEY_REGION,
                        lambda: threading.Thread(target=select_region, daemon=True).start())
    keyboard.add_hotkey(HOTKEY_QUIT, lambda: os._exit(0))

    if CROP:
        PANEL.idle()
        log("就绪。识别区域 %s，点悬浮窗识别" % (CROP,))
    else:
        PANEL.show("", Panel.NO_CROP_CN, Panel.NO_CROP_NOTE)
        log("就绪，但没有识别区域：点悬浮窗即可框选")
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
