"""长期记忆：子 LLM 提炼的记忆写入文本文件持久化，召回用 BM25（纯手搓，无第三方依赖）。

文件格式：data/memories/{时间戳}_{session}.txt
    第一行是元数据（session/时间），其后是记忆正文；整文件作为一个文档参与 BM25 索引。
"""
import math
import re
import time
from pathlib import Path

_CJK_RE = re.compile(r"[一-鿿]+")
_LATIN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文同时产出单字与 2-gram（『北京天气』→ 北/京/天/气/北京/京天/天气），英文/数字按词切分。

    只用 2-gram 会有词表不匹配问题：查询『猫叫什么名字』与记忆『名叫雪球的…毛猫』无共同 2-gram。
    补上单字后靠 IDF 自然加权——常见字（的/了）IDF 低几乎不影响排序，稀有字（猫/雪球）IDF 高直接决定命中。
    """
    tokens: list[str] = []
    for seg in _CJK_RE.findall(text):
        tokens.extend(seg)                                   # 单字
        if len(seg) >= 2:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))  # 2-gram
    for seg in _LATIN_RE.findall(text.lower()):
        tokens.append(seg)
    return tokens


class BM25:
    """Okapi BM25（k1=1.5, b=0.75），语料小，实现从简。"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        self._len = [len(d) for d in docs]
        self._avg = (sum(self._len) / len(docs)) if docs else 0.0
        for d in docs:
            tf: dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        self._n = len(docs)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], idx: int) -> float:
        s, tf, dl = 0.0, self._tf[idx], self._len[idx]
        if not tf:
            return 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            freq = tf[t] * (self.k1 + 1)
            norm = tf[t] + self.k1 * (1 - self.b + self.b * dl / self._avg) if self._avg else tf[t]
            s += self._idf(t) * freq / norm
        return s

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """返回 (文档下标, 得分) 列表，按得分降序，只保留得分 > 0 的。"""
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self._n)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class MemoryStore:
    """记忆的写入与 BM25 召回。每次召回时惰性重载文件（语料小，直接重扫即可）。"""

    def __init__(self, root: Path, max_chars: int = 400) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars

    def add(self, session_id: str, text: str) -> str:
        """写入一条记忆文件，返回文件相对地址。超长硬截断兜底。"""
        text = text.strip()
        if not text:
            raise ValueError("记忆内容为空")
        if len(text) > self.max_chars:
            text = text[:self.max_chars - 1] + "…"
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.root / f"{ts}_{session_id}.txt"
        # 同秒多条时避免覆盖
        seq = 1
        while path.exists():
            seq += 1
            path = self.root / f"{ts}_{session_id}_{seq}.txt"
        body = f"session={session_id} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n"
        path.write_text(body, encoding="utf-8")
        # 相对项目根的地址（root=data/memories，root.parent.parent=项目根）
        return path.relative_to(self.root.parent.parent).as_posix()

    def _load_all(self) -> list[dict]:
        memories = []
        for path in sorted(self.root.glob("*.txt")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            if not lines:
                continue
            meta, text = lines[0], "\n".join(lines[1:]).strip()
            sid = meta.split("session=", 1)[-1].split(" ", 1)[0] if "session=" in meta else "?"
            when = meta.split("time=", 1)[-1].strip() if "time=" in meta else "?"
            memories.append({"session": sid, "time": when, "text": text,
                             "path": path.relative_to(self.root.parent.parent).as_posix()})
        return memories

    def search(self, query: str, k: int) -> list[dict]:
        """BM25 召回最相关的 k 条记忆（带得分，得分 > 0 才返回）。"""
        memories = self._load_all()
        if not memories:
            return []
        bm25 = BM25([tokenize(m["text"] + " " + m["session"]) for m in memories])
        hits = bm25.top(query, k)
        return [{**memories[i], "score": round(s, 3)} for i, s in hits]
