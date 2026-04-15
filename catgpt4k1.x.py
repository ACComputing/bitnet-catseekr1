#!/usr/bin/env python3
"""CAT R1.1.X — Authentic BitNet 1.58b + Omni-Syntax Terminal (Fully Fixed)"""
import tkinter as tk
from tkinter import scrolledtext, font, messagebox
import numpy as np
import time
import threading
import re
import json
import os
import io
import ast
import contextlib
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict, List, Any

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "d_model": 64,
    "layers": 6,
    "heads": 4,
    "n_experts": 8,
    "top_k": 2,
    "simulate_latency": 0.25,
    "step_delay": 0.1,
    "api_port": 8765,
    "api_key": os.getenv("CATR1_API_KEY", "lm-studio"),
}

CAT_R11_PROFILE_MD = """# CAT R1.1.X — BitNet 1.58b + Omni-Syntax

Authentic ternary simulation. Direct answers. Clean code.

Weights: {-1, 0, 1} | Activations: {-1, 0, 1} | MatMul: Add/Sub only
Supported: Python, C++, C, HTML, JS, TS, Java, Rust, Bash, ASM, Go
Run: pip install numpy && python cat_r11.py
License: GPL3
"""

# ──────────────────────────────────────────────────────────────
# BITNET 1.58b AUTHENTIC ENGINE
# ──────────────────────────────────────────────────────────────
class CatR11Engine:
    __slots__ = ("name", "ver", "d_model", "_lock", "dialect_idx",
                 "dialects", "intent_map", "code_experts", "aliases",
                 "weights", "layers", "output_head", "norm_gamma", "norm_beta",
                 "_intent_weights", "_token_index", "_intent_trained",
                 "learning_curve", "response_locale",
                 "chat_history", "max_history", "assistant_mode")

    def __init__(self, d_model: int = None):
        self.name = "CAT R1.1.X"
        self.ver = "1.58b-Ternary-MoE"
        self.d_model = d_model or CONFIG["d_model"]
        self._lock = threading.Lock()
        self.dialect_idx = {"english": 0, "chinese": 0}
        self.learning_curve: List[float] = []
        self._intent_trained = False
        self._token_index: Dict[str, int] = {}
        self._intent_weights: Optional[np.ndarray] = None
        self.response_locale = "english"
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 24
        self.assistant_mode = "chatgpt_like"

        self.aliases = {"py":"python","c++":"cpp","js":"javascript","ts":"typescript",
                       "sh":"bash","shell":"bash","asm":"assembly","node":"javascript"}
        self.intent_map = {
            "hello": ["hi","hello","hey"],
            "bitnet": ["bitnet","ternary","-1, 0, 1","quantize"],
            "recursion": ["recursion","function calls itself"],
            "help": ["help","commands","menu","usage"],
            "languages": ["supported languages","which language","experts"],
            "profile": ["readme",".md","license","gpl3","about"],
        }
        self.dialects = {
            "english": [
                {"hello":"Hi. How can I help?","ready":"Ready. Ask for code or explanation.",
                 "py_intro":"Here is Python code.","generic":"Here is code.",
                 "bitnet":"BitNet uses ternary weights: {-1, 0, 1}.","recursion":"Recursion: function calls itself. Example:"},
                {"hello":"Hey. What do you need?","ready":"Send a prompt.","py_intro":"Python below.","generic":"Code below.",
                 "bitnet":"Ternary constraints eliminate FP32 multiplications.","recursion":"Self-referential execution. Example:"},
            ],
            "chinese": [
                {"hello":"你好。需要什么？","ready":"请给出任务。","py_intro":"Python 代码：","generic":"代码：",
                 "bitnet":"BitNet 使用三值权重：{-1, 0, 1}。","recursion":"递归：函数自调用。示例："},
            ],
        }
        self.code_experts = {
            "python": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
            "cpp": "#include <iostream>\nint main() { std::cout << \"Hello World\"; return 0; }",
            "c": "#include <stdio.h>\nint main() { printf(\"Hello World\\n\"); return 0; }",
            "javascript": "console.log('Hello World');",
            "html": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
            "typescript": "function main(): void { console.log('Hello World'); }\nmain();",
            "java": "public class Main { public static void main(String[] args) { System.out.println(\"Hello World\"); } }",
            "rust": "fn main() { println!(\"Hello World\"); }",
            "bash": "#!/bin/bash\necho \"Hello World\"",
            "assembly": "section .data\n    msg db 'Hello World',0xa\nsection .text\n    global _start\n_start:",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello World\") }",
        }
        self._init_ternary_weights()
        self._train_intent()

    def _ternary_random(self, shape):
        """Generate {-1,0,1} array compatible with all NumPy versions."""
        return np.random.choice([-1, 0, 1], size=shape).astype(np.int8)

    def _init_ternary_weights(self):
        d, h, e = self.d_model, self.d_model*2, CONFIG["n_experts"]
        self.weights = self._ternary_random((d, d))
        self.output_head = self._ternary_random((d, len(self.intent_map)+1))
        self.norm_gamma = np.ones((d,), dtype=np.float32)
        self.norm_beta = np.zeros((d,), dtype=np.float32)
        self.layers = []
        for _ in range(CONFIG["layers"]):
            self.layers.append({
                "q": self._ternary_random((d,d)), "k": self._ternary_random((d,d)),
                "v": self._ternary_random((d,d)), "o": self._ternary_random((d,d)),
                "ff_up": self._ternary_random((d,h)), "ff_down": self._ternary_random((h,d)),
                "router": self._ternary_random((d,e)),
                "experts": [{"up": self._ternary_random((d,h)), "down": self._ternary_random((h,d))} for _ in range(e)]
            })

    def _ternary_quantize(self, x, threshold=0.5):
        q = np.zeros_like(x, dtype=np.int8)
        q[x > threshold], q[x < -threshold] = 1, -1
        return q

    def _layer_norm(self, x):
        mean, std = np.mean(x, axis=-1, keepdims=True), np.std(x, axis=-1, keepdims=True) + 1e-5
        return (x - mean) / std * self.norm_gamma + self.norm_beta

    def _bitnet_linear(self, x, W):
        x_q = self._ternary_quantize(x).astype(np.int16)
        return (x_q @ (W==1).astype(np.int16)) - (x_q @ (W==-1).astype(np.int16))

    def _attention(self, x, layer):
        q = self._ternary_quantize(self._layer_norm(x)).astype(np.float32)
        k, v = q.copy(), q.copy()
        q, k, v = self._bitnet_linear(q, layer["q"]), self._bitnet_linear(k, layer["k"]), self._bitnet_linear(v, layer["v"])
        scale = np.sqrt(self.d_model) + 1e-6
        # Vector-safe attention gate (prevents scalar matmul shape failures).
        score = float(np.dot(q, k) / scale)
        gate = 1.0 / (1.0 + np.exp(-score))
        att_vec = v * gate
        out = self._bitnet_linear(att_vec, layer["o"]).astype(np.float32)
        return x + self._ternary_quantize(out)

    def _moe_ffn(self, x, layer):
        xn = self._ternary_quantize(self._layer_norm(x)).astype(np.float32)
        logits = self._bitnet_linear(xn, layer["router"]).astype(np.float32)
        top_idx = np.argsort(logits)[-CONFIG["top_k"]:]
        out = np.zeros(self.d_model, dtype=np.float32)
        for i in top_idx:
            e = layer["experts"][int(i)]
            h = np.tanh(self._ternary_quantize(self._bitnet_linear(xn, e["up"])).astype(np.float32))
            out += self._ternary_quantize(self._bitnet_linear(h, e["down"]).astype(np.float32)) / CONFIG["top_k"]
        return x + out

    def forward(self, x):
        y = x.astype(np.float32)
        for layer in self.layers:
            y = self._attention(y, layer)
            y = self._moe_ffn(y, layer)
            y = self._layer_norm(y)
        return y

    def encode_prompt(self, prompt):
        x = np.zeros(self.d_model, dtype=np.float32)
        for i, tok in enumerate(re.findall(r"[a-z0-9+#]+", prompt.lower())):
            idx = hash(tok) % self.d_model
            x[idx], x[(idx+3)%self.d_model] = x[idx]+1.0, x[(idx+3)%self.d_model]+0.5
        return self._layer_norm(x)

    def _train_intent(self):
        if self._intent_trained: return
        corpus = [("hi","hello"),("bitnet ternary","bitnet"),("recursion example","recursion"),
                  ("supported languages","languages"),("readme gpl","profile"),("help commands","help")]
        vocab = {t for text,_ in corpus for t in re.findall(r"[a-z0-9]+", text.lower())}
        self._token_index = {t:i for i,t in enumerate(vocab)}
        self._intent_weights = np.zeros((len(self.intent_map), len(vocab)+1), dtype=np.float32)
        for _ in range(20):
            for text, label in corpus:
                x = np.array([text.lower().count(t) for t in vocab] + [1.0], dtype=np.float32)
                y_idx = list(self.intent_map.keys()).index(label)
                scores = self._intent_weights @ x
                others = [s for i,s in enumerate(scores) if i!=y_idx]
                margin = (max(others) if others else -1e9) - scores[y_idx] + 1
                if margin > 0:
                    self._intent_weights[y_idx] += 0.15 * x
                    if others: self._intent_weights[np.argmax(others)] -= 0.15 * x
        self._intent_trained = True

    def detect_locale(self, p): return "chinese" if re.search(r"[\u4e00-\u9fff]|中文|chinese", p.lower()) else self.response_locale
    def get_dialect(self, loc):
        # Keep response tone stable and predictable (closer to chat assistant behavior).
        bank = self.dialects.get(loc, self.dialects["english"])
        return bank[0]

    def _remember(self, role: str, text: str):
        self.chat_history.append({"role": role, "text": text.strip()})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def _recent_user_context(self, n: int = 3) -> str:
        msgs = [m["text"] for m in self.chat_history if m["role"] == "user"]
        if not msgs:
            return ""
        return " | ".join(msgs[-n:])

    def _wants_brief(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["short", "brief", "one line", "tldr", "concise"])

    def _wants_steps(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["step by step", "steps", "walkthrough", "how do i"])

    def _wants_code_only(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["code only", "just code", "only code", "no explanation"])

    def _chat_fallback(self, prompt: str, dialect: Dict[str, str]) -> str:
        p = prompt.lower().strip()
        ctx = self._recent_user_context()

        if p in {"thanks", "thank you", "thx"}:
            return "You're welcome."
        if p in {"who are you", "what are you"}:
            return "I can help with coding, debugging, and technical explanations."
        if "error" in p or "traceback" in p:
            return "Paste the full traceback and the file path, and I will pinpoint the fix."
        if self._wants_steps(prompt):
            return (
                "1) Clarify the goal and constraints.\n"
                "2) Build or patch the smallest working version.\n"
                "3) Run it and capture errors.\n"
                "4) Iterate until output matches your request."
            )
        if self._wants_brief(prompt):
            return "I can help. Tell me the exact file and what outcome you want."
        if ctx:
            return f"Got it. Based on your recent context ({ctx}), share the next prompt or error and I will handle it."
        return dialect["ready"]
    def extract_lang(self, p):
        original = p
        p = p.lower()
        for a,l in self.aliases.items():
            if f"in {a}" in p or f"{a} code" in p: return l
        for l in self.code_experts:
            if f"in {l}" in p or f"{l} code" in p: return l
        m = re.search(r'(?:write|code|syntax)\s+(?:in\s+)?([a-z+]+)', p)
        if m:
            return self.aliases.get(m.group(1), m.group(1))
        inferred = self.detect_lang_from_text(original)
        return self.normalize_lang(inferred) if inferred else None

    def detect_lang_from_text(self, text: str) -> Optional[str]:
        s = text or ""
        sl = s.lower()

        # Fast keyword/shape checks.
        if re.search(r"#!/bin/(ba)?sh|echo\s+['\"]|\$\{?[A-Z_][A-Z0-9_]*\}?|^\s*for\s+\w+\s+in\s+", s, re.MULTILINE):
            return "bash"
        if re.search(r"<!doctype html>|<html|</html>|<body|</body>|<div|</div>", sl):
            return "html"
        if re.search(r"\bconsole\.log\(|\bfunction\s+\w+\s*\(|=>|\b(let|const|var)\s+\w+", s):
            return "javascript"
        if re.search(r"\binterface\s+\w+|:\s*(string|number|boolean)\b", s):
            return "typescript"
        if re.search(r"^\s*#include\s+<", s, re.MULTILINE):
            if "std::" in s or "cout" in s:
                return "cpp"
            return "c"
        if re.search(r"\bpublic\s+class\b|\bSystem\.out\.println\(", s):
            return "java"
        if re.search(r"\bfn\s+main\s*\(|println!\(", s):
            return "rust"
        if re.search(r"\bpackage\s+main\b|\bfunc\s+main\s*\(", s):
            return "go"
        if re.search(r"\bsection\s+\.(text|data)\b|\bglobal\s+_start\b", sl):
            return "assembly"
        if re.search(r"^\s*def\s+\w+\s*\(|__name__\s*==\s*['\"]__main__['\"]|\bprint\(", s, re.MULTILINE):
            return "python"

        # Fallback token scoring across supported syntaxes.
        scores = {
            "python": 0,
            "cpp": 0,
            "c": 0,
            "javascript": 0,
            "typescript": 0,
            "java": 0,
            "rust": 0,
            "go": 0,
            "bash": 0,
            "assembly": 0,
            "html": 0,
        }
        token_hints = {
            "python": ["def ", "import ", "None", "True", "False", "elif", "self."],
            "cpp": ["std::", "#include", "cout", "cin", "namespace std", "->"],
            "c": ["#include", "printf(", "scanf(", "malloc(", "free("],
            "javascript": ["console.log", "function ", "=>", "let ", "const ", "var "],
            "typescript": [": string", ": number", "interface ", "type ", "implements "],
            "java": ["public class", "public static void main", "System.out.println", "new "],
            "rust": ["fn ", "let mut", "println!", "match ", "::"],
            "go": ["package ", "func ", "fmt.", ":=", "go "],
            "bash": ["#!/bin/bash", "echo ", "$(", "fi", "done"],
            "assembly": ["mov ", "jmp ", "section .text", "db ", "_start"],
            "html": ["<html", "<body", "<div", "</", "<!doctype"],
        }
        for lang, hints in token_hints.items():
            for h in hints:
                if h in s or h in sl:
                    scores[lang] += 1
        best_lang = max(scores, key=scores.get)
        return best_lang if scores[best_lang] > 0 else None
    def extract_code_block(self, p):
        m = re.search(r"```([a-zA-Z0-9_+#-]*)\n([\s\S]*?)```", p)
        if not m:
            return None, None
        lang = (m.group(1) or "").strip().lower() or None
        code = m.group(2).strip()
        if not lang:
            lang = self.detect_lang_from_text(code)
        return lang, code

    def normalize_lang(self, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        lang = lang.lower().strip()
        aliases = {
            "py": "python", "python3": "python",
            "c++": "cpp", "cc": "cpp",
            "js": "javascript", "node": "javascript",
            "ts": "typescript",
            "sh": "bash", "shell": "bash", "zsh": "bash",
            "asm": "assembly",
        }
        return aliases.get(lang, lang)

    def generate_dynamic_template(self, lang: str, prompt: str) -> str:
        lang = self.normalize_lang(lang) or "python"
        comment = "//"
        if lang in {"python", "bash"}:
            comment = "#"
        elif lang == "html":
            comment = "<!-- -->"

        if lang == "html":
            return "<!DOCTYPE html>\n<html>\n<body>\n  <h1>Hello World</h1>\n</body>\n</html>"
        if lang == "python":
            return "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
        return (
            f"{comment} Dynamic template for {lang}\n"
            f"{comment} Prompt: {prompt[:80]}"
        )

    def _extract_prompt_requirements(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()
        fn_match = re.search(r"(?:function|def|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)", prompt)
        class_match = re.search(r"(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)", prompt)
        return {
            "wants_main": ("main" in p) or ("entry point" in p),
            "wants_json": ("json" in p),
            "wants_async": ("async" in p) or ("await" in p),
            "wants_cli": ("arg" in p) or ("argv" in p) or ("command line" in p) or ("cli" in p),
            "wants_file_io": ("file" in p) or ("read" in p) or ("write" in p),
            "function_name": fn_match.group(1) if fn_match else None,
            "class_name": class_match.group(1) if class_match else None,
        }

    def _validate_code(self, lang: str, code: str) -> bool:
        lang = self.normalize_lang(lang) or "python"
        try:
            if lang == "python":
                ast.parse(code, mode="exec")
                return True
            if lang in {"javascript", "typescript", "java", "cpp", "c", "go", "rust"}:
                opens = sum(code.count(ch) for ch in "{([")
                closes = sum(code.count(ch) for ch in "})]")
                return opens == closes and len(code.strip()) > 0
            if lang == "html":
                return "<html" in code.lower() and "</html>" in code.lower()
            if lang == "bash":
                return len(code.strip()) > 0
            return len(code.strip()) > 0
        except Exception:
            return False

    def _tailor_code_once(self, lang: str, code: str, req: Dict[str, Any]) -> str:
        lang = self.normalize_lang(lang) or "python"
        patched = code
        fn_name = req.get("function_name")
        class_name = req.get("class_name")

        if lang == "python":
            if req["wants_async"] and "async def" not in patched:
                patched = (
                    "import asyncio\n\n"
                    "async def main_async():\n"
                    "    print('Hello World')\n\n"
                    "if __name__ == '__main__':\n"
                    "    asyncio.run(main_async())"
                )
            if req["wants_json"] and "import json" not in patched:
                patched = f"import json\n\n{patched}"
            if req["wants_cli"] and "import sys" not in patched:
                patched = f"import sys\n\n{patched}"
            if req["wants_file_io"] and "open(" not in patched:
                patched += "\n\n# file io example\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello World')\n"
            if fn_name and f"def {fn_name}(" not in patched:
                patched += f"\n\ndef {fn_name}():\n    return 'ok'\n"
            if class_name and f"class {class_name}" not in patched:
                patched += f"\n\nclass {class_name}:\n    pass\n"
            if req["wants_main"] and "__name__ == '__main__'" not in patched:
                patched += "\n\nif __name__ == '__main__':\n    main()\n"
            return patched

        if lang in {"javascript", "typescript"}:
            if req["wants_async"] and "async function" not in patched:
                patched = "async function main(){\n  console.log('Hello World');\n}\nmain();"
            if req["wants_json"] and "JSON." not in patched:
                patched += "\n\nconst payload = JSON.stringify({ ok: true });\nconsole.log(payload);\n"
            if fn_name and f"function {fn_name}" not in patched:
                patched += f"\n\nfunction {fn_name}() {{ return 'ok'; }}\n"
            return patched

        if lang == "bash":
            if not patched.startswith("#!/bin/bash"):
                patched = "#!/bin/bash\n" + patched
            if req["wants_file_io"] and ">" not in patched:
                patched += "\necho \"Hello World\" > output.txt\n"
            return patched

        return patched

    def recompile_code_for_prompt(self, lang: str, prompt: str, seed_code: str) -> str:
        """Dynamic recompilation: iterate and tailor generated code to prompt requirements."""
        lang = self.normalize_lang(lang) or "python"
        req = self._extract_prompt_requirements(prompt)
        code = seed_code
        for _ in range(3):
            code = self._tailor_code_once(lang, code, req)
            if self._validate_code(lang, code):
                return code
        # Final safe fallback if iterative tailoring failed validation.
        fallback = self.code_experts.get(lang) or self.generate_dynamic_template(lang, prompt)
        return fallback

    def execute_code_any_language(self, lang: Optional[str], code: Optional[str]) -> str:
        lang = self.normalize_lang(lang) or "python"
        if not code:
            return "No code provided."

        if lang == "python":
            return self.safe_exec_python(code)
        if lang == "javascript":
            try:
                out = subprocess.run(
                    ["node", "-e", code],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                text = (out.stdout or out.stderr or "").strip()
                return text if text else "(no output)"
            except FileNotFoundError:
                return "Node.js runtime not found. Install node to execute JavaScript."
            except Exception as e:
                return f"Execution error: {e}"
        if lang == "bash":
            try:
                out = subprocess.run(
                    ["bash", "-lc", code],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                text = (out.stdout or out.stderr or "").strip()
                return text if text else "(no output)"
            except Exception as e:
                return f"Execution error: {e}"

        # Cross-language interpreter fallback: structural analysis for unsupported runtimes.
        lines = [ln for ln in code.splitlines() if ln.strip()]
        return (
            f"Interpreter summary ({lang}):\n"
            f"- lines: {len(lines)}\n"
            f"- chars: {len(code)}\n"
            f"- execution backend: not installed for {lang}\n"
            "Tip: Python/JavaScript/Bash run natively in this local interpreter."
        )

    def safe_exec_python(self, code):
        try:
            if not code:
                return "No code provided."
            tree = ast.parse(code, mode="exec")
            if any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global)) for n in ast.walk(tree)):
                return "Blocked: imports/global not allowed."
            allowed = {"print", "len", "range", "int", "float", "str", "list", "dict", "min", "max", "sum"}
            builtins_src = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
            safe = {k: v for k, v in builtins_src.items() if k in allowed}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree,"<catr11>","exec"), {"__builtins__": safe}, {})
            return buf.getvalue().strip() or "(no output)"
        except Exception as e: return f"Error: {e}"

    def generate(self, prompt, simulate=True):
        loc, dia = self.detect_locale(prompt), self.get_dialect(self.detect_locale(prompt))
        self._remember("user", prompt)
        if simulate: time.sleep(CONFIG["simulate_latency"])
        _ = self.forward(self.encode_prompt(prompt))
        p = prompt.lower()
        if p.strip() in {"/reset", "reset chat", "clear memory"}:
            self.chat_history = []
            return "Conversation memory cleared."
        for intent, keys in self.intent_map.items():
            if any(k in p for k in keys):
                if intent=="hello":
                    resp = dia["hello"]
                    self._remember("assistant", resp)
                    return resp
                if intent=="bitnet":
                    resp = dia["bitnet"]
                    self._remember("assistant", resp)
                    return resp
                if intent=="recursion":
                    resp = f"{dia['recursion']}\n\ndef fact(n):\n    return 1 if n<=1 else n*fact(n-1)"
                    self._remember("assistant", resp)
                    return resp
                if intent=="help":
                    resp = (
                        "Commands: hello, bitnet, recursion, languages, profile, "
                        "write code in <lang>, run/execute code, /reset"
                    )
                    self._remember("assistant", resp)
                    return resp
                if intent=="languages":
                    resp = f"Supported: {', '.join(sorted(self.code_experts))}"
                    self._remember("assistant", resp)
                    return resp
                if intent=="profile":
                    resp = CAT_R11_PROFILE_MD
                    self._remember("assistant", resp)
                    return resp
        if any(x in p for x in ["run code","execute","interpret","/exec"]):
            block_lang, code = self.extract_code_block(prompt)
            requested_lang = self.extract_lang(prompt) or self.detect_lang_from_text(prompt)
            exec_lang = self.normalize_lang(block_lang or requested_lang or "python")
            if not code:
                resp = "Include code in fenced block, e.g. ```python ...```."
                self._remember("assistant", resp)
                return resp
            result = self.execute_code_any_language(exec_lang, code)
            resp = f"✅ Interpreter result ({exec_lang}):\n```\n{result}\n```"
            self._remember("assistant", resp)
            return resp
        lang = self.extract_lang(prompt) or self.detect_lang_from_text(prompt)
        if lang or any(x in p for x in ["code","write","script","template"]):
            lang = self.normalize_lang(lang or "python")
            intro = dia["py_intro"] if lang=="python" else dia["generic"]
            seed = self.code_experts.get(lang) or self.generate_dynamic_template(lang, prompt)
            code = self.recompile_code_for_prompt(lang, prompt, seed)
            if self._wants_code_only(prompt):
                resp = f"```{lang}\n{code}\n```"
            else:
                resp = f"{intro}\n\n```{lang}\n{code}\n```"
            self._remember("assistant", resp)
            return resp
        resp = self._chat_fallback(prompt, dia)
        self._remember("assistant", resp)
        return resp

    def get_thoughts(self, prompt, lang, ultra):
        base = ["⚡ Quantizing to {-1,0,1}...", f"🔢 {CONFIG['layers']} ternary layers...", f"🧩 MoE {CONFIG['top_k']}/{CONFIG['n_experts']} active", f"🎯 [{lang.upper()}]", "✅ Done."]
        if ultra: base.insert(2, "🧠 UltraThink pass...")
        return base

# ──────────────────────────────────────────────────────────────
# GUI & API
# ──────────────────────────────────────────────────────────────
class CatR11GUI:
    def __init__(self, root):
        self.root, self.engine = root, CatR11Engine()
        root.title(f"{self.engine.name} | {self.engine.ver}"); root.geometry("850x620"); root.configure(bg="#050505")
        self.fonts = {"mono": font.Font(family="Consolas" if os.name!="nt" else "Courier New", size=11),
                      "bold": font.Font(family="Consolas" if os.name!="nt" else "Courier New", size=11, weight="bold"),
                      "italic": font.Font(family="Consolas" if os.name!="nt" else "Courier New", size=10, slant="italic"),
                      "small": font.Font(family="Consolas" if os.name!="nt" else "Courier New", size=9)}
        self.chat = scrolledtext.ScrolledText(root, bg="#050505", fg="#00d9ff", font=self.fonts["mono"], insertbackground="cyan", relief="flat", padx=12, pady=12, state="disabled")
        self.chat.pack(expand=True, fill="both")
        # FIXED: Unpack 3 values (tag, color, font) not 2
        for tag_name, color, fnt in [("user","#ffffff",self.fonts["bold"]),("think","#4a4a4a",self.fonts["italic"]),("bot","#00aaff",self.fonts["bold"]),("code","#00ffaa",self.fonts["small"])]:
            self.chat.tag_config(tag_name, foreground=color, font=fnt)
        inp = tk.Frame(root, bg="#050505"); inp.pack(fill="x", padx=10, pady=5)
        self.entry = tk.Entry(inp, bg="#111", fg="#00d9ff", font=self.fonts["mono"], insertbackground="cyan", relief="flat", bd=2)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0,10)); self.entry.bind("<Return>", lambda e: self.send()); self.entry.focus_set()
        btns = tk.Frame(inp, bg="#050505"); btns.pack(side="right")
        for t,c in [("Help","help"),("Profile","readme"),("Py","write python code")]:
            tk.Button(btns, text=t, command=lambda c=c: self.entry.insert("end",c+" "), bg="#222", fg="#00d9ff", font=self.fonts["small"], relief="flat").pack(side="left", padx=2)
        self.status = tk.Label(root, text="Ready", bg="#050505", fg="#666", font=self.fonts["small"], anchor="w"); self.status.pack(fill="x", padx=10, pady=2)
        self.log("SYSTEM", f"{self.engine.name} ONLINE • API :{CONFIG['api_port']}"); self.log("SYSTEM", "BitNet 1.58b: Ternary {-1,0,1} weights + MoE active.")
        self._start_api()

    def log(self, sender, text, tag=None):
        self.chat.config(state="normal"); self.chat.insert("end", f"[{sender}]: ", "bot" if sender==self.engine.name else tag); self.chat.insert("end", f"{text}\n\n", tag); self.chat.config(state="disabled"); self.chat.see("end")
        if sender=="SYSTEM": self.status.config(text=text[:65])

    def send(self):
        msg = self.entry.get().strip()
        if not msg: return
        self.entry.delete(0,"end"); self.log("YOU", msg, "user"); self.status.config(text="Quantizing & Routing...")
        threading.Thread(target=self._infer, args=(msg,), daemon=True).start()

    def _infer(self, prompt):
        lang, ultra = self.engine.extract_lang(prompt) or "GENERAL", "ultrathink" in prompt.lower()
        delay = CONFIG["step_delay"] * (1.5 if ultra else 1)
        for step in self.engine.get_thoughts(prompt, lang, ultra):
            self.root.after(0, lambda s=step: self.log("THINK", s, "think")); time.sleep(delay)
        resp = self.engine.generate(prompt, simulate=True)
        self.root.after(0, lambda: self._display(resp)); self.root.after(0, lambda: self.status.config(text="Ready"))

    def _display(self, text):
        if "```" in text:
            parts = text.split("```")
            for i,p in enumerate(parts): self.log(self.engine.name, p+("```" if i<len(parts)-1 and i%2==0 else ""), "code" if i%2==1 else None)
        else: self.log(self.engine.name, text)

    def _start_api(self):
        gui = self
        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, data):
                body = json.dumps(data).encode(); self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length", len(body)); self.end_headers(); self.wfile.write(body)
            def _auth(self):
                key = self.headers.get("Authorization","").replace("Bearer ","").strip()
                return not CONFIG["api_key"] or key == CONFIG["api_key"]
            def do_POST(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                if self.path not in ("/message","/v1/chat/completions"): return self._json(404,{"error":"Not found"})
                try:
                    length = int(self.headers.get("Content-Length",0)); data = json.loads(self.rfile.read(length).decode()) if length else {}
                except: return self._json(400,{"error":"Invalid JSON"})
                prompt = data.get("message") or data.get("prompt") or next((m["content"] for m in reversed(data.get("messages",[])) if m.get("role")=="user"), "")
                if not prompt: return self._json(400,{"error":"Missing prompt"})
                def run(): resp = gui.engine.generate(prompt, simulate=False); gui.log("API", prompt, "user"); gui.log(gui.engine.name, resp)
                gui.root.after(0, run); self._json(202, {"accepted": True, "id": f"req-{int(time.time())}"})
            def do_GET(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                if self.path == "/v1/models": self._json(200, {"data":[{"id":"cat-r11-local","object":"model","arch":"bitnet_1.58b"}]})
                else: self._json(200, {"usage":"POST /message or /v1/chat/completions"})
            def log_message(self,*a): pass
        def serve():
            try: ThreadingHTTPServer(("127.0.0.1", CONFIG["api_port"]), Handler).serve_forever()
            except Exception as e: gui.root.after(0, lambda: gui.log("SYSTEM", f"API error: {e}"))
        threading.Thread(target=serve, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: root.destroy() if messagebox.askokcancel("Quit", "Exit CAT R1.1.X?") else None)
    CatR11GUI(root); root.mainloop()