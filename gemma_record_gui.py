import threading
import tempfile
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkfont

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
MODEL_ID = "google/gemma-3n-e4b-it"
INPUT_DEVICE_INDEX = 7          # PortAudio index mapping to hw:1,6
CHANNELS            = 2
SAMPLE_RATE         = 48_000
DURATION_SEC        = 4         # seconds to record per click
SYSTEM_PROMPT       = (
    "You are a friendly assistant. Respond in a natural, conversational tone. "
    "Avoid numbered or bulleted lists; instead write short sentences or paragraphs."
)

# ------------------------------------------------------
# AUDIO UTILS
# ------------------------------------------------------

def record_wav(path: str):
    """Capture audio from the chosen device and save as 16‑bit WAV."""
    sd.default.device = (INPUT_DEVICE_INDEX, None)
    buf = sd.rec(int(DURATION_SEC * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=CHANNELS, dtype="int16")
    sd.wait()
    write(path, SAMPLE_RATE, buf)

# ------------------------------------------------------
# MODEL SINGLETON (load once, reuse)
# ------------------------------------------------------

_model = None
_processor = None
_model_lock = threading.Lock()

def get_model_and_processor():
    global _model, _processor
    with _model_lock:
        if _model is None or _processor is None:
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            _model = (
                Gemma3nForConditionalGeneration
                .from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
                .eval()
            )
        return _model, _processor

# ------------------------------------------------------
# SIMPLE SANITISER TO AVOID UNICODE GLYPHS MISSING IN SOME FONTS
# ------------------------------------------------------

_REPLACE_MAP = {
    "•": "-",
    "▪": "-",
    "●": "-",
    "◦": "-",
    "—": "-",
    "–": "-",
}

def sanitize(text: str) -> str:
    for bad, good in _REPLACE_MAP.items():
        text = text.replace(bad, good)
    return text

# ------------------------------------------------------
# GUI
# ------------------------------------------------------


class GemmaGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gemma‑3n Conversational Audio Demo")

        # 4‑K scaling: triple default font sizes
        for f_name in (
            "TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkFixedFont",
            "TkMenuFont", "TkCaptionFont", "TkSmallCaptionFont",
            "TkIconFont", "TkTooltipFont",
        ):
            try:
                fnt = tkfont.nametofont(f_name)
                fnt.configure(size=int(fnt.cget("size") * 3))
            except tk.TclError:
                pass

        # Widgets
        self.record_btn = ttk.Button(self, text="🎙️  Record", command=self.start_record)
        self.record_btn.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.output = scrolledtext.ScrolledText(
            self, width=60, height=20, wrap="word", font=("Helvetica", 12)
        )
        self.output.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Disable recording until model is loaded
        self.record_btn.config(state="disabled")
        self._append_output("Loading model … please wait.\n")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    # --------------------------------------------------
    # THREAD‑SAFE OUTPUT
    # --------------------------------------------------

    def _append_output(self, txt: str):
        """Safely append text from any thread."""
        def inner():
            self.output.insert(tk.END, txt)
            self.output.see(tk.END)
        self.output.after(0, inner)

    # --------------------------------------------------
    # MODEL LOADER THREAD
    # --------------------------------------------------

    def _load_model_thread(self):
        try:
            get_model_and_processor()
            self._append_output("Model loaded. You can click Record.\n")
        except Exception as e:
            self._append_output(f"Error loading model: {e}\n")
            messagebox.showerror("Model Load Error", str(e))
        finally:
            # Enable the record button whether load succeeded or failed
            self.record_btn.after(0, lambda: self.record_btn.config(state="normal"))

    # --------------------------------------------------
    # RECORD → RUN MODEL in background thread
    # --------------------------------------------------

    def start_record(self):
        self.record_btn.config(state="disabled")
        self.output.delete("1.0", tk.END)
        threading.Thread(target=self._record_and_generate, daemon=True).start()

    def _record_and_generate(self):
        try:
            self._append_output(f"Recording… speak now ({DURATION_SEC} s)\n")
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "input.wav")
                record_wav(wav_path)

                self._append_output("Processing with Gemma … this may take a moment.\n")
                model, processor = get_model_and_processor()

                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is my audio message:"},
                            {"type": "audio", "audio": wav_path},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device, dtype=model.dtype)

                with torch.inference_mode():
                    generation = model.generate(
                        **inputs, max_new_tokens=256, disable_compile=True
                    )

                input_len = inputs["input_ids"].shape[-1]
                response = processor.decode(
                    generation[0][input_len:], skip_special_tokens=True
                )
                response = sanitize(response)

                self._append_output("\n===== Gemma response =====\n" + response + "\n")

        except Exception as e:
            self._append_output(f"Error: {e}\n")
            messagebox.showerror("Error", str(e))
        finally:
            self.record_btn.after(0, lambda: self.record_btn.config(state="normal"))

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------


if __name__ == "__main__":
    app = GemmaGUI()
    # Extra scaling for high‑DPI—each logical pixel gets 3× physical pixels
    app.tk.call("tk", "scaling", 3.0)
    app.mainloop()
