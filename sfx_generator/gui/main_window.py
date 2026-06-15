"""SFX Forge 메인 윈도우.

설계 의도(Why):
- 라이브러리 폴더와 출력 폴더는 '실행 시점에' 사용자가 지정한다(하드코딩 금지).
- 프롬프트 → 레시피 JSON → 렌더의 흐름을 그대로 UI에 노출한다. 생성된 레시피 JSON을
  편집해 다시 렌더할 수 있어, 룰 엔진의 결과를 손으로 미세조정하는 길을 열어둔다.
- 오디오 재생은 sounddevice를 지연 임포트로 시도하고, 없으면 재생만 비활성화한다
  (백엔드가 없어도 앱은 죽지 않는다).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __app_name__, __credit__, __version__
from ..core.interpreter import Interpreter
from ..core.library import Library
from ..core.llm import DEFAULT_HOST, list_models, make_ollama_llm
from ..core.aigen import DEFAULT_AI_HOST, health as ai_health
from ..core.adjust import apply_knobs
from ..core.presets import concept_pack, detect_concept
from ..core.process import build_recipe, list_styles, make_source_resolver, variation_recipes
from ..core.recipe import Recipe
from ..core.stems import split_stem
from .browser import LibraryBrowser
from .waveform import WaveformWidget
from .worker import AiGenWorker, ProcessWorker, RenderWorker, ScanWorker

_AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{__app_name__} — 방송 효과음 생성기")
        self.resize(820, 760)
        self.setAcceptDrops(True)  # 원본 오디오 드래그&드롭 허용

        # 설정 영구 저장(앱 재실행 후에도 경로 유지)
        self._settings = QtCore.QSettings("JTBC", "SFX_Generator")

        # 상태
        self.library: Library | None = None
        saved_out = self._settings.value("out_dir", "", str)
        self.out_dir: Path = Path(saved_out) if saved_out else (Path.home() / "SFX_Generator_Output")
        self._samples: np.ndarray | None = None  # 재생/파형용 (N,2)
        self._sr: int = 48000
        self._worker: RenderWorker | None = None
        self._scan_worker: ScanWorker | None = None
        self._proc_worker: ProcessWorker | None = None
        self._source_path: Path | None = None  # 원본 가공용 입력 파일
        self._ai_worker: AiGenWorker | None = None
        self._ai_host: str = self._settings.value("ai_host", DEFAULT_AI_HOST, str)
        self._current_recipe: Recipe | None = None  # 조절 노브의 기준(원본) 레시피
        self._base_resolver = None                   # 콜라주 등 소스 매핑 resolver
        self._knob_timer = QtCore.QTimer(self)       # 슬라이더 드래그 디바운스
        self._knob_timer.setSingleShot(True)
        self._knob_timer.setInterval(300)
        self._knob_timer.timeout.connect(self._render_adjusted)

        self._build_menu()
        self._build_ui()
        self._refresh_out_label()
        # 저장된 라이브러리 경로가 있으면 자동 로드(백그라운드)
        saved_lib = self._settings.value("library_dir", "", str)
        if saved_lib and Path(saved_lib).exists():
            self._load_library(Path(saved_lib), force=False)
        self._autodetect_ollama()  # Ollama가 떠 있으면 자동으로 켠다(이해 정확도↑)
        self._autodetect_ai()      # AI 생성 서버가 떠 있으면 옵션 활성화

    def _build_menu(self) -> None:
        """메뉴바: macOS 앱 메뉴에 자동 편입되는 About + 종료."""
        menu = self.menuBar().addMenu("도움말")
        about = menu.addAction(f"{__app_name__} 정보")
        about.triggered.connect(self._show_about)

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            f"{__app_name__} 정보",
            f"<b>{__app_name__}</b> v{__version__}<br><br>"
            "보유 라이브러리 변형·결합 + 절차적 합성으로<br>"
            "<b>저작권 프리</b> 방송 효과음을 만드는 도구.<br><br>"
            "프롬프트 → 레시피 → 24bit/48kHz WAV<br>"
            "완전 오프라인 · 로컬 Ollama(선택) 지원<br><br>"
            f"<span style='color:#9ca3af;'>{__credit__}</span>",
        )

    # ── UI 구성 ───────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setSpacing(10)

        # 1) 라이브러리
        lib_box = QtWidgets.QGroupBox("① 라이브러리 (선택)")
        lib_v = QtWidgets.QVBoxLayout(lib_box)
        lib_l = QtWidgets.QHBoxLayout()
        self.lib_btn = QtWidgets.QPushButton("📂 폴더 선택")
        self.lib_btn.clicked.connect(self._on_select_library)
        self.refresh_btn = QtWidgets.QPushButton("🔄 새로고침")
        self.refresh_btn.setToolTip("폴더 변경분을 다시 색인(강제 재분석)")
        self.refresh_btn.clicked.connect(self._on_refresh_library)
        self.refresh_btn.setEnabled(False)
        self.browse_btn = QtWidgets.QPushButton("🔎 찾아보기")
        self.browse_btn.setToolTip("샘플 검색·미리듣기·여러 개 합치기")
        self.browse_btn.clicked.connect(self._on_browse)
        self.browse_btn.setEnabled(False)
        self.stem_btn = QtWidgets.QPushButton("✂️ 스템 분할")
        self.stem_btn.setToolTip("효과 스템(SFX만 있는 트랙)을 개별 효과음으로 잘라 라이브러리에 추가")
        self.stem_btn.clicked.connect(self._on_split_stem)
        self.stem_btn.setEnabled(False)
        self.lib_label = QtWidgets.QLabel("미지정 — 합성만으로 생성됩니다")
        self.lib_label.setStyleSheet("color:#9ca3af;")
        lib_l.addWidget(self.lib_btn)
        lib_l.addWidget(self.refresh_btn)
        lib_l.addWidget(self.browse_btn)
        lib_l.addWidget(self.stem_btn)
        lib_l.addWidget(self.lib_label, 1)
        lib_v.addLayout(lib_l)
        # 색인 진행률(평소 숨김)
        self.scan_bar = QtWidgets.QProgressBar()
        self.scan_bar.setVisible(False)
        self.scan_bar.setTextVisible(True)
        lib_v.addWidget(self.scan_bar)
        root.addWidget(lib_box)

        # 2) 프롬프트
        prm_box = QtWidgets.QGroupBox("② 프롬프트")
        prm_l = QtWidgets.QHBoxLayout(prm_box)
        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText("예: 공포 영화 긴장감 라이저 끝에 묵직한 저음 쿵")
        self.prompt_edit.returnPressed.connect(self._on_generate_prompt)
        self.var_spin = QtWidgets.QSpinBox()
        self.var_spin.setRange(1, 10)
        self.var_spin.setPrefix("변주 ")
        self.gen_btn = QtWidgets.QPushButton("🎛️ 생성")
        self.gen_btn.clicked.connect(self._on_generate_prompt)
        prm_l.addWidget(self.prompt_edit, 1)
        prm_l.addWidget(self.var_spin)
        prm_l.addWidget(self.gen_btn)
        root.addWidget(prm_box)

        # 2-b) Ollama(선택): 켜면 LLM 해석 우선, 실패 시 룰 폴백
        llm_l = QtWidgets.QHBoxLayout()
        self.ollama_chk = QtWidgets.QCheckBox("🧠 Ollama 사용")
        self.ollama_chk.toggled.connect(self._on_toggle_ollama)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setEnabled(False)
        self.model_combo.setMinimumWidth(220)
        llm_l.addWidget(self.ollama_chk)
        llm_l.addWidget(self.model_combo)
        llm_l.addStretch(1)
        # AI 생성(Stable Audio Open 등 로컬 서버) — 켜면 생성 버튼이 AI 생성으로 동작
        self.ai_chk = QtWidgets.QCheckBox("🤖 AI 생성")
        self.ai_chk.setToolTip("로컬 생성 서버(Stable Audio Open)로 만들고, 합성/변주로 마스터링")
        self.ai_chk.setEnabled(False)
        self.ai_var_chk = QtWidgets.QCheckBox("생성 후 변주까지")
        self.ai_var_chk.setChecked(True)
        self.ai_status = QtWidgets.QLabel("AI 서버 미감지")
        self.ai_status.setStyleSheet("color:#9ca3af;")
        self.ai_setup_btn = QtWidgets.QPushButton("⚙️ AI 서버 설치/시작")
        self.ai_setup_btn.setToolTip("ai_server 설치 스크립트를 실행(최초 1회 venv·의존성·모델 설치)")
        self.ai_setup_btn.clicked.connect(self._on_ai_setup)
        self.ai_detect_btn = QtWidgets.QPushButton("🔄")
        self.ai_detect_btn.setToolTip("AI 서버 다시 감지")
        self.ai_detect_btn.setMaximumWidth(36)
        self.ai_detect_btn.clicked.connect(self._autodetect_ai)
        llm_l.addWidget(self.ai_chk)
        llm_l.addWidget(self.ai_var_chk)
        llm_l.addWidget(self.ai_status)
        llm_l.addWidget(self.ai_setup_btn)
        llm_l.addWidget(self.ai_detect_btn)
        root.addLayout(llm_l)

        # 2-c) 생성 옵션: 길이(초) · 라우드니스 타깃 · 영감(랜덤)
        opt_l = QtWidgets.QHBoxLayout()
        self.length_chk = QtWidgets.QCheckBox("길이 지정")
        self.length_spin = QtWidgets.QDoubleSpinBox()
        self.length_spin.setRange(0.3, 30.0)
        self.length_spin.setValue(2.0)
        self.length_spin.setSingleStep(0.5)
        self.length_spin.setSuffix(" 초")
        self.length_spin.setEnabled(False)
        self.length_chk.toggled.connect(self.length_spin.setEnabled)
        opt_l.addWidget(self.length_chk)
        opt_l.addWidget(self.length_spin)
        opt_l.addSpacing(16)
        opt_l.addWidget(QtWidgets.QLabel("라우드니스:"))
        self.lufs_combo = QtWidgets.QComboBox()
        # (라벨, LUFS) — 방송 타깃
        for label, val in [("-16 일반", -16.0), ("-23 EBU R128", -23.0), ("-24 ATSC A/85", -24.0)]:
            self.lufs_combo.addItem(label, val)
        opt_l.addWidget(self.lufs_combo)
        opt_l.addStretch(1)
        root.addLayout(opt_l)

        # 2-b) 원본 가공: 입력 음원을 실제 소스로 받아 변형/보강
        src_box = QtWidgets.QGroupBox("②-b 원본 가공 (변주·어울리는 효과) — 파일을 끌어다 놔도 됩니다")
        src_l = QtWidgets.QHBoxLayout(src_box)
        self.src_btn = QtWidgets.QPushButton("📂 원본 선택")
        self.src_btn.clicked.connect(self._on_pick_source)
        self.src_label = QtWidgets.QLabel("원본 미지정")
        self.src_label.setStyleSheet("color:#9ca3af;")
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItem("자동 변주")
        for st in list_styles():
            self.style_combo.addItem(st)
        self.process_btn = QtWidgets.QPushButton("🎚 가공 생성")
        self.process_btn.clicked.connect(self._on_process)
        src_l.addWidget(self.src_btn)
        src_l.addWidget(self.src_label, 1)
        src_l.addWidget(QtWidgets.QLabel("스타일:"))
        src_l.addWidget(self.style_combo)
        src_l.addWidget(self.process_btn)
        root.addWidget(src_box)

        self.matched_label = QtWidgets.QLabel("")
        self.matched_label.setStyleSheet("color:#60a5fa;")
        root.addWidget(self.matched_label)

        # 3) 레시피 JSON(편집 → 재렌더)
        rec_box = QtWidgets.QGroupBox("③ 레시피 JSON (편집 가능)")
        rec_l = QtWidgets.QVBoxLayout(rec_box)
        self.recipe_edit = QtWidgets.QPlainTextEdit()
        self.recipe_edit.setPlaceholderText("프롬프트로 생성하면 여기에 레시피가 채워집니다. 직접 수정 후 재렌더도 가능합니다.")
        self.recipe_edit.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 12px;")
        self.render_json_btn = QtWidgets.QPushButton("↻ 이 JSON으로 렌더")
        self.render_json_btn.clicked.connect(self._on_render_json)
        rec_l.addWidget(self.recipe_edit, 1)
        rec_l.addWidget(self.render_json_btn, alignment=QtCore.Qt.AlignRight)
        root.addWidget(rec_box, 1)

        # 4) 파형 + 재생
        self.waveform = WaveformWidget()
        root.addWidget(self.waveform)
        play_l = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ 재생")
        self.play_btn.clicked.connect(self._on_play)
        self.stop_btn = QtWidgets.QPushButton("■ 정지")
        self.stop_btn.clicked.connect(self._on_stop)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        play_l.addWidget(self.play_btn)
        play_l.addWidget(self.stop_btn)
        play_l.addStretch(1)
        root.addLayout(play_l)

        # 4) 조절 노브: 생성된 소리를 실시간으로 다듬어 상상에 맞춘다
        knob_box = QtWidgets.QGroupBox("④ 조절 (생성 후 슬라이더로 다듬기)")
        kg = QtWidgets.QGridLayout(knob_box)
        # (라벨, 키, 최소, 최대, 기본, 표시배율, 단위)
        specs = [
            ("밝기", "brightness", -12, 12, 0, 1, "dB"),
            ("피치", "pitch", -12, 12, 0, 1, "반음"),
            ("어택", "attack", 0, 50, 0, 0.01, "s"),
            ("공간감", "space", 0, 90, 0, 0.01, ""),
            ("무게", "weight", 0, 12, 0, 1, "dB"),
            ("거칠기", "grit", 0, 15, 0, 1, "dB"),
        ]
        self.knobs: dict[str, QtWidgets.QSlider] = {}
        self._knob_labels: dict[str, QtWidgets.QLabel] = {}
        self._knob_specs = {s[1]: s for s in specs}
        for i, (label, key, lo, hi, dv, scale, unit) in enumerate(specs):
            r, c = divmod(i, 3)
            cell = QtWidgets.QVBoxLayout()
            head = QtWidgets.QLabel(f"{label}")
            val = QtWidgets.QLabel("0")
            val.setStyleSheet("color:#9ca3af;")
            self._knob_labels[key] = val
            sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld.setRange(lo, hi)
            sld.setValue(dv)
            sld.valueChanged.connect(lambda v, k=key, vlabel=val: self._on_knob_changed(k, v, vlabel))
            self.knobs[key] = sld
            top = QtWidgets.QHBoxLayout()
            top.addWidget(head)
            top.addStretch(1)
            top.addWidget(val)
            cell.addLayout(top)
            cell.addWidget(sld)
            kg.addLayout(cell, r, c)
        self.knob_reset_btn = QtWidgets.QPushButton("↩︎ 노브 초기화")
        self.knob_reset_btn.clicked.connect(self._reset_knobs)
        kg.addWidget(self.knob_reset_btn, 2, 2)
        root.addWidget(knob_box)

        # 5) 출력
        out_box = QtWidgets.QGroupBox("④ 출력")
        out_l = QtWidgets.QVBoxLayout(out_box)
        out_top = QtWidgets.QHBoxLayout()
        self.out_btn = QtWidgets.QPushButton("📁 저장 폴더")
        self.out_btn.clicked.connect(self._on_select_output)
        self.out_label = QtWidgets.QLabel("")
        out_top.addWidget(self.out_btn)
        out_top.addWidget(self.out_label, 1)
        out_l.addLayout(out_top)
        self.result_list = QtWidgets.QListWidget()
        self.result_list.itemClicked.connect(self._on_result_clicked)
        out_l.addWidget(self.result_list)
        root.addWidget(out_box, 1)

        # 하단 푸터: 제작자 정보
        footer = QtWidgets.QLabel(__credit__)
        footer.setAlignment(QtCore.Qt.AlignCenter)
        footer.setStyleSheet("color:#6b7280; font-size:11px; padding:6px 0 2px 0;")
        root.addWidget(footer)

        self.statusBar().showMessage("준비됨")

    # ── 라이브러리/출력 폴더 지정 ─────────────────────────────────────────────
    def _on_select_library(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "라이브러리 폴더 선택")
        if path:
            self._load_library(Path(path), force=False)

    # ── 라이브러리 브라우저 / 스템 분할 ───────────────────────────────────
    def _on_browse(self) -> None:
        if self.library is None:
            return
        dlg = LibraryBrowser(self.library, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        paths = dlg.selected_paths()
        if not paths:
            return
        if dlg.action == "source":
            self._set_source(paths[0])  # 원본 가공으로
            self.statusBar().showMessage(f"원본으로 설정: {paths[0].name} — '🎚 가공 생성' 가능")
        else:  # merge: 여러 샘플을 레이어로 합쳐 새 효과음
            self._render_collage(paths)

    def _render_collage(self, paths: list[Path]) -> None:
        """선택한 실제 샘플들을 레이어로 합쳐(콜라주) 렌더. 노브로 이어서 다듬을 수 있음."""
        mapping: dict[str, Path] = {}
        layers = []
        for i, p in enumerate(paths):
            q = f"__sel{i}__"
            mapping[q] = p
            layers.append({"source": {"query": q, "pick": "best"}})
        recipe = Recipe.model_validate({
            "name": "collage", "seed": 0, "layers": layers,
            "master": {"loudness_lufs": float(self.lufs_combo.currentData()),
                       "limiter": True, "format": {"bit": 24, "rate": 48000}, "prefix": "[SFX]"},
        })
        resolver = (lambda q, pick, _m=mapping: _m.get(q))
        self.matched_label.setText(f"✚ 샘플 합치기 — {len(paths)}개 레이어")
        self.recipe_edit.setPlainText(recipe.model_dump_json(indent=2, by_alias=True))
        self._set_base_recipe(recipe, resolver_override=resolver)  # 노브로 이어서 다듬기 가능
        self._start_render(recipe, resolver_override=resolver)

    def _on_split_stem(self) -> None:
        if self.library is None:
            return
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "효과 스템 선택 (SFX만 있는 트랙)", "", "오디오 (*.wav *.aif *.aiff *.flac)")
        if not fn:
            return
        try:
            dest = self.library.root / "_stems"
            clips = split_stem(Path(fn), dest)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "스템 분할 실패", str(e))
            return
        if not clips:
            QtWidgets.QMessageBox.information(self, "스템 분할", "잘라낼 효과음을 찾지 못했습니다(무음 임계값을 확인하세요).")
            return
        QtWidgets.QMessageBox.information(
            self, "스템 분할", f"{len(clips)}개 클립을 라이브러리에 추가했습니다.\n새로고침으로 색인합니다.")
        self._load_library(self.library.root, force=True)  # 재색인

    def _on_refresh_library(self) -> None:
        """현재 라이브러리를 강제로 다시 색인(폴더 변경분 반영)."""
        if self.library is not None:
            self._load_library(self.library.root, force=True)

    def _load_library(self, path: Path, force: bool) -> None:
        """라이브러리를 색인한다. 캐시가 있으면 즉시 로드, 없거나 강제면 백그라운드 풀 스캔."""
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.statusBar().showMessage("이미 색인 중입니다…")
            return
        try:
            lib = Library(path, cache_dir=Path(path) / ".sfx_cache")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "오류", f"라이브러리 열기 실패:\n{e}")
            return
        self._settings.setValue("library_dir", str(path))  # 경로 유지(재실행 후에도)

        # 강제가 아니면 캐시 즉시 로드 시도(시그니처 재계산·재분석 없이 → 빠른 시작)
        if not force:
            count = lib.load_cached()
            if count is not None:
                self.library = lib
                self.lib_label.setText(f"{count}개 음원 (캐시) — {path}  · 변경 시 🔄 새로고침")
                self.lib_label.setStyleSheet("color:#4ade80;")
                self.refresh_btn.setEnabled(True)
                self.browse_btn.setEnabled(True)
                self.stem_btn.setEnabled(True)
                self.statusBar().showMessage(f"라이브러리 준비됨 ({count}개, 캐시)")
                return

        # 캐시 없음/강제 → 백그라운드 풀 스캔(진행률 표시)
        self.lib_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.lib_label.setText(f"색인 중… — {path}")
        self.lib_label.setStyleSheet("color:#fbbf24;")
        self.scan_bar.setRange(0, 0)  # 우선 불확정 모드(파일 수집 중)
        self.scan_bar.setValue(0)
        self.scan_bar.setVisible(True)

        self._pending_lib = lib
        self._scan_worker = ScanWorker(lib, force=force)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished_ok.connect(lambda c, p=path: self._on_scan_done(c, p))
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _on_scan_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.scan_bar.setRange(0, total)
            self.scan_bar.setValue(done)
            self.scan_bar.setFormat(f"색인 중 {done}/{total}")

    def _on_scan_done(self, count: int, path: Path) -> None:
        self.library = self._pending_lib
        self.scan_bar.setVisible(False)
        self.lib_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.stem_btn.setEnabled(True)
        self.lib_label.setText(f"{count}개 음원 색인됨 — {path}")
        self.lib_label.setStyleSheet("color:#4ade80;")
        self.statusBar().showMessage(f"라이브러리 준비됨 ({count}개)")

    def _on_scan_failed(self, msg: str) -> None:
        self.scan_bar.setVisible(False)
        self.lib_btn.setEnabled(True)
        self.refresh_btn.setEnabled(self.library is not None)
        QtWidgets.QMessageBox.critical(self, "오류", f"라이브러리 색인 실패:\n{msg}")

    def _on_select_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if path:
            self.out_dir = Path(path)
            self._settings.setValue("out_dir", str(self.out_dir))  # 경로 유지
            self._refresh_out_label()

    def _refresh_out_label(self) -> None:
        self.out_label.setText(str(self.out_dir))

    # ── 생성/렌더 ─────────────────────────────────────────────────────────
    def _length_sec(self):
        """길이 지정 체크 시 초 값, 아니면 None(자연 길이)."""
        return self.length_spin.value() if self.length_chk.isChecked() else None

    def _apply_lufs(self, recipe: Recipe) -> Recipe:
        """선택한 라우드니스 타깃을 레시피 마스터에 반영."""
        recipe.master.loudness_lufs = float(self.lufs_combo.currentData())
        return recipe

    def _autodetect_ollama(self) -> None:
        """Ollama가 실행 중이면 자동으로 사용 체크(추상적 프롬프트 이해 정확도↑)."""
        try:
            models = list_models(DEFAULT_HOST, timeout=1.0)
        except Exception:  # noqa: BLE001 - 미실행/미설치면 조용히 무시
            models = []
        if models:
            self.ollama_chk.setChecked(True)  # toggled → 모델 콤보 자동 채움
            self.statusBar().showMessage(f"🧠 Ollama 감지됨 — 자동 사용 ({len(models)}개 모델)")

    def _autodetect_ai(self) -> None:
        """로컬 AI 생성 서버가 떠 있으면 활성화. 없으면 설치된 venv가 있을 때 자동 기동 시도."""
        info = ai_health(self._ai_host, timeout=1.0)
        if not info:
            self._maybe_autostart_ai()  # 이전에 설치해둔 서버를 조용히 켜본다
            info = ai_health(self._ai_host, timeout=1.5)
        if info:
            self.ai_chk.setEnabled(True)
            self.ai_status.setText(f"AI 서버 연결됨 ({info.get('mode', '?')})")
            self.ai_status.setStyleSheet("color:#4ade80;")
        else:
            self.ai_chk.setEnabled(False)
            self.ai_chk.setChecked(False)
            self.ai_status.setText("AI 서버 미감지 — ⚙️로 설치/시작")
            self.ai_status.setStyleSheet("color:#9ca3af;")

    def _find_ai_server(self) -> Path | None:
        """ai_server 폴더를 알 만한 위치들에서 찾는다(설정에 기억된 경로 우선)."""
        saved = self._settings.value("ai_server_dir", "", str)
        cands = [Path(saved)] if saved else []
        here = Path(__file__).resolve()
        cands += [
            here.parents[2] / "ai_server",   # 개발/리포 레이아웃
            here.parents[3] / "ai_server",
            Path.home() / "SFX_Generator" / "ai_server",
            Path.home() / "Downloads" / "SFX_Generator" / "ai_server",
        ]
        for c in cands:
            if c and (c / "AI_설치및실행.command").exists():
                return c
        return None

    def _maybe_autostart_ai(self) -> None:
        """설치된 venv가 있으면 백그라운드로 서버를 띄운다(두 번째 실행부터 자동)."""
        d = self._find_ai_server()
        if d is None:
            return
        venv_py = d / "venv" / "bin" / "python"
        if not venv_py.exists():
            return
        try:
            subprocess.Popen([str(venv_py), str(d / "sao_server.py"), "--device", "mps"],
                             cwd=str(d), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.statusBar().showMessage("AI 서버를 시작하는 중…")
        except Exception:  # noqa: BLE001
            pass

    def _on_ai_setup(self) -> None:
        """AI 서버 설치/시작 스크립트를 실행(최초 1회 venv·의존성·모델 설치)."""
        d = self._find_ai_server()
        if d is None:
            picked = QtWidgets.QFileDialog.getExistingDirectory(self, "ai_server 폴더 선택 (GitHub에서 받은 폴더 안)")
            if not picked:
                QtWidgets.QMessageBox.information(
                    self, "AI 서버",
                    "GitHub에서 받은 폴더의 'ai_server' 안에 있는\n'AI_설치및실행.command' 를 더블클릭해도 됩니다.")
                return
            d = Path(picked)
            if not (d / "AI_설치및실행.command").exists():
                QtWidgets.QMessageBox.warning(self, "AI 서버", "선택한 폴더에 설치 스크립트가 없습니다.")
                return
            self._settings.setValue("ai_server_dir", str(d))
        cmd = d / "AI_설치및실행.command"
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(cmd)])  # 터미널에서 진행 상황 표시
            else:
                subprocess.Popen(["bash", str(cmd)], cwd=str(d))
            QtWidgets.QMessageBox.information(
                self, "AI 서버",
                "설치/시작을 시작했습니다. 터미널 창에서 진행 상황을 확인하세요.\n"
                "(최초 1회는 의존성·모델 다운로드로 몇 분 걸립니다)\n\n"
                "완료되면 '🔄' 버튼으로 다시 감지하세요.")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "AI 서버", f"실행 실패:\n{e}\n\n직접 실행: {cmd}")

    def _on_toggle_ollama(self, checked: bool) -> None:
        """Ollama를 켜면 설치된 모델 목록을 불러와 콤보를 채운다. 실패 시 자동 해제."""
        if not checked:
            self.model_combo.setEnabled(False)
            return
        try:
            models = list_models(DEFAULT_HOST)
        except Exception as e:  # noqa: BLE001
            self.ollama_chk.setChecked(False)
            QtWidgets.QMessageBox.warning(
                self, "Ollama", f"Ollama에 연결할 수 없습니다.\n실행 중인지 확인하세요.\n\n{e}"
            )
            return
        if not models:
            self.ollama_chk.setChecked(False)
            QtWidgets.QMessageBox.warning(self, "Ollama", "설치된 모델이 없습니다.")
            return
        self.model_combo.clear()
        self.model_combo.addItems(models)
        # qwen 계열이 있으면 기본 선택
        for i, m in enumerate(models):
            if "qwen" in m.lower():
                self.model_combo.setCurrentIndex(i)
                break
        self.model_combo.setEnabled(True)

    def _on_generate_prompt(self) -> None:
        prompt = self.prompt_edit.text().strip()
        if not prompt:
            self.statusBar().showMessage("프롬프트를 입력하세요")
            return
        # AI 생성이 켜져 있으면 로컬 생성 서버로 만들고 합성/변주로 연결
        if self.ai_chk.isChecked() and self.ai_chk.isEnabled():
            self._generate_via_ai(prompt)
            return
        # 카테고리/모음 요청('예능 자막 효과음' 등)이면 대표 큐 '팩'을 생성
        concept = detect_concept(prompt)
        if concept is not None:
            self._generate_concept_pack(concept)
            return
        # Ollama 사용 시 llm 구성(실패하면 룰 폴백)
        llm = None
        if self.ollama_chk.isChecked() and self.model_combo.currentText():
            try:
                llm = make_ollama_llm(DEFAULT_HOST, self.model_combo.currentText())
            except Exception as e:  # noqa: BLE001
                self.statusBar().showMessage(f"Ollama 사용 불가 → 룰 사용: {e}")
                llm = None
        try:
            result = Interpreter(library=self.library, llm=llm, length_sec=self._length_sec()).interpret(prompt)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "오류", f"프롬프트 해석 실패:\n{e}")
            return
        self._apply_lufs(result.recipe)
        types = ", ".join(result.matched_types) or "없음"
        mods = ", ".join(result.matched_modifiers) or "없음"
        self.matched_label.setText(f"🔎 해석({result.source}) 소리: {types}   🔧 수식어: {mods}")
        if result.source == "rules" and result.matched_types == ["default"]:
            self.statusBar().showMessage("구체적 소리를 못 찾아 기본 톤으로 생성했습니다 — 🧠 Ollama를 켜면 추상적 표현도 더 정확합니다")
        # 생성된 레시피를 JSON 편집창에 채워 사용자가 확인/수정 가능하게
        self.recipe_edit.setPlainText(result.recipe.model_dump_json(indent=2, by_alias=True))
        self._set_base_recipe(result.recipe)  # 조절 노브 기준으로 설정(노브 0 초기화)
        self._start_render(result.recipe)

    def _generate_concept_pack(self, concept: str) -> None:
        """컨셉(예능/교양)에 어울리는 대표 자막 효과음 묶음을 한 번에 생성."""
        if self._proc_worker is not None and self._proc_worker.isRunning():
            self.statusBar().showMessage("이미 생성 중입니다…")
            return
        n = max(4, self.var_spin.value())
        pack = concept_pack(concept, n=n)
        recipes = []
        for label, rec in pack:
            self._apply_lufs(rec)
            rec.name = f"{concept}_{label}"
            recipes.append(rec)
        self.matched_label.setText(f"🎬 {concept} 자막 효과음 팩 — {len(recipes)}개 ({', '.join(l for l, _ in pack)})")
        resolver = self.library.as_resolver() if self.library is not None else None
        self._set_busy(True)
        self.statusBar().showMessage(f"{concept} 자막 효과음 팩 생성 중… ({len(recipes)}개)")
        self._proc_worker = ProcessWorker(recipes, self.out_dir, resolver, self._length_sec())
        self._proc_worker.finished_ok.connect(self._on_render_done)
        self._proc_worker.failed.connect(self._on_render_failed)
        self._proc_worker.start()

    def _generate_via_ai(self, prompt: str) -> None:
        """로컬 AI 서버로 생성 → 완료되면 합성/변주 파이프라인으로 연결."""
        if self._ai_worker is not None and self._ai_worker.isRunning():
            self.statusBar().showMessage("이미 AI 생성 중입니다…")
            return
        seconds = self._length_sec() or 8.0
        self.matched_label.setText(f"🤖 AI 생성 중… (Stable Audio) — {prompt}")
        self._set_busy(True)
        self.statusBar().showMessage("AI 서버에서 생성 중… (수십 초 걸릴 수 있음)")
        self._ai_worker = AiGenWorker(prompt, seconds, self._ai_host, 0, self.out_dir)
        self._ai_worker.finished_ok.connect(self._on_ai_done)
        self._ai_worker.failed.connect(self._on_render_failed)
        self._ai_worker.start()

    def _on_ai_done(self, base_path: str) -> None:
        """AI 생성 결과(원본)를 24/48·라우드니스·[SFX]로 마스터링하고, 옵션 시 변주까지."""
        base = Path(base_path)
        self._set_source(base)  # 원본 가공에서 이어서 다룰 수 있게 소스로 설정
        recipes = [build_recipe("원본 정리", name=f"{base.stem}_AI")]
        if self.ai_var_chk.isChecked():
            recipes += [r for _, r in variation_recipes(max(3, self.var_spin.value()), base_name=f"{base.stem}_AI")]
        for r in recipes:
            self._apply_lufs(r)
        resolver = make_source_resolver(base, self.library.as_resolver() if self.library is not None else None)
        self.matched_label.setText(f"🤖 AI 생성 완료 → 마스터링/변주 {len(recipes)}개")
        self._proc_worker = ProcessWorker(recipes, self.out_dir, resolver, self._length_sec())
        self._proc_worker.finished_ok.connect(self._on_render_done)
        self._proc_worker.failed.connect(self._on_render_failed)
        self._proc_worker.start()

    def _on_render_json(self) -> None:
        text = self.recipe_edit.toPlainText().strip()
        if not text:
            self.statusBar().showMessage("렌더할 레시피 JSON이 없습니다")
            return
        try:
            recipe = Recipe.from_json(text)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "레시피 오류", f"JSON 검증 실패:\n{e}")
            return
        self._set_base_recipe(recipe)
        self._start_render(recipe)

    def _start_render(self, recipe: Recipe, variations: int | None = None, resolver_override=None) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("이미 렌더 중입니다…")
            return
        resolver = resolver_override or (self.library.as_resolver() if self.library is not None else None)
        n = variations if variations is not None else self.var_spin.value()
        self._set_busy(True)
        self.statusBar().showMessage("렌더 중…")
        self._worker = RenderWorker(recipe, self.out_dir, resolver, n, self._length_sec())
        self._worker.finished_ok.connect(self._on_render_done)
        self._worker.failed.connect(self._on_render_failed)
        self._worker.start()

    def _on_render_done(self, paths: list) -> None:
        self._set_busy(False)
        self.result_list.clear()
        for p in paths:
            self.result_list.addItem(p)
        self.statusBar().showMessage(f"완료 — {len(paths)}개 파일 생성")
        if paths:
            self._load_for_preview(Path(paths[0]))

    def _on_render_failed(self, msg: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("렌더 실패")
        QtWidgets.QMessageBox.critical(self, "렌더 실패", msg)

    def _set_busy(self, busy: bool) -> None:
        for w in (self.gen_btn, self.render_json_btn, self.lib_btn, self.src_btn, self.process_btn):
            w.setEnabled(not busy)

    # ── 조절 노브 ─────────────────────────────────────────────────────────
    def _on_knob_changed(self, key: str, value: int, vlabel: QtWidgets.QLabel) -> None:
        scale = self._knob_specs[key][5]
        unit = self._knob_specs[key][6]
        shown = value * scale
        vlabel.setText(f"{shown:+.2f}{unit}" if scale != 1 else f"{value:+d}{unit}")
        if self._current_recipe is None:
            self.statusBar().showMessage("먼저 프롬프트로 소리를 생성하면 조절할 수 있어요")
            return
        self._knob_timer.start()  # 드래그가 멈춘 뒤 0.3초에 재렌더(디바운스)

    def _knob_value(self, key: str) -> float:
        return self.knobs[key].value() * self._knob_specs[key][5]

    def _render_adjusted(self) -> None:
        if self._current_recipe is None:
            return
        adj = apply_knobs(
            self._current_recipe,
            brightness=self._knob_value("brightness"),
            pitch=self._knob_value("pitch"),
            attack=self._knob_value("attack"),
            space=self._knob_value("space"),
            weight=self._knob_value("weight"),
            grit=self._knob_value("grit"),
        )
        self._apply_lufs(adj)
        self.recipe_edit.setPlainText(adj.model_dump_json(indent=2, by_alias=True))
        self.statusBar().showMessage("조절 적용 중…")
        self._start_render(adj, variations=1, resolver_override=self._base_resolver)  # 조절 미리듣기는 1개만

    def _reset_knobs(self) -> None:
        for key, sld in self.knobs.items():
            sld.blockSignals(True)
            sld.setValue(0)
            sld.blockSignals(False)
            self._knob_labels[key].setText("0")
        if self._current_recipe is not None:
            self._render_adjusted()
        self.statusBar().showMessage("노브 초기화")

    def _set_base_recipe(self, recipe: Recipe, resolver_override=None) -> None:
        """조절 노브의 기준 레시피를 설정하고 슬라이더를 0으로(무음) 초기화."""
        self._current_recipe = recipe
        self._base_resolver = resolver_override
        for key, sld in self.knobs.items():
            sld.blockSignals(True)
            sld.setValue(0)
            sld.blockSignals(False)
            self._knob_labels[key].setText("0")

    # ── 원본 가공(드래그&드롭/버튼) ───────────────────────────────────────
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        urls = e.mimeData().urls() if e.mimeData().hasUrls() else []
        if any(Path(u.toLocalFile()).suffix.lower() in _AUDIO_EXTS for u in urls):
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in _AUDIO_EXTS:
                self._set_source(p)
                break

    def _on_pick_source(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "원본 음원 선택", "", "오디오 (*.wav *.aif *.aiff *.flac *.ogg *.mp3)")
        if fn:
            self._set_source(Path(fn))

    def _set_source(self, path: Path) -> None:
        self._source_path = path
        self.src_label.setText(path.name)
        self.src_label.setStyleSheet("color:#4ade80;")
        self.statusBar().showMessage(f"원본 선택됨: {path.name} — '🎚 가공 생성'을 누르세요")

    def _on_process(self) -> None:
        if self._source_path is None:
            self._on_pick_source()
            if self._source_path is None:
                return
        if self._proc_worker is not None and self._proc_worker.isRunning():
            self.statusBar().showMessage("이미 가공 중입니다…")
            return
        stem = self._source_path.stem
        style = self.style_combo.currentText()
        n = self.var_spin.value()
        if style == "자동 변주":
            recipes = [r for _, r in variation_recipes(n, base_name=stem)]
        else:
            recipes = [build_recipe(style, name=f"{stem}_{style}_{i + 1}", seed=i) for i in range(n)]
        for r in recipes:
            self._apply_lufs(r)
        base = self.library.as_resolver() if self.library is not None else None
        resolver = make_source_resolver(self._source_path, base)
        self._set_busy(True)
        self.statusBar().showMessage(f"원본 가공 중… ({len(recipes)}개)")
        self._proc_worker = ProcessWorker(recipes, self.out_dir, resolver, self._length_sec())
        self._proc_worker.finished_ok.connect(self._on_render_done)
        self._proc_worker.failed.connect(self._on_render_failed)
        self._proc_worker.start()

    # ── 미리보기/재생 ─────────────────────────────────────────────────────
    def _on_result_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        self._load_for_preview(Path(item.text()))

    def _load_for_preview(self, path: Path) -> None:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"미리보기 로드 실패: {e}")
            return
        self._samples = data
        self._sr = sr
        self.waveform.set_samples(data)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def _on_play(self) -> None:
        if self._samples is None:
            return
        try:
            import sounddevice as sd  # 지연 임포트: 백엔드 없으면 여기서만 실패

            sd.stop()
            sd.play(self._samples, self._sr)
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"재생 불가(오디오 백엔드 없음): {e}")

    def _on_stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001
            pass
