"""GUI 오프스크린 스모크 테스트.

화면이 없는 환경에서도 창 구성과 렌더 파이프라인 연결이 깨지지 않는지 검증한다.
실제 렌더는 백그라운드 QThread에서 돌므로 이벤트 루프를 돌려가며 완료를 기다린다.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# 헤드리스: 오프스크린 플랫폼 강제(디스플레이 불필요)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 패키지 임포트를 위해 sfx_generator의 부모 경로를 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6 import QtWidgets  # noqa: E402

from sfx_generator.gui.main_window import MainWindow  # noqa: E402

PASS, FAIL = "✅", "❌"
_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {msg}")


def wait_until(app, predicate, timeout=40.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.03)
    return False


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as td:
        w = MainWindow()
        w.out_dir = Path(td)
        check(w.windowTitle().startswith("SFX_Generator"), "메인 윈도우 생성")

        # 1) 프롬프트 생성 → 워커 렌더 완료 대기
        w.prompt_edit.setText("공포 영화 긴장감 라이저 끝에 묵직한 저음 쿵")
        w._on_generate_prompt()
        ok = wait_until(app, lambda: w.result_list.count() > 0)
        check(ok, "프롬프트 생성 → 파일 출력")
        check(w._samples is not None, "렌더 결과 파형/재생 버퍼 로드")
        check(w.recipe_edit.toPlainText().strip().startswith("{"), "레시피 JSON 편집창 채워짐")
        check("riser" in w.matched_label.text(), "감지 근거 라벨 표시")

        # 2) 변주 3개 생성
        w.var_spin.setValue(3)
        w.prompt_edit.setText("경쾌한 디지털 UI 알림음")
        w._on_generate_prompt()
        ok = wait_until(app, lambda: w.result_list.count() == 3)
        check(ok, "변주 3개 생성")

        # 3) JSON 직접 수정 → 재렌더(레시피 중심 워크플로 검증)
        w.var_spin.setValue(1)
        edited = (
            '{"name":"manual_edit","seed":1,'
            '"layers":[{"synth":{"kind":"whoosh","freq":300,"decay":1.0},'
            '"transforms":[{"op":"reverb","kind":"hall","wet":0.4}]}],'
            '"master":{"format":{"bit":24,"rate":48000},"prefix":"[SFX]"}}'
        )
        w.recipe_edit.setPlainText(edited)
        w._on_render_json()
        ok = wait_until(app, lambda: any("manual_edit" in w.result_list.item(i).text()
                                         for i in range(w.result_list.count())))
        check(ok, "JSON 직접 편집 → 재렌더")

        # 4) 잘못된 JSON은 크래시 없이 거부(메시지박스 자동 닫기 위해 타이머)
        from PySide6 import QtCore

        QtCore.QTimer.singleShot(200, lambda: [
            wgt.close() for wgt in app.topLevelWidgets() if isinstance(wgt, QtWidgets.QMessageBox)
        ])
        w.recipe_edit.setPlainText('{"name": "broken"')  # 깨진 JSON
        w._on_render_json()
        app.processEvents()
        check(True, "깨진 JSON 입력 시 크래시 없음")

        w.close()

    passed = sum(_results)
    print(f"\n{'='*50}\n결과: {passed}/{len(_results)} 통과")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
