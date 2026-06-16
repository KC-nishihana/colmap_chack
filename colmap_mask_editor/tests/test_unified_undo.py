"""V0.11: 統一 Undo/Redo (UnifiedEditHistory) のテスト。

AI候補適用も手動編集と同じ履歴へ入り、マスクと AI 判断状態が同じ歩調で戻る。
"""

import numpy as np

from core.mask_ops import MaskEditor
from core.unified_edit_command import UnifiedEditCommand, UnifiedEditHistory


def _full(h=10, w=10):
    return np.full((h, w), 255, np.uint8)


def test_command_dataclass_fields():
    c = UnifiedEditCommand(source="ai_automatic", operation="remove",
                           affected_segment_ids=[12], before_decisions={},
                           after_decisions={"12": "remove"})
    assert c.source == "ai_automatic"
    assert c.affected_segment_ids == [12]


def test_spec_sequence_undo_redo_keeps_mask_and_decisions_in_sync():
    editor = MaskEditor(_full())
    h = UnifiedEditHistory(editor, {})

    snap0 = editor.mask.copy()

    # 1) AI候補 REMOVE (一括差し替え)
    m1 = _full(); m1[0:3, :] = 0
    editor.replace(m1)
    h.record(source="ai_automatic", operation="remove",
             after_decisions={"12": "remove"}, affected_segment_ids=[12])
    snap1 = editor.mask.copy()

    # 2) ブラシ ADD (対話編集: 判断は変えない)
    editor.begin_stroke()
    editor.paint(5, 5, 2, add=True)
    h.record(source="brush", operation="add")   # decisions 不変
    snap2 = editor.mask.copy()

    # 3) ポリゴン REMOVE (一括差し替え)
    m3 = editor.mask.copy(); m3[8:10, :] = 0
    editor.replace(m3)
    h.record(source="polygon", operation="remove",
             after_decisions={"12": "remove", "35": "remove"},
             affected_segment_ids=[35])
    snap3 = editor.mask.copy()

    assert h.current_decisions == {"12": "remove", "35": "remove"}

    # ----- Undo×3 (ポリゴン -> ブラシ -> AI の順で戻る) -----
    h.undo()
    assert np.array_equal(editor.mask, snap2)
    assert h.current_decisions == {"12": "remove"}     # ポリゴン前へ
    h.undo()
    assert np.array_equal(editor.mask, snap1)
    assert h.current_decisions == {"12": "remove"}     # ブラシ前 (判断不変)
    h.undo()
    assert np.array_equal(editor.mask, snap0)
    assert h.current_decisions == {}                   # AI前 (空)

    # ----- Redo×3 -----
    h.redo()
    assert np.array_equal(editor.mask, snap1)
    assert h.current_decisions == {"12": "remove"}
    h.redo()
    assert np.array_equal(editor.mask, snap2)
    assert h.current_decisions == {"12": "remove"}
    h.redo()
    assert np.array_equal(editor.mask, snap3)
    assert h.current_decisions == {"12": "remove", "35": "remove"}


def test_brush_add_undo_only_reverts_brush():
    # 仕様例: AI候補REMOVE → ブラシADD → Ctrl+Z でブラシだけ戻る
    editor = MaskEditor(_full())
    h = UnifiedEditHistory(editor, {})
    m1 = _full(); m1[0:3, :] = 0
    editor.replace(m1)
    h.record(source="ai_automatic", operation="remove", after_decisions={"5": "remove"})
    after_ai = editor.mask.copy()

    editor.begin_stroke(); editor.paint(5, 5, 2, add=True)
    h.record(source="brush", operation="add")

    cmd = h.undo()
    assert cmd.source == "brush"
    assert np.array_equal(editor.mask, after_ai)       # ブラシADDだけ戻る
    assert h.current_decisions == {"5": "remove"}      # AI判断は残る


def test_new_edit_clears_redo():
    editor = MaskEditor(_full())
    h = UnifiedEditHistory(editor, {})
    editor.replace(_full() * 0)
    h.record(source="ai_automatic", operation="remove", after_decisions={"1": "remove"})
    h.undo()
    assert h.can_redo()
    editor.begin_stroke(); editor.paint(1, 1, 1, add=False)
    h.record(source="brush", operation="remove")
    assert not h.can_redo()                            # 新編集で redo は消える


def test_can_undo_redo_and_empty_returns_none():
    editor = MaskEditor(_full())
    h = UnifiedEditHistory(editor, {})
    assert not h.can_undo()
    assert h.undo() is None
    assert h.redo() is None


def test_reset_clears_history():
    editor = MaskEditor(_full())
    h = UnifiedEditHistory(editor, {"1": "remove"})
    editor.replace(_full() * 0)
    h.record(source="brush", operation="remove")
    h.reset({"9": "add"})
    assert not h.can_undo()
    assert h.current_decisions == {"9": "add"}


def test_history_cap_matches_editor_max():
    editor = MaskEditor(_full())
    h = UnifiedEditHistory(editor, {})
    n = editor.MAX_HISTORY + 10
    for i in range(n):
        editor.begin_stroke(); editor.paint(1, 1, 1, add=(i % 2 == 0))
        h.record(source="brush", operation="add")
    # コマンド数は MaskEditor の上限と揃う (件数同期で undo がずれない)
    assert h.undo_depth == editor.MAX_HISTORY
    # 上限まで両方 undo できる
    count = 0
    while h.can_undo():
        h.undo(); count += 1
    assert count == editor.MAX_HISTORY
