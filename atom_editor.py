import ipywidgets as widgets
from IPython.display import display
import numpy as np
from ase.visualize.ngl import view_ngl


class AtomEditorUI:
    """
    ASE + nglview ベースのインタラクティブ原子エディタ。

    機能:
    - Jupyter 上で 3D ビューとスライダーを使って原子位置を編集
    - 原子クリックで選択（単一 / 複数切り替え）
    - tag によるフラグメント一括選択
    - 表示形式 (ball+stick / spacefill / licorice) の切り替え
    - ボールサイズのリアルタイム変更
    - 複数選択時はフラグメントを剛体平行移動（相対配置を維持）
    """

    def __init__(self, atoms):
        self.atoms = atoms

        # ASE の NGL ビュワー (nglview + control パネルの HBox)
        self.ngl_gui = view_ngl(atoms)
        self.view = self.ngl_gui.view   # nglview.NGLWidget

        # ----------------------
        # 状態変数
        # ----------------------
        self.selected_atoms = set()
        self.is_syncing = False

        # ドラッグ用の基準状態
        self._drag_ref_positions = self.atoms.get_positions().copy()
        self._drag_ref_slider = (0.0, 0.0, 0.0)
        self._drag_valid = False

        # ----------------------
        # ウィジェット作成
        # ----------------------

        # 複数選択モード
        self.multi_select = widgets.ToggleButton(
            value=False, description="複数選択モード", icon="check"
        )

        # 現在の選択状態表示
        self.selection_label = widgets.Label("選択中の原子: なし")

        # 原子 ID スライダー
        n_atoms = len(atoms)
        self.atom_id = widgets.IntSlider(
            description="atom id",
            min=0,
            max=n_atoms - 1,
            value=0,
            continuous_update=False,  # これは「選択」なのでドラッグ終了時でOK
        )

        # 座標スライダーの範囲設定
        pos = atoms.get_positions()
        x_min, x_max = pos[:, 0].min() - 5, pos[:, 0].max() + 5
        y_min, y_max = pos[:, 1].min() - 5, pos[:, 1].max() + 5
        z_min, z_max = pos[:, 2].min() - 5, pos[:, 2].max() + 5

        self.x_slider = widgets.FloatSlider(
            description="x",
            min=x_min, max=x_max,
            value=pos[0, 0],
            step=0.02,
            continuous_update=True,
        )
        self.y_slider = widgets.FloatSlider(
            description="y",
            min=y_min, max=y_max,
            value=pos[0, 1],
            step=0.02,
            continuous_update=True,
        )
        self.z_slider = widgets.FloatSlider(
            description="z",
            min=z_min, max=z_max,
            value=pos[0, 2],
            step=0.02,
            continuous_update=True,
        )

        # ボールサイズスライダー
        self.ball_size = widgets.FloatSlider(
            description="ball size",
            min=0.1, max=2.0,
            value=0.5,
            step=0.05,
            continuous_update=True,
        )

        # 表示形式 (ball+stick / spacefill / licorice)
        self.display_mode = widgets.Dropdown(
            options=["ball+stick", "spacefill", "licorice"],
            value="ball+stick",
            description="表示形式",
        )

        # タグによる選択
        tags = [atom.tag for atom in atoms]
        self.unique_tags = sorted(set(tags))
        self.tag_selector = widgets.Dropdown(
            options=self.unique_tags,
            description="Tag選択",
        )
        self.select_tag_button = widgets.Button(
            description="このTagを選択",
            button_style="info",
        )

        # ----------------------
        # イベント設定
        # ----------------------
        self.atom_id.observe(self.sync_sliders_from_atom, names="value")

        for s in (self.x_slider, self.y_slider, self.z_slider):
            s.observe(self.apply_position, names="value")

        self.ball_size.observe(self.on_ball_size_change, names="value")
        self.display_mode.observe(self.on_display_mode_change, names="value")
        self.select_tag_button.on_click(self.on_select_tag)

        # nglview の picking（クリック選択）
        self.view.observe(self.on_picked, names=["picked"])

        # 初期表示形式を反映（view_ngl が追加した既存 rep を上書き）
        self._init_representation()

        # UI 組み立て
        self.ui = self._build_ui()

    # =============================
    # 内部ロジック
    # =============================

    def _init_representation(self):
        """初期状態で display_mode の内容に合わせて representation を設定"""
        self.on_display_mode_change({"new": self.display_mode.value})

    def sync_sliders_from_atom(self, change=None):
        """atom_id が変わったときに、その原子の座標をスライダーへ反映"""
        self.is_syncing = True

        idx = self.atom_id.value
        p = self.atoms.positions[idx]
        self.x_slider.value = float(p[0])
        self.y_slider.value = float(p[1])
        self.z_slider.value = float(p[2])

        self.is_syncing = False
        self._drag_valid = False   # 選択変更時はドラッグ基準をリセット

    def apply_position(self, change=None):
        """スライダー操作に応じて Atoms + ビューをリアルタイム更新"""
        if self.is_syncing:
            return

        positions = self.atoms.get_positions()
        target_indices = sorted(self.selected_atoms) if self.selected_atoms else [self.atom_id.value]

        # ドラッグ開始時に基準を記録（1 回だけ）
        if not self._drag_valid:
            self._drag_ref_positions = positions.copy()
            self._drag_ref_slider = (
                self.x_slider.value,
                self.y_slider.value,
                self.z_slider.value,
            )
            self._drag_valid = True

        x0, y0, z0 = self._drag_ref_slider
        dx = self.x_slider.value - x0
        dy = self.y_slider.value - y0
        dz = self.z_slider.value - z0

        new_positions = self._drag_ref_positions.copy()

        # 選択された原子を平行移動（相対配置を維持）
        for i in target_indices:
            new_positions[i, 0] = self._drag_ref_positions[i, 0] + dx
            new_positions[i, 1] = self._drag_ref_positions[i, 1] + dy
            new_positions[i, 2] = self._drag_ref_positions[i, 2] + dz

        self.atoms.set_positions(new_positions)

        coords = new_positions.astype("float32")
        try:
            # 座標だけを高速更新
            self.view.set_coordinates({0: coords})
        except Exception:
            # 互換性の問題があればフル更新
            self.view.update_ase(self.atoms)

    def on_ball_size_change(self, change):
        """ボールサイズスライダーの変更に応じて半径を更新"""
        mode = self.display_mode.value
        scale = change["new"]

        if mode == "ball+stick":
            self.view.update_ball_and_stick(
                radiusScale=scale,
                colorScheme="element",
            )
        elif mode == "spacefill":
            self.view.update_spacefill(
                radiusScale=scale,
                colorScheme="element",
            )
        elif mode == "licorice":
            self.view.update_licorice(
                radiusScale=scale,
                colorScheme="element",
            )

    def on_display_mode_change(self, change):
        """表示形式 (ball+stick / spacefill / licorice) の切り替え"""
        mode = change["new"]

        # いったん全ての表現をクリア
        self.view.clear_representations()

        # 原子の表示を追加
        if mode == "ball+stick":
            self.view.add_ball_and_stick(
                radiusScale=self.ball_size.value,
                colorScheme="element",
            )
        elif mode == "spacefill":
            self.view.add_spacefill(
                radiusScale=self.ball_size.value,
                colorScheme="element",
            )
        elif mode == "licorice":
            self.view.add_licorice(
                radiusScale=self.ball_size.value,
                colorScheme="element",
            )

        # 周期境界セルがある場合は unitcell を表示する
        try:
            if getattr(self.atoms, "pbc", None) is not None and np.any(self.atoms.pbc):
                self.view.add_unitcell()
        except Exception:
            # pbc が無い / 変な形式などの場合は黙ってスキップ
            pass

    def on_picked(self, change):
        """3D ビューで原子をクリックしたときの処理"""
        picked = change["new"]
        if "atom1" not in picked:
            return

        idx = picked["atom1"].get("index", None)
        if idx is None:
            return

        if self.multi_select.value:
            # 複数選択モード：クリックで on/off
            if idx in self.selected_atoms:
                self.selected_atoms.remove(idx)
            else:
                self.selected_atoms.add(idx)
        else:
            # 単一選択モード：その原子のみ
            self.selected_atoms = {idx}
            self.atom_id.value = idx  # スライダーも同期

        if self.selected_atoms:
            self.selection_label.value = "選択中の原子: " + ", ".join(
                str(i) for i in sorted(self.selected_atoms)
            )
        else:
            self.selection_label.value = "選択中の原子: なし"

        self._drag_valid = False   # 選択変更時はドラッグ基準をリセット

    def on_select_tag(self, btn):
        """Tag を指定して、その Tag を持つ原子を一括選択"""
        tag = self.tag_selector.value
        self.selected_atoms = {i for i, a in enumerate(self.atoms) if a.tag == tag}

        if self.selected_atoms:
            self.selection_label.value = "選択中（Tag={}）: {}".format(
                tag, ", ".join(str(i) for i in sorted(self.selected_atoms))
            )
            # 代表として1つ atom_id に反映
            self.atom_id.value = next(iter(self.selected_atoms))
        else:
            self.selection_label.value = f"Tag={tag} の原子はありません"

        self._drag_valid = False   # 選択変更時はドラッグ基準をリセット

    # =============================
    # UI レイアウト構築
    # =============================

    def _build_ui(self):
        editor_box = widgets.VBox(
            [
                widgets.HTML("<b>原子位置編集</b>"),
                self.multi_select,
                self.selection_label,
                self.atom_id,
                self.x_slider,
                self.y_slider,
                self.z_slider,
                widgets.HTML("<b>Tagによる選択</b>"),
                self.tag_selector,
                self.select_tag_button,
                widgets.HTML("<b>表示設定</b>"),
                self.display_mode,
                self.ball_size,
            ]
        )

        # 左: ngl ビュー + 標準コントロール, 右: 編集用パネル
        return widgets.HBox([self.ngl_gui, editor_box])

    # =============================
    # Notebook 用表示
    # =============================

    def display(self):
        """Jupyter Notebook 上に UI を表示"""
        display(self.ui)
