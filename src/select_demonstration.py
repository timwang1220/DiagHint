# select_demonstration.py
# Select demonstration examples using tree similarity model

import os
import sys
import glob
import json
import pickle
import tempfile
import importlib.util
import re
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Add src to path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from plan_node.tree_model import TreePairClassifier
from plan_node.encoder import build_tree_from_plan_json
from plan_node.embedding import TextEncoder


def choose_sql_by_filename(filename: str, demo_base: str = os.path.join(ROOT_DIR, "outputs", "demo_pool")):
    """
    根据文件名选择演示SQL数据。
    :param filename: 测试查询文件名，如 '17b'
    :param demo_base: 演示数据根目录
    :return: (sql, demo_base_plan, demo_recommended_hint, demo_hinted_plan, infer_reason)
    """
    # get the number in filename,like 17b -> 17
    template_number = int(''.join(filter(str.isdigit, filename)))
    demonstrate_folder = os.path.join(demo_base, str(template_number) + 'a')

    # read the base execution plan 'original_execution_plan.txt'
    with open(os.path.join(demonstrate_folder, 'original_execution_plan.txt'), 'r') as f:
        original_execution_plan = f.read()

    # read the suggested hint 'suggest_hint.txt'
    with open(os.path.join(demonstrate_folder, 'suggest_hint.txt'), 'r') as f:
        suggested_hint = f.read()

    # read the execution plan with hint 'execution_plan.txt'
    with open(os.path.join(demonstrate_folder, 'execution_plan.txt'), 'r') as f:
        execution_plan = f.read()

    # read the inference reason 'infer_reason.txt'
    try:
        with open(os.path.join(demonstrate_folder, 'infer_reason.txt'), 'r') as f:
            infer_reason = f.read()
    except FileNotFoundError:
        infer_reason = "Optimization based on execution plan analysis and performance patterns."

    # read the SQL query
    sql_file = os.path.join(demonstrate_folder, 'query.sql')
    with open(sql_file, 'r') as f:
        sql = f.read()

    return sql, original_execution_plan, suggested_hint, execution_plan, infer_reason


class UtilityDemoSelector:
    """Select demo via utility model score(source_plan, target_plan)."""

    def __init__(
        self,
        demo_pool_dir: str,
        ckpt_path: str = os.path.join(ROOT_DIR, "models", "utility", "best.pt"),
        artifacts_dir: str = os.path.join(ROOT_DIR, "models", "cardinality_bias"),
        text_model_path: str = os.environ.get("DIAGHINT_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        device: str = "cpu",
        db_name: str = "",
        encoder_mode: str = "",
        encoder_artifacts_dir: str = "",
        use_predicate_pca: Optional[bool] = None,
    ):
        self.demo_pool_dir = os.path.abspath(demo_pool_dir)
        self.ckpt_path = ckpt_path
        self.artifacts_dir = artifacts_dir
        self.text_model_path = text_model_path
        self.device = device
        self.db_name = db_name
        self.encoder_mode_override = encoder_mode
        self.encoder_artifacts_dir_override = encoder_artifacts_dir
        self.use_predicate_pca_override = use_predicate_pca
        self.utility_dir = self._resolve_utility_dir()
        self.last_selected_demo_id: Optional[str] = None
        self.last_selected_score: Optional[float] = None
        self.last_selected_best_hint: Optional[str] = None
        self._load_modules_and_model()
        self._build_demo_index()

    def _resolve_utility_dir(self) -> str:
        candidates = [
            os.path.join(ROOT_DIR, "src", "utility-model"),
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        raise RuntimeError("Cannot find utility model directory (utility-model / utilty-model / utilty-model0)")

    def _load_local_module(self, module_name: str, filename: str):
        path = os.path.join(self.utility_dir, filename)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        # Register module before execution so decorators (e.g., @dataclass)
        # can resolve cls.__module__ via sys.modules correctly.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_modules_and_model(self) -> None:
        util_dir = self.utility_dir
        if util_dir not in sys.path:
            sys.path.insert(0, util_dir)

        model_mod = self._load_local_module("utility_model_local_runtime", "model.py")
        data_mod = self._load_local_module("utility_data_local_runtime", "data_utils.py")
        SourceTargetTreeModel = model_mod.SourceTargetTreeModel
        PlanTreeEncoder = data_mod.PlanTreeEncoder
        encode_hint_opt_vec = getattr(data_mod, "encode_hint_opt_vec")
        self.encode_hint_opt_vec = encode_hint_opt_vec

        state = torch.load(self.ckpt_path, map_location=self.device)
        cfg = state["config"]
        self.score_shift = float(cfg.get("score_shift", 0.0))
        ckpt_encoder_mode = str(cfg.get("encoder_mode", "current"))
        ckpt_encoder_artifacts_dir = str(cfg.get("encoder_artifacts_dir", "") or "")
        ckpt_use_predicate_pca = bool(cfg.get("use_predicate_pca", True))
        effective_encoder_mode = self.encoder_mode_override.strip() or ckpt_encoder_mode
        effective_encoder_artifacts_dir = self.encoder_artifacts_dir_override.strip() or ckpt_encoder_artifacts_dir
        effective_use_predicate_pca = ckpt_use_predicate_pca if self.use_predicate_pca_override is None else bool(self.use_predicate_pca_override)

        model = SourceTargetTreeModel(
            node_input_dim=int(cfg["feature_dim"]),
            tree_hidden_dim=int(cfg.get("tree_hidden_dim", cfg.get("hidden_dim", 128))),
            scorer_hidden_dim=int(cfg.get("scorer_hidden_dim", 128)),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(self.device)
        try:
            model.load_state_dict(state["model_state_dict"])
        except RuntimeError:
            model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()

        self.model = model
        self.tree_encoder = PlanTreeEncoder(
            artifacts_dir=self.artifacts_dir,
            predicate_fit_dir=self.demo_pool_dir,
            model_name=self.text_model_path,
            text_device=self.device,
            db_name=(self.db_name.strip() or None),
            encoder_mode=effective_encoder_mode,
            encoder_artifacts_dir=effective_encoder_artifacts_dir,
            use_predicate_pca=effective_use_predicate_pca,
        )

    def _build_demo_index(self) -> None:
        self.demo_records = []
        for name in sorted(os.listdir(self.demo_pool_dir)):
            d = os.path.join(self.demo_pool_dir, name)
            if not os.path.isdir(d):
                continue
            plan0 = os.path.join(d, "0", "plan.json")
            sql = os.path.join(d, "query.sql")
            base = os.path.join(d, "original_execution_plan.txt")
            hint = os.path.join(d, "suggest_hint.txt")
            hinted = os.path.join(d, "execution_plan.txt")
            if not (os.path.exists(plan0) and os.path.exists(sql) and os.path.exists(base) and os.path.exists(hint) and os.path.exists(hinted)):
                continue

            sf, sc = self.tree_encoder.encode_path(plan0)
            hint_text = ""
            final_hint = os.path.join(d, "final_combined_hint.txt")
            suggest_hint = os.path.join(d, "suggest_hint.txt")
            for hp in (final_hint, suggest_hint):
                if os.path.exists(hp):
                    with open(hp, "r", encoding="utf-8") as f:
                        hint_text = f.read().strip()
                    if hint_text:
                        break
            if not hint_text:
                hint_text = "/*+ */"
            opt_vec = torch.tensor(self.encode_hint_opt_vec(hint_text), dtype=torch.float32)
            self.demo_records.append(
                {
                    "demo_id": name,
                    "demo_dir": d,
                    "source_features": sf.to(self.device),
                    "source_children": sc.to(self.device),
                    "source_opt_vec": opt_vec.to(self.device),
                }
            )

        if not self.demo_records:
            raise RuntimeError(f"No valid demos found in {self.demo_pool_dir}")

    @staticmethod
    def _template_num(name: str) -> str:
        out = []
        for ch in (name or ""):
            if ch.isdigit():
                out.append(ch)
            else:
                break
        return "".join(out)

    @staticmethod
    def _is_effectively_empty_hint(hint: str) -> bool:
        text = (hint or "").strip()
        if not text:
            return True
        return bool(re.fullmatch(r"/\*\+\s*\*/", text, flags=re.DOTALL))

    def _load_demo_payload(self, demo_dir: str) -> Tuple[str, str, str, str, str]:
        def _read(path: str, default: str = "") -> str:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return default

        demo_sql = _read(os.path.join(demo_dir, "query.sql"))
        demo_base_plan = _read(os.path.join(demo_dir, "original_execution_plan.txt"))
        demo_recommended_hint = _read(os.path.join(demo_dir, "suggest_hint.txt"))
        demo_hinted_plan = _read(os.path.join(demo_dir, "execution_plan.txt"))
        infer_reason = _read(os.path.join(demo_dir, "infer_reason.txt"), "N/A")
        best_hint = _read(
            os.path.join(demo_dir, "final_combined_hint.txt"),
            demo_recommended_hint,
        ).strip()
        if not best_hint:
            best_hint = demo_recommended_hint.strip()
        self.last_selected_best_hint = best_hint
        return demo_sql, demo_base_plan, demo_recommended_hint, demo_hinted_plan, infer_reason

    @torch.no_grad()
    def select_from_raw_plan(self, raw_plan: dict, target_name: str = "target") -> Tuple[str, str, str, str, str]:
        with tempfile.NamedTemporaryFile("w", suffix=f"_{target_name}_plan.json", delete=False) as tf:
            json.dump(raw_plan, tf)
            tmp_path = tf.name

        try:
            tf_feat, tf_child = self.tree_encoder.encode_path(tmp_path)
            tf_feat = tf_feat.to(self.device)
            tf_child = tf_child.to(self.device)

            best_idx = -1
            best_score = float("-inf")
            target_template = self._template_num(str(target_name))
            for idx, rec in enumerate(self.demo_records):
                source_template = self._template_num(str(rec["demo_id"]))
                reuse = 1.0 if (target_template and source_template == target_template) else 0.0
                out = self.model(
                    source_features=[rec["source_features"]],
                    source_children=[rec["source_children"]],
                    target_features=[tf_feat],
                    target_children=[tf_child],
                    reuse=torch.tensor([[reuse]], dtype=torch.float32, device=self.device),
                    opt_vec=rec["source_opt_vec"].unsqueeze(0),
                )
                score = float(out["score"][0].item() + self.score_shift)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            best_demo = self.demo_records[best_idx]
            self.last_selected_score = float(best_score)
            if best_score < 0.0:
                # Do not select demo when calibrated score is negative.
                self.last_selected_demo_id = None
                self.last_selected_best_hint = None
                print(f"Utility model skipped demo (best score={best_score:.5f} < 0)")
                return "", "", "", "", "No reliable demonstration selected."

            self.last_selected_demo_id = str(best_demo["demo_id"])
            print(f"Utility model selected demo: {best_demo['demo_id']} (score={best_score:.5f})")
            return self._load_demo_payload(best_demo["demo_dir"])
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @torch.no_grad()
    def select_same_template_if_positive(
        self,
        raw_plan: dict,
        target_template: str,
        target_name: str = "target",
    ) -> Optional[Tuple[str, str, str, str, str]]:
        """Try same-template demo first; return payload only when best score > 0."""
        if not target_template:
            return None

        with tempfile.NamedTemporaryFile("w", suffix=f"_{target_name}_plan.json", delete=False) as tf:
            json.dump(raw_plan, tf)
            tmp_path = tf.name

        try:
            tf_feat, tf_child = self.tree_encoder.encode_path(tmp_path)
            tf_feat = tf_feat.to(self.device)
            tf_child = tf_child.to(self.device)

            candidate_idx: List[int] = []
            for idx, rec in enumerate(self.demo_records):
                if self._template_num(str(rec["demo_id"])) == target_template:
                    candidate_idx.append(idx)
            if not candidate_idx:
                return None

            best_idx = -1
            best_score = float("-inf")
            for idx in candidate_idx:
                rec = self.demo_records[idx]
                out = self.model(
                    source_features=[rec["source_features"]],
                    source_children=[rec["source_children"]],
                    target_features=[tf_feat],
                    target_children=[tf_child],
                    reuse=torch.tensor([[1.0]], dtype=torch.float32, device=self.device),
                    opt_vec=rec["source_opt_vec"].unsqueeze(0),
                )
                score = float(out["score"][0].item() + self.score_shift)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1 or best_score <= 0.0:
                return None

            best_demo = self.demo_records[best_idx]
            self.last_selected_demo_id = str(best_demo["demo_id"])
            self.last_selected_score = float(best_score)
            print(
                f"Same-template utility hit: demo={best_demo['demo_id']} "
                f"(score={best_score:.5f} > 0)"
            )
            return self._load_demo_payload(best_demo["demo_dir"])
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @torch.no_grad()
    def select_in_context_nonempty_if_positive(
        self,
        raw_plan: dict,
        target_template: str,
        target_name: str = "target",
        allow_non_positive: bool = False,
    ) -> Optional[Tuple[str, str, str, str, str]]:
        """Select from non-empty-hint in-context demos.

        By default, return payload only when best score > 0.
        When allow_non_positive=True, return best payload regardless of score.
        """
        with tempfile.NamedTemporaryFile("w", suffix=f"_{target_name}_plan.json", delete=False) as tf:
            json.dump(raw_plan, tf)
            tmp_path = tf.name

        try:
            tf_feat, tf_child = self.tree_encoder.encode_path(tmp_path)
            tf_feat = tf_feat.to(self.device)
            tf_child = tf_child.to(self.device)

            candidate_idx: List[int] = []
            for idx, rec in enumerate(self.demo_records):
                demo_id = str(rec["demo_id"])
                source_template = self._template_num(demo_id)
                # in-context only: exclude same-template demos
                if target_template and source_template == target_template:
                    continue

                hint_text = ""
                final_hint = os.path.join(str(rec["demo_dir"]), "final_combined_hint.txt")
                suggest_hint = os.path.join(str(rec["demo_dir"]), "suggest_hint.txt")
                for hp in (final_hint, suggest_hint):
                    if os.path.exists(hp):
                        with open(hp, "r", encoding="utf-8") as f:
                            hint_text = f.read().strip()
                        if hint_text:
                            break
                if self._is_effectively_empty_hint(hint_text):
                    continue
                candidate_idx.append(idx)

            if not candidate_idx:
                self.last_selected_demo_id = None
                self.last_selected_score = None
                self.last_selected_best_hint = None
                print("In-context selector skipped: no non-empty-hint candidates.")
                return None

            best_idx = -1
            best_score = float("-inf")
            for idx in candidate_idx:
                rec = self.demo_records[idx]
                out = self.model(
                    source_features=[rec["source_features"]],
                    source_children=[rec["source_children"]],
                    target_features=[tf_feat],
                    target_children=[tf_child],
                    reuse=torch.tensor([[0.0]], dtype=torch.float32, device=self.device),
                    opt_vec=rec["source_opt_vec"].unsqueeze(0),
                )
                score = float(out["score"][0].item() + self.score_shift)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1:
                self.last_selected_demo_id = None
                self.last_selected_score = float("-inf")
                self.last_selected_best_hint = None
                print("In-context selector skipped: no valid candidate index.")
                return None

            if best_score <= 0.0 and not allow_non_positive:
                self.last_selected_demo_id = None
                self.last_selected_score = float(best_score)
                self.last_selected_best_hint = None
                print(f"In-context selector skipped: best non-empty score={best_score:.5f} <= 0")
                return None

            best_demo = self.demo_records[best_idx]
            self.last_selected_demo_id = str(best_demo["demo_id"])
            self.last_selected_score = float(best_score)
            if best_score > 0.0:
                print(
                    f"In-context utility hit: demo={best_demo['demo_id']} "
                    f"(score={best_score:.5f} > 0, non-empty hint)"
                )
            else:
                print(
                    f"In-context fallback hit: demo={best_demo['demo_id']} "
                    f"(score={best_score:.5f} <= 0, non-empty hint, still call LLM)"
                )
            return self._load_demo_payload(best_demo["demo_dir"])
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @torch.no_grad()
    def select_top_in_context_nonempty(
        self,
        raw_plan: dict,
        target_template: str,
        target_name: str = "target",
        top_k: int = 1,
        allow_non_positive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return top-k non-empty in-context demos ranked by utility score."""
        if top_k <= 0:
            return []

        with tempfile.NamedTemporaryFile("w", suffix=f"_{target_name}_plan.json", delete=False) as tf:
            json.dump(raw_plan, tf)
            tmp_path = tf.name

        try:
            tf_feat, tf_child = self.tree_encoder.encode_path(tmp_path)
            tf_feat = tf_feat.to(self.device)
            tf_child = tf_child.to(self.device)

            scored: List[Dict[str, Any]] = []
            for rec in self.demo_records:
                demo_id = str(rec["demo_id"])
                source_template = self._template_num(demo_id)
                if target_template and source_template == target_template:
                    continue

                hint_text = ""
                final_hint = os.path.join(str(rec["demo_dir"]), "final_combined_hint.txt")
                suggest_hint = os.path.join(str(rec["demo_dir"]), "suggest_hint.txt")
                for hp in (final_hint, suggest_hint):
                    if os.path.exists(hp):
                        with open(hp, "r", encoding="utf-8") as f:
                            hint_text = f.read().strip()
                        if hint_text:
                            break
                if self._is_effectively_empty_hint(hint_text):
                    continue

                out = self.model(
                    source_features=[rec["source_features"]],
                    source_children=[rec["source_children"]],
                    target_features=[tf_feat],
                    target_children=[tf_child],
                    reuse=torch.tensor([[0.0]], dtype=torch.float32, device=self.device),
                    opt_vec=rec["source_opt_vec"].unsqueeze(0),
                )
                score = float(out["score"][0].item() + self.score_shift)
                if (score <= 0.0) and (not allow_non_positive):
                    continue
                scored.append(
                    {
                        "demo_id": demo_id,
                        "score": score,
                        "payload": self._load_demo_payload(str(rec["demo_dir"])),
                        "best_hint": self.last_selected_best_hint,
                    }
                )

            scored.sort(key=lambda x: x["score"], reverse=True)
            picked = scored[:top_k]
            if not picked:
                self.last_selected_demo_id = None
                self.last_selected_score = None
                self.last_selected_best_hint = None
                print("Top-k in-context selector skipped: no eligible non-empty candidates.")
                return []

            first = picked[0]
            self.last_selected_demo_id = str(first["demo_id"])
            self.last_selected_score = float(first["score"])
            self.last_selected_best_hint = str(first.get("best_hint") or "")
            print(
                "Top-k in-context utility hit: "
                + ", ".join(f"{item['demo_id']}({item['score']:.5f})" for item in picked)
            )
            return picked
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class TreeSimilaritySearcher:
    """
    使用树-LSTM模型查找相似的查询执行计划。

    数据结构说明:
    - 每个演示文件夹（如 1a, 2a）包含子文件夹 0-8
    - 每个子文件夹包含一个 plan.json（不同hint下的执行计划）
    - 父文件夹包含: suggest_hint.txt, execution_plan.txt, infer_reason.txt,
                    best_global_hint.txt, best_leading_hint.txt, best_node_hint.txt
    - 子文件夹包含: plan.json, hint.txt (可选)
    """

    def __init__(
        self,
        demo_fold: str,
        model_path: str = os.path.join(ROOT_DIR, "models", "tree_similarity", "model.pt"),
        model_type: str = 'classifier',  # 'classifier' or 'similarity'
        hidden_dim: int = 256,
        dropout: float = 0.2,
        device: str = 'cpu',
        embedding_artifacts_dir: str = os.path.join(ROOT_DIR, "models", "embedding_artifacts"),
    ):
        """
        初始化树相似度搜索器。

        Args:
            demo_fold: 演示数据目录，包含多个子文件夹（如 '1a/', '2a/'）
            model_path: 训练好的模型路径
            model_type: 模型类型 ('classifier' 或 'similarity')
            hidden_dim: 模型隐藏层维度
            dropout: Dropout率
            device: 运行设备 ('cpu' 或 'cuda')
            embedding_artifacts_dir: TextEncoder artifacts目录
        """
        self.demo_fold = demo_fold
        self.model_path = model_path
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.embedding_artifacts_dir = embedding_artifacts_dir

        # 池数据: List of Dict {
        #   'parent': '1a',  # 父文件夹名称
        #   'subfolder': '0',  # 子文件夹名称 (0-8)
        #   'tree': {...},  # 树结构
        #   'parent_path': '/path/to/1a',  # 父文件夹路径
        # }
        self.plan_pool = []

        # 模型和encoder延迟加载
        self.model = None
        self.encoder = None
        self.norm_stats = None

    def _load_model_and_encoder(self):
        """延迟加载模型和encoder。"""
        if self.model is None:
            # 加载TextEncoder
            from plan_node.embedding import TextEncoder

            self.encoder = TextEncoder()
            self.encoder.load_cache(self.embedding_artifacts_dir)

            # 加载norm_stats
            norm_stats_path = os.path.join(self.embedding_artifacts_dir, "norm_stats.npy")
            self.norm_stats = tuple(np.load(norm_stats_path))

            # 加载树相似度模型
            feature_dim = 1163  # 6 + 384*3 + 5

            if self.model_type == 'classifier':
                self.model = TreePairClassifier(
                    feature_dim=feature_dim,
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                )
            else:
                from plan_node.tree_model import TreeSimilarityModel
                self.model = TreeSimilarityModel(
                    feature_dim=feature_dim,
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                )

            # 加载模型权重
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

    def build_pool(
        self,
        save_path: str = os.path.join(ROOT_DIR, "outputs", "plan_tree_pool.pkl"),
    ):
        """
        构建执行计划树池。
        遍历每个父文件夹（1a, 2a, ...）的每个子文件夹（0-8），读取其中的 plan.json。

        Args:
            save_path: 保存池的路径
        """
        print(f"开始从 '{self.demo_fold}' 构建执行计划树池...")

        # 初始化模型和encoder
        self._load_model_and_encoder()

        # 查找所有父文件夹
        parent_folders = glob.glob(os.path.join(self.demo_fold, '*/'))

        self.plan_pool = []
        total_count = 0
        success_count = 0

        for parent_path in parent_folders:
            parent_name = os.path.basename(os.path.normpath(parent_path))

            # 查找所有子文件夹（通常是数字文件夹 0-8）
            subfolders = glob.glob(os.path.join(parent_path, '*/'))

            for subfolder_path in subfolders:
                subfolder_name = os.path.basename(os.path.normpath(subfolder_path))
                plan_file_path = os.path.join(subfolder_path, 'plan.json')

                total_count += 1

                if os.path.exists(plan_file_path):
                    try:
                        with open(plan_file_path, 'r', encoding='utf-8') as f:
                            plan_data = json.load(f)

                        # 构建树结构
                        tree = build_tree_from_plan_json(
                            plan_data,
                            self.encoder,
                            self.norm_stats,
                        )

                        if tree is not None and len(tree['features']) > 0:
                            self.plan_pool.append({
                                'parent': parent_name,
                                'subfolder': subfolder_name,
                                'tree': tree,
                                'parent_path': parent_path,
                            })
                            success_count += 1

                    except Exception as e:
                        print(f"处理文件 '{plan_file_path}' 时出错: {e}")

        print(f"扫描完成: 总计 {total_count} 个计划文件，成功加载 {success_count} 个执行计划树。")

        # 保存池
        try:
            save_data = {
                'plan_pool': self.plan_pool,
                'demo_fold': self.demo_fold,
            }
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"执行计划树池已保存到 '{save_path}'。")
        except Exception as e:
            print(f"保存文件 '{save_path}' 时出错: {e}")

    def load_pool(self, path: str = os.path.join(ROOT_DIR, "outputs", "plan_tree_pool.pkl")):
        """加载已保存的计划树池。"""
        print(f"从 '{path}' 加载执行计划树池...")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.plan_pool = data['plan_pool']
        # 保存时可能记录了demo_fold，确保使用传入的路径
        if 'demo_fold' in data:
            self.demo_fold = data['demo_fold']

        print(f"成功加载 {len(self.plan_pool)} 个执行计划树。")

    def _plan_to_tree(
        self,
        plan_json: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        将执行计划JSON转换为树结构。

        Args:
            plan_json: 执行计划JSON

        Returns:
            树结构字典，如果转换失败则返回None
        """
        # 确保模型和encoder已加载
        self._load_model_and_encoder()

        try:
            tree = build_tree_from_plan_json(
                plan_json,
                self.encoder,
                self.norm_stats,
            )
            return tree
        except Exception as e:
            print(f"将计划转换为树时出错: {e}")
            return None

    def get_all_similarities(
        self,
        input_plan: Dict[str, Any],
    ) -> torch.Tensor:
        """
        计算输入执行计划与池中所有计划的相似度分数。

        Args:
            input_plan: 输入执行计划JSON

        Returns:
            相似度分数张量
        """
        # 转换输入计划为树
        input_tree = self._plan_to_tree(input_plan)
        if input_tree is None or len(input_tree['features']) == 0:
            return torch.zeros(len(self.plan_pool))

        # 确保模型已加载
        self._load_model_and_encoder()

        # 计算相似度
        with torch.no_grad():
            similarities = []

            for pool_item in self.plan_pool:
                pool_tree = pool_item['tree']

                # 准备输入
                features1 = torch.tensor(
                    input_tree['features'],
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)
                children1 = torch.tensor(
                    input_tree['children'],
                    dtype=torch.long,
                ).unsqueeze(0).to(self.device)

                features2 = torch.tensor(
                    pool_tree['features'],
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)
                children2 = torch.tensor(
                    pool_tree['children'],
                    dtype=torch.long,
                ).unsqueeze(0).to(self.device)

                # 计算相似度
                if self.model_type == 'classifier':
                    prob = self.model(
                        (features1, children1),
                        (features2, children2),
                    )
                    similarities.append(prob.item())
                else:
                    similarity = self.model(
                        (features1, children1),
                        (features2, children2),
                    )
                    similarities.append(similarity.item())

        return torch.tensor(similarities)

    def _read_file_safe(self, file_path: str, default: str = "") -> str:
        """安全地读取文件，如果文件不存在则返回默认值。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return default
        except Exception as e:
            print(f"读取文件 '{file_path}' 时出错: {e}")
            return default

    def search(
        self,
        input_plan: Dict[str, Any],
        top_k: int = 1,
    ) -> Tuple[str, str, str, str, str, str, str, str]:
        """
        搜索与输入执行计划最相似的演示。

        Args:
            input_plan: 输入执行计划JSON
            top_k: 返回前k个最相似的结果

        Returns:
            (sql, original_execution_plan, suggested_hint, hinted_hint,
             execution_plan, infer_reason, best_global_hint, best_leading_hint, best_node_hint)
            - sql: 演示SQL查询
            - original_execution_plan: 原始执行计划文本
            - suggested_hint: 建议的hint
            - hinted_hint: 子文件夹中的hint.txt内容（如果不存在则为 "/*+ */"）
            - execution_plan: 加hint后的执行计划文本
            - infer_reason: 推理原因
            - best_global_hint: 最佳全局hint
            - best_leading_hint: 最佳leading hint
            - best_node_hint: 最佳节点hint
        """
        # 计算所有相似度
        similarities = self.get_all_similarities(input_plan)

        # 获取top-k
        top_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices

        # 使用第一个（最相似的）
        best_idx = top_indices[0].item()
        best_match = self.plan_pool[best_idx]

        parent_name = best_match['parent']
        subfolder_name = best_match['subfolder']
        parent_path = best_match['parent_path']


        print(f"parent_path: {parent_path}")

        print(f"找到最相似的演示: 父文件夹 '{parent_name}', 子文件夹 '{subfolder_name}'，相似度分数: {similarities[best_idx].item():.4f}")
        # 读取父文件夹下的文件
        suggest_hint_path = os.path.join(parent_path, 'suggest_hint.txt')
        execution_plan_path = os.path.join(parent_path, 'execution_plan.txt')
        infer_reason_path = os.path.join(parent_path, 'infer_reason.txt')
        best_global_hint_path = os.path.join(parent_path, 'best_global_hint.txt')
        best_leading_hint_path = os.path.join(parent_path, 'best_leading_hint.txt')
        best_node_hint_path = os.path.join(parent_path, 'best_node_hint.txt')

        # 读取hinted_plan.txt 或尝试从子文件夹的plan.json读取
        subfolder_path = os.path.join(parent_path, subfolder_name)
        hint_txt_path = os.path.join(subfolder_path, 'hint.txt')

        # 读取各个文件
        suggested_hint = self._read_file_safe(suggest_hint_path)
        execution_plan = self._read_file_safe(execution_plan_path)
        infer_reason = self._read_file_safe(infer_reason_path, "Optimization based on execution plan analysis.")
        best_global_hint = self._read_file_safe(best_global_hint_path)
        best_leading_hint = self._read_file_safe(best_leading_hint_path)
        best_node_hint = self._read_file_safe(best_node_hint_path)

        # 读取子文件夹中的hint.txt，如果不存在则使用空hint
        hinted_hint = self._read_file_safe(hint_txt_path, "/*+ */")

        # 如果hinted_hint是空的，使用 "/*+ */"
        if not hinted_hint.strip() or hinted_hint == "":
            hinted_hint = "/*+ */"

        # 读取original_execution_plan.txt或从子文件夹的plan.json读取
        original_plan_path = os.path.join(parent_path, 'original_execution_plan.txt')
        original_execution_plan = self._read_file_safe(original_plan_path)
        if not original_execution_plan:
            # 尝试从子文件夹的plan.json读取并转换
            plan_json_path = os.path.join(subfolder_path, 'plan.json')
            try:
                with open(plan_json_path, 'r') as f:
                    plan_data = json.load(f)
                original_execution_plan = json.dumps(plan_data, indent=2)
            except:
                original_execution_plan = ""

        # 读取SQL文件
        sql = ""
        sql_files = glob.glob(os.path.join(parent_path, '*.sql'))
        if sql_files:
            sql = self._read_file_safe(sql_files[0])

        # 如果没有找到，尝试query.sql
        if not sql:
            query_sql_path = os.path.join(parent_path, 'query.sql')
            sql = self._read_file_safe(query_sql_path)

        return (
            sql,
            original_execution_plan,
            suggested_hint,
            hinted_hint,
            execution_plan,
            infer_reason,
            best_global_hint,
            best_leading_hint,
            best_node_hint,
        )


def parse_plan_string(plan_str: str) -> Dict[str, Any]:
    """
    将计划字符串解析为JSON字典。

    Args:
        plan_str: 计划字符串（可能是JSON或EXPLAIN输出）

    Returns:
        计划JSON字典
    """
    import json

    plan_str = plan_str.strip()

    # 尝试直接解析JSON
    if plan_str.startswith('{'):
        try:
            return json.loads(plan_str)
        except json.JSONDecodeError:
            pass

    # 尝试查找JSON部分
    start_idx = plan_str.find('{')
    if start_idx != -1:
        # 尝试找到匹配的结束括号
        depth = 0
        for i in range(start_idx, len(plan_str)):
            if plan_str[i] == '{':
                depth += 1
            elif plan_str[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(plan_str[start_idx:i+1])
                    except json.JSONDecodeError:
                        pass
                    break

    # 如果是EXPLAIN输出，尝试解析
    # 这里可以添加更多的解析逻辑

    raise ValueError(f"无法解析计划字符串: {plan_str[:100]}...")


if __name__ == "__main__":
    # 测试代码
    import os
    from postgresql import get_sql_base_explain_plan

    # 创建搜索器
    searcher = TreeSimilaritySearcher(
        demo_fold=os.path.join(ROOT_DIR, "outputs", "demo_pool"),
        model_path=os.path.join(ROOT_DIR, "models", "tree_similarity", "model.pt"),
        device='cpu',
    )

    # 构建池（首次运行时）
    pool_path = os.path.join(ROOT_DIR, "outputs", "plan_tree_pool.pkl")
    if not os.path.exists(pool_path):
        searcher.build_pool(save_path=pool_path)
    else:
        searcher.load_pool(pool_path)

    # 测试搜索
    test_sql_path = os.path.join(ROOT_DIR, "data", "test-query", "17b.sql")
    if os.path.exists(test_sql_path):
        with open(test_sql_path, "r") as f:
            sql = f.read()

        # 获取执行计划
        try:
            # get_sql_base_explain_plan 返回 (plan_text, plan_json)
            initial_plan_text, plan_json = get_sql_base_explain_plan(sql)

            # 搜索相似的演示
            import time
            start_time = time.perf_counter()
            result = searcher.search(plan_json, top_k=1)
            end_time = time.perf_counter()

            (sql_demo, original_plan, suggest_hint, hinted_hint,
             exec_plan, infer_reason, best_global, best_leading, best_node) = result

            print(f"搜索结果:")
            print(f"  Demo SQL: {sql_demo[:100] if sql_demo else 'N/A'}...")
            print(f"  Suggested Hint: {suggest_hint if suggest_hint else 'N/A'}")
            print(f"  Hinted Hint (from subfolder): {hinted_hint}")
            print(f"  Best Global: {best_global if best_global else 'N/A'}")
            print(f"  Best Leading: {best_leading if best_leading else 'N/A'}")
            print(f"  Best Node: {best_node if best_node else 'N/A'}")
            print(f"  Time: {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"搜索出错: {e}")
            import traceback
            traceback.print_exc()
