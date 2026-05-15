import torch
from collections import defaultdict
from .constants import DIM_SINK, MODEL_LLM
import copy
from src.stash import MetadataStation, ValueMonitor


class DefaultVars:
    # Model Config
    llm_name = "llama-v2-7b"

    # IMPORTANT:
    # Default should be False.
    # Otherwise baseline mode logic=0 may still trigger HeadFork / VAR logic.
    logic_flag = False

    # DimProspector
    indices = defaultdict(list)
    dim_sink = DIM_SINK[llm_name]

    # Layer selection default.
    # This avoids AttributeError when baseline mode has not called activate().
    sink_select_layers = []

    # HeadFork
    forked_head = {}
    forked_head_per_token = defaultdict(dict)


class LogicEngine:
    defaultVars = DefaultVars()

    logic_flag = defaultVars.logic_flag
    llm_name = defaultVars.llm_name
    indices = defaultVars.indices
    forked_head = defaultVars.forked_head
    forked_head_per_token = defaultVars.forked_head_per_token
    dim_sink = defaultVars.dim_sink
    sink_select_layers = defaultVars.sink_select_layers

    @classmethod
    def activate(
        cls,
        tau=20,
        rho=0.5,
        summ=0.2,
        p=0.6,
        except_last_layer=True,
        layer="all",
    ):
        cls.set_flag(True)
        cls.tau = tau
        cls.rho = rho
        cls.summ = summ
        cls.p = p
        cls.except_last_layer = except_last_layer
        cls.set_sink_select_layers(layer)

    @classmethod
    def set_sink_select_layers(cls, layer="all"):
        if layer == "all":
            cls.sink_select_layers = [
                i for i in range(MetadataStation.model_config["num_hidden_layers"])
            ][2:]
        else:
            assert isinstance(layer, int)
            cls.sink_select_layers = layer

    @classmethod
    def set_llm_name(cls, model_name):
        for body, llm in MODEL_LLM.items():
            if model_name in body:
                cls.llm_name = llm
                break

    @classmethod
    def set_flag(cls, flag_value=True):
        cls.logic_flag = flag_value

    @classmethod
    def _flag(cls, name=None):
        return bool(cls.logic_flag)

    @classmethod
    def run_logic(cls):
        raise NotImplementedError("Running basic logic in LogicEngine.")

    @classmethod
    def clear(cls):
        """
        Clear per-sample cached states.

        Do NOT turn off logic_flag here.
        If VAR is activated for this run, it should stay active across samples.
        If baseline logic=0, logic_flag remains False.
        """
        _defaultVars = DefaultVars()

        cls.llm_name = _defaultVars.llm_name
        cls.dim_sink = _defaultVars.dim_sink

        # Keep current logic_flag.
        # baseline: False
        # VAR run: True
        cls.logic_flag = cls.logic_flag

        cls.indices = defaultdict(list)
        cls.forked_head = {}
        cls.forked_head_per_token = defaultdict(dict)

        # Only reset layer selection in baseline mode.
        # In VAR mode, keep the selected layers set by activate().
        if not cls.logic_flag:
            cls.sink_select_layers = []


class DimProspector(LogicEngine):
    @classmethod
    def fix_dim(cls, llm_name="llama-7b"):
        print(f"{cls.__name__} fixed dims: {cls.dim_sink[llm_name]}")

    @classmethod
    def rmsnorm(cls, hidden_states, eps=1e-6):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + eps)

    @classmethod
    def run_logic(cls, hs, layer):
        if not cls.__base__._flag():
            return

        rms_norm_hs = torch.abs(cls.rmsnorm(hs))  # [bsz, tok, dim]

        rms_values = torch.stack(
            [rms_norm_hs[:, :, idx] for idx in cls.__base__.dim_sink],
            dim=-1,
        )  # [bsz, tok, num_sink_dims]

        max_rms_values = torch.max(rms_values, dim=-1)[0]  # [bsz, tok]

        indices = torch.nonzero(max_rms_values > cls.tau)[:, 1]
        cls.__base__.indices[layer] = indices


class HeadFork(LogicEngine):
    @classmethod
    def run_logic(cls, attn, layer_idx):
        if not cls.__base__._flag():
            return

        layer = layer_idx

        sink_select_layers = getattr(
            cls,
            "sink_select_layers",
            getattr(cls.__base__, "sink_select_layers", []),
        )

        if isinstance(sink_select_layers, list):
            sink_inds = cls.indices[layer]
        elif sink_select_layers is None:
            cls.__base__.forked_head[layer] = []
            cls.__base__.forked_head_per_token[
                ValueMonitor.get_output_token_count()
            ][layer] = []
            return
        else:
            sink_inds = cls.indices[sink_select_layers]

        if len(sink_inds) == 0:
            cls.__base__.forked_head[layer] = []
            cls.__base__.forked_head_per_token[
                ValueMonitor.get_output_token_count()
            ][layer] = []
            return

        im = MetadataStation.segments["begin_pos"]["image"]
        pa = MetadataStation.metadata["vis_len"]

        vis_sink_inds = [
            i.unsqueeze(0)
            for i in sink_inds
            if im <= int(i.item()) < im + pa
        ]

        if len(vis_sink_inds) > 0:
            vis_sink_inds = torch.cat(vis_sink_inds, dim=0)

            image_attn = attn[:, :, :, im : im + pa]

            portion = torch.sum(
                image_attn[:, :, :, vis_sink_inds - im],
                dim=-1,
            ) / torch.sum(image_attn + 1e-6, dim=-1)

            summation = torch.sum(image_attn, dim=-1)

            # Condition 1. Portion <= rho
            portion_condition = portion <= cls.rho

            # Condition 2. Summation >= summ
            summation_condition = summation >= cls.summ

            candidate_coords = torch.nonzero(
                portion_condition & summation_condition
            )

            cls.__base__.forked_head[layer] = candidate_coords.clone()

        else:
            cls.__base__.forked_head[layer] = []

        cls.__base__.forked_head_per_token[
            ValueMonitor.get_output_token_count()
        ][layer] = cls.__base__.forked_head[layer]

        return


class VARProcessor(LogicEngine):
    except_last_layer = False

    @classmethod
    def config_last_layer(cls, flag):
        cls.except_last_layer = flag

    @classmethod
    def set_selected_token(cls, selected_tokens):
        cls.selected_tokens = selected_tokens

    @classmethod
    def check_target_layer(cls):
        return cls.TARGET_LAYERS

    @classmethod
    def attn_redist(cls, attention_map, layer_idx):
        """
        attention_map:
            Usually shape [bsz, heads, query_len, key_len].

        VAR redistributes attention mass from sink tokens to non-sink visual tokens.
        """
        if not cls.__base__._flag():
            return attention_map

        p = cls.p

        current_decoder_layer = getattr(cls, "current_decoder_layer", layer_idx)
        model_config = getattr(cls, "model_config", None)

        if (
            cls.except_last_layer
            and model_config is not None
            and current_decoder_layer == model_config.num_hidden_layers - 1
        ):
            return attention_map

        im = MetadataStation.segments["begin_pos"]["image"]
        pa = MetadataStation.metadata["vis_len"]

        coord = HeadFork.forked_head.get(layer_idx, [])
        indices = cls.__base__.indices[layer_idx]

        if len(coord) == 0 or len(indices) == 0:
            return attention_map

        model_head_num = MetadataStation.model_config["num_attention_heads"]

        for h in range(model_head_num):
            query_coord = coord[coord[:, 1] == h][:, 2]

            if ValueMonitor.get_output_token_count() < 0:
                query_coord = query_coord[im + pa <= query_coord]

            bsz_coord = coord[coord[:, 1] == h][:, 0][: len(query_coord)]
            head_coord = coord[coord[:, 1] == h][:, 1][: len(query_coord)]

            if not query_coord.shape[0] or not head_coord.shape[0]:
                continue

            selected_attn_map = attention_map[
                bsz_coord,
                head_coord,
                query_coord,
                :,
            ].clone()

            indices = indices.to(selected_attn_map.device)

            vis_indices = indices[(im <= indices) & (indices < im + pa)]
            text_indices = indices[~torch.isin(indices, vis_indices)]

            if len(vis_indices) == 0 and len(text_indices) == 0:
                continue

            copied_attention_map = copy.deepcopy(selected_attn_map.detach())

            # Decrease sink-token attention by p.
            if len(text_indices) > 0:
                selected_attn_map[:, text_indices] *= p

            if len(vis_indices) > 0:
                selected_attn_map[:, vis_indices] *= p

            # Attention budget from sink tokens.
            weight_budget_vis = (
                copied_attention_map[:, vis_indices].sum(dim=1) * (1 - p)
                if len(vis_indices) > 0
                else torch.zeros(
                    selected_attn_map.shape[0],
                    device=selected_attn_map.device,
                    dtype=selected_attn_map.dtype,
                )
            )

            weight_budget_text = (
                copied_attention_map[:, text_indices].sum(dim=1) * (1 - p)
                if len(text_indices) > 0
                else torch.zeros(
                    selected_attn_map.shape[0],
                    device=selected_attn_map.device,
                    dtype=selected_attn_map.dtype,
                )
            )

            # Remove sink visual tokens before computing ratios.
            if len(vis_indices) > 0:
                copied_attention_map[:, vis_indices] *= 0

            visual_region = copied_attention_map[:, im : im + pa]
            denom = visual_region.sum(dim=1, keepdim=True)

            # Avoid division by zero.
            safe_denom = denom.clamp_min(1e-6).to(selected_attn_map.dtype)

            ratios_vis = visual_region / safe_denom

            selected_attn_map[:, im : im + pa] += (
                weight_budget_vis + weight_budget_text
            ).view(-1, 1) * ratios_vis

            attention_map[bsz_coord, head_coord, query_coord, :] = selected_attn_map

        return attention_map
