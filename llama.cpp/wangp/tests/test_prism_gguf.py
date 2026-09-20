"""Prism GGUF parsing, transforms and compiler contracts (small local fixtures)."""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shared.qtypes import gguf as handler
from shared.qtypes.prism import PrismEmbedding, PrismLinear, PrismFirstRowsLinear, PrismHadamard, hadamard_reference, _hadamard_cuda, ptq_linear, ptq_embedding


class PrismGGUFTests(unittest.TestCase):
    def test_quantization_selection_and_download_contract(self):
        from shared.prompt_enhancer.qwen35_vl import get_qwen35_quantization, ensure_qwen35_prompt_enhancer_assets
        self.assertEqual(get_qwen35_quantization('gguf_ptq1', variant='27b'), 'gguf_ptq1')
        with self.assertRaises(ValueError):
            get_qwen35_quantization('gguf_ptq1', variant='9b')
        downloads = []
        ensure_qwen35_prompt_enhancer_assets(lambda **kw: downloads.append(kw), backend='gguf_ptq1', variant='27b', speculative_decoding=True)
        files = [name for request in downloads for group in request['fileList'] for name in group]
        self.assertIn('Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf', files)
        self.assertIn('Ternary-Bonsai-2-27B-MTP-Q8_0.gguf', files)
        downloads.clear()
        ensure_qwen35_prompt_enhancer_assets(lambda **kw: downloads.append(kw), backend='gguf_ptq1', variant='27b', speculative_decoding=False)
        target_only_files = [name for request in downloads for group in request['fileList'] for name in group]
        self.assertFalse(any('MTP-' in name for name in target_only_files))
        self.assertNotIn('Qwen3.8-27B-Uncensored-Q4_K_M.gguf', files)

    def test_private_type_and_stream_parser(self):
        def string(s):
            value = s.encode()
            return struct.pack('<Q', len(value)) + value
        # A minimal, real GGUF file with an opaque Prism extension field.
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
        header += string('prism.hadamard.version') + struct.pack('<II', 4, 1)
        header += string('output.weight') + struct.pack('<IQQIQ', 2, 128, 1, 143, 0)
        header += b'\0' * (-len(header) % 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.gguf'
            path.write_bytes(header + b'\0' * 28)
            parsed = handler._gguf_get_index(path)
            self.assertEqual(parsed.tensor_infos[0].tensor_type.name, 'PTQ1_0')
            self.assertEqual(parsed.tensor_infos[0].n_bytes, 28)
            self.assertEqual(dict(parsed.prism_metadata)['prism.hadamard.version'], 1)
        self.assertEqual(handler._quantization_sizes(handler.gguf.GGMLQuantizationType.Q4_K), (256, 144))

    def test_ptq_reference_decoder_all_trits(self):
        rng = np.random.default_rng(142)
        trits = rng.integers(0, 3, (32, 128), dtype=np.uint16)
        raw = np.zeros((32, 28), dtype=np.uint8)
        for start, offset, width, count in ((0, 0, 16, 5), (80, 16, 8, 5), (120, 24, 2, 4)):
            q = np.zeros((32, width), dtype=np.uint16)
            for n in range(count):
                q = q * 3 + trits[:, start + n*width:start + (n+1)*width]
            if count == 4:
                q *= 3
            raw[:, offset:offset+width] = (q*256+242)//243
        raw[:, 26:28] = np.array([.25], dtype='<f2').view(np.uint8)
        actual = handler._gguf_dequantize_tensor(torch.from_numpy(raw), handler.PrismQuantizationType.PTQ1_0, (32, 128), torch.float32)
        torch.testing.assert_close(actual, torch.from_numpy((trits.astype(np.float32)-1)*.25), rtol=0, atol=0)

    def test_hadamard_inverse(self):
        x = torch.randn(3, 5120, device='cpu')
        signs = torch.randint(0, 2, (5120,), dtype=torch.int8, device='cpu')*2-1
        transformed = hadamard_reference(x, signs)
        torch.testing.assert_close(hadamard_reference(transformed, signs, inverse=True), x, rtol=1e-5, atol=1e-6)

    def test_fake_cuda_ops(self):
        with FakeTensorMode():
            x = torch.empty(3, 5120, device='cuda', dtype=torch.bfloat16)
            signs = torch.empty(5120, device='cuda', dtype=torch.int8)
            raw = torch.empty(128, 1120, device='cuda', dtype=torch.uint8)
            indices = torch.empty(3, 2, device='cuda', dtype=torch.int64)
            self.assertEqual(_hadamard_cuda(x, signs, False, [0,0,0]).shape, x.shape)
            self.assertEqual(ptq_linear(x, raw, [128,5120], None, x.dtype).shape, (3,128))
            self.assertEqual(ptq_embedding(indices, raw, [128,5120], x.dtype).shape, (3,2,5120))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_resident_prism_embedding_moves_cpu_indices_to_loaded_weight(self):
        raw = torch.zeros((2, 224), dtype=torch.uint8, device="cuda")
        weight = handler.GGUFWeightTensor.create(
            raw,
            (2, 1024),
            (1024, 1),
            torch.bfloat16,
            tensor_type=handler.PrismQuantizationType.PTQ1_0,
            tensor_shape=(2, 1024),
        )
        embedding = PrismEmbedding(2, 1024, device="meta", dtype=torch.bfloat16, weights=handler._GGUF_QTYPE)
        embedding.weight = torch.nn.Parameter(weight, requires_grad=False)
        embedding.prism_transform = PrismHadamard(torch.ones(1024, dtype=torch.int8, device="cuda"), inverse=True)
        output = embedding(torch.tensor([[0, 1]], dtype=torch.int64, device="cpu"))
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(output.shape, (1, 2, 1024))

    def test_non_prism_embedding_alias_is_unchanged(self):
        from types import SimpleNamespace
        from shared.prompt_enhancer.qwen35_vl import alias_qwen35_text_embedding_for_mmgp

        source = torch.nn.Embedding(3, 4)
        alias = alias_qwen35_text_embedding_for_mmgp(SimpleNamespace(token_embd=source))
        self.assertIs(type(alias), torch.nn.Embedding)
        self.assertFalse(hasattr(alias, "prism_transform"))
        self.assertIs(alias.weight, source.weight)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_draft_vocabulary_view_preserves_rotation_and_graph_replay(self):
        rng = np.random.default_rng(41)
        raw = rng.integers(0, 256, (8, 8, 28), dtype=np.uint8)
        raw[..., 26:] = np.array([.125], dtype='<f2').view(np.uint8)
        weight = handler.GGUFWeightTensor.create(torch.from_numpy(raw.reshape(8, -1)).cuda(), (8, 1024), (1024, 1), torch.bfloat16,
                                               tensor_type=handler.PrismQuantizationType.PTQ1_0, tensor_shape=(8, 1024))
        source = PrismLinear(1024, 8, bias=False, device='meta', dtype=torch.bfloat16, weights=handler._GGUF_QTYPE)
        source.weight = torch.nn.Parameter(weight, requires_grad=False)
        source.prism_transform = PrismHadamard(torch.randint(0, 2, (1024,), device='cuda', dtype=torch.int8) * 2 - 1)
        draft = PrismFirstRowsLinear(source, 5)
        x = torch.randn(1, 1024, device='cuda', dtype=torch.bfloat16)
        expected = source(x)[:, :5]
        torch.testing.assert_close(draft(x), expected, rtol=0, atol=0)
        self.assertIs(draft.weight, source.weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = draft(x)
        x.normal_()
        graph.replay()
        torch.testing.assert_close(result, source(x)[:, :5], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_fused_linear_attention_verification_matches_separate_projections(self):
        import copy
        from types import SimpleNamespace
        from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5Block
        from shared.llm_engines.nanovllm.utils.context import set_context, reset_context
        from shared.prompt_enhancer.qwen35_text import _apply_qwen35_projection_fusions

        config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                                 linear_num_key_heads=1, linear_num_value_heads=2,
                                 linear_key_head_dim=64, linear_value_head_dim=64,
                                 layer_types=['linear_attention'])
        torch.manual_seed(137)
        with torch.device('cuda'):
            reference = Qwen3_5Block(config, 0).to(dtype=torch.bfloat16)
        optimized = copy.deepcopy(reference)
        _apply_qwen35_projection_fusions(SimpleNamespace(blk=[optimized], mtp=None), prism=True)
        x = torch.randn(1, 3, 128, device='cuda', dtype=torch.bfloat16) * .1
        conv = torch.randn(1, reference.key_dim * 2 + reference.value_dim, 4, device='cuda', dtype=torch.bfloat16) * .1
        recurrent = torch.randn(1, 2, 64, 64, device='cuda', dtype=torch.bfloat16) * .1
        results = []
        try:
            for block in (reference, optimized):
                block.prepare_sequence_state(1, x.device, x.dtype)
                block.prepare_speculative_state(3)
                block.conv_state_buffer.copy_(conv)
                block.recurrent_state_buffer.copy_(recurrent)
                set_context(False, has_previous_state=True, speculative_verify=True)
                results.append(block._forward_linear_attention([x.clone()], 0, None, None))
            torch.testing.assert_close(results[1], results[0], atol=.002, rtol=.02)
            torch.testing.assert_close(optimized.conv_state_buffer, reference.conv_state_buffer, atol=.002, rtol=.02)
            torch.testing.assert_close(optimized.speculative_conv_state_buffer, reference.speculative_conv_state_buffer, atol=.002, rtol=.02)
            torch.testing.assert_close(optimized.recurrent_state_buffer, reference.recurrent_state_buffer, atol=.002, rtol=.02)
        finally:
            reset_context()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_fused_full_attention_preserves_q_gate_k_v_order(self):
        import copy
        from types import SimpleNamespace
        from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5Block, Qwen3_5DynamicCache, Qwen3_5TextRotaryEmbedding
        from shared.prompt_enhancer.qwen35_text import _apply_qwen35_projection_fusions

        config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                                 layer_types=['full_attention'])
        torch.manual_seed(138)
        with torch.device('cuda'):
            reference = Qwen3_5Block(config, 0).to(dtype=torch.bfloat16)
            rotary = Qwen3_5TextRotaryEmbedding(config)
        optimized = copy.deepcopy(reference)
        _apply_qwen35_projection_fusions(SimpleNamespace(blk=[optimized], mtp=None), prism=True)
        for tokens in (1, 3, 17):
            x = torch.randn(1, tokens, 128, device='cuda', dtype=torch.bfloat16) * .1
            rope = rotary(x, torch.arange(tokens, device='cuda').unsqueeze(0))
            expected = reference._forward_full_attention([x.clone()], rope, 0, Qwen3_5DynamicCache(config))
            actual = optimized._forward_full_attention([x.clone()], rope, 0, Qwen3_5DynamicCache(config))
            torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_grouped_gdn_load_layout_preserves_decode_and_verify_state(self):
        import copy
        from types import SimpleNamespace
        from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5Block, _reorder_v_heads_tiled_to_grouped
        from shared.llm_engines.nanovllm.utils.context import set_context, reset_context
        from shared.prompt_enhancer.qwen35_text import _apply_qwen35_projection_fusions
        from shared.qtypes.prism import prepare_prism_gdn_layout

        config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                                 linear_num_key_heads=2, linear_num_value_heads=4,
                                 linear_key_head_dim=32, linear_value_head_dim=32,
                                 layer_types=['linear_attention'])
        torch.manual_seed(139)
        with torch.device('cpu'):
            reference = Qwen3_5Block(config, 0).to(dtype=torch.bfloat16)
        reference._gguf_v_head_reordered = reference._gguf_ssm_param_reordered = True
        _apply_qwen35_projection_fusions(SimpleNamespace(blk=[reference], mtp=None), prism=True)
        optimized = copy.deepcopy(reference)
        prepare_prism_gdn_layout(SimpleNamespace(blk=[optimized]), {'prism.hadamard.gdn_v_grouped': True})
        self.assertFalse(optimized._gguf_v_head_reordered)
        self.assertEqual([p.dtype for p in reference.parameters()], [p.dtype for p in optimized.parameters()])
        # Isolate the layout conversion: the real Prism output transform also
        # turns tiled heads back into grouped heads before its Hadamard.
        hook = reference.ssm_out.register_forward_pre_hook(lambda module, args: (
            _reorder_v_heads_tiled_to_grouped(args[0], -1, 2, 4, 32),))
        reference.cuda()
        optimized.cuda()
        heads = torch.tensor([0, 2, 1, 3], device='cuda')
        values = (heads[:, None] * 32 + torch.arange(32, device='cuda')).reshape(-1)
        conv_rows = torch.cat((torch.arange(128, device='cuda'), 128 + values))
        try:
            for tokens in (1, 3, 17):
                x = torch.randn(1, tokens, 128, device='cuda', dtype=torch.bfloat16) * .1
                conv = torch.randn(1, 256, 4, device='cuda', dtype=torch.bfloat16) * .1
                recurrent = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.bfloat16) * .1
                outputs = []
                for block in (reference, optimized):
                    block.prepare_sequence_state(1, x.device, x.dtype)
                    block.prepare_speculative_state(tokens)
                    block.conv_state_buffer.copy_(conv if block is reference else conv.index_select(1, conv_rows))
                    block.recurrent_state_buffer.copy_(recurrent)
                    set_context(False, has_previous_state=True, speculative_verify=tokens > 1)
                    outputs.append(block._forward_linear_attention([x.clone()], 0, None, None))
                torch.testing.assert_close(outputs[1], outputs[0], atol=.002, rtol=.02)
                torch.testing.assert_close(optimized.conv_state_buffer, reference.conv_state_buffer.index_select(1, conv_rows), atol=.002, rtol=.02)
                torch.testing.assert_close(optimized.recurrent_state_buffer, reference.recurrent_state_buffer, atol=.002, rtol=.02)
                if tokens > 1:
                    torch.testing.assert_close(optimized.speculative_conv_state_buffer, reference.speculative_conv_state_buffer.index_select(2, conv_rows), atol=.002, rtol=.02)
                    torch.testing.assert_close(optimized.speculative_recurrent_state_buffer, reference.speculative_recurrent_state_buffer, atol=.002, rtol=.02)
        finally:
            hook.remove()
            reset_context()


if __name__ == '__main__':
    unittest.main()
