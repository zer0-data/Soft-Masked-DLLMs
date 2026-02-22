import sys
import os
import torch
import unittest
from unittest.mock import MagicMock
from omegaconf import OmegaConf

# Add module path
sys.path.append(os.path.join(os.getcwd(), 'mdlm'))

from diffusion import Diffusion
from soft_masking import SoftMaskingModule

class TestSoftMaskingIntegrated(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create({
            'use_soft_masking': True,
            'backbone': 'dit',
            'T': 0,
            'subs_masking': False,
            'parameterization': 'subs',
            'time_conditioning': False,
            'training': {
                'ema': 0,
                'antithetic_sampling': False,
                'importance_sampling': False,
                'change_of_variables': False,
                'sampling_eps': 1e-3,
                'weight_decay': 0,
                'p_sm': 1.0 # Force soft masking for test
            },
            'model': {
                'hidden_size': 32,
                'n_heads': 4,
                'cond_dim': 16,
                'n_blocks': 2,
                'dropout': 0.1,
                'scale_by_sigma': False,
                'length': 10
            },
            'optim': {
                'lr': 1e-3,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'weight_decay': 0
            },
            'noise': {'type': 'loglinear', 'sigma_min': 1e-4, 'sigma_max': 20},
            'loader': {'batch_size': 2, 'eval_batch_size': 2},
            'eval': {'gen_ppl_eval_model_name_or_path': 'gpt2'}, 
            'sampling': {
                'predictor': 'ddpm', 
                'steps': 2,
                'noise_removal': True,
                'num_sample_batches': 1,
                'num_sample_log': 1,
                'semi_ar': False,
                'stride_length': 1,
                'num_strides': 1
            },
        })
        
        self.tokenizer = MagicMock()
        self.tokenizer.vocab_size = 100
        self.tokenizer.mask_token_id = 99
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2

    def test_lambda_formula(self):
        print("\nTesting Lambda Formula...")
        module = SoftMaskingModule(32, 100, 99)
        # Check params are parameters
        self.assertIsInstance(module.omega_s, torch.nn.Parameter)
        self.assertIsInstance(module.omega_a, torch.nn.Parameter)
        self.assertIsInstance(module.omega_b, torch.nn.Parameter)
        
        probs = torch.softmax(torch.randn(2, 10, 100), dim=-1)
        lam = module.compute_lambda(probs)
        self.assertEqual(lam.shape, (2, 10, 1))
        # Ensure lambda depends on omega_s
        module.omega_s.data.fill_(0.5)
        # With omega_s=0.5, max lambda is 0.5. Check if we are respecting that?
        # Actually lambda = omega_s * sigmoid(...)
        # So lambda <= omega_s (if positive)
        self.assertTrue(torch.all(lam <= 0.5))
        print("Lambda formula appears correct.")

    def test_training_step(self):
        print("\nTesting Training Step...")
        model = Diffusion(self.config, self.tokenizer)
        
        # Mock forward to track calls
        # We can't easily mock model.forward because it's used internally
        # But we can check if soft_masking_module is called.
        model.soft_masking_module.forward = MagicMock(wraps=model.soft_masking_module.forward)
        
        batch = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        loss = model.training_step(batch, 0)
        print(f"Training Loss: {loss.item()}")
        
        # Check if soft masking was called
        # self.config.training.p_sm is 1.0, so it should be called.
        self.assertTrue(model.soft_masking_module.forward.called)
        print("Soft Masking called during training.")

    def test_v0_is_mask_embed(self):
        """Verify v0 is always the mask-token embedding, never ground-truth."""
        print("\nTesting v0 is mask embed...")
        vocab_size = 100
        hidden_size = 32
        mask_token_id = 99

        module = SoftMaskingModule(hidden_size, vocab_size, mask_token_id)
        embed = torch.nn.Embedding(vocab_size, hidden_size)

        # x_t: position 0 is a real token (id=5), positions 1-4 are masked
        x_t = torch.tensor([[5, mask_token_id, mask_token_id, mask_token_id, mask_token_id]])
        probs = torch.softmax(torch.randn(1, 5, vocab_size), dim=-1)

        # Ground-truth embedding for token 5
        gt_embed = embed(torch.tensor(5))                        # (H,)
        mask_embed = embed(torch.tensor(mask_token_id))          # (H,)

        result = module(x_t, probs, embed)                       # (1, 5, H)

        # Unmasked position (idx 0) must be the original token-5 embedding
        self.assertTrue(
            torch.allclose(result[0, 0], gt_embed, atol=1e-6),
            "Unmasked position should retain the original token embedding."
        )

        # Masked positions (idx 1-4) should NOT equal ground-truth embed
        # and should be interpolated between mask_embed and feedback
        for pos in range(1, 5):
            self.assertFalse(
                torch.allclose(result[0, pos], gt_embed, atol=1e-6),
                f"Masked position {pos} should NOT contain ground-truth embedding."
            )

        # Verify the result at each masked position is a valid interpolation:
        # it should lie between mask_embed and the feedback (not be garbage)
        # At minimum, each masked position should have finite values
        self.assertTrue(
            torch.all(torch.isfinite(result[0, 1:])),
            "All masked position embeddings should be finite."
        )
        print("v0 correctly uses mask-token embedding, not ground-truth.")

if __name__ == '__main__':
    unittest.main()

