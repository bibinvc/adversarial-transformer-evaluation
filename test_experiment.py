"""Invariant checks for the methods, data separation and interpretation safeguards."""
import unittest

import torch
import torch.nn.functional as F

from experiment import (Config, PREFIX, LENGTH, TinyTransformer, apply_trigger, bootstrap_mean,
                        dataset, detach_output, deterministic, embedding_attack, entropy,
                        freeze, js_divergence, metrics, objective, patched_forward, project,
                        random_matched, spectrum, trigger_search)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        deterministic(17)
        self.data = dataset(Config(train_size=24, fit_size=12, test_size=12))
        self.x, self.y = self.data['test']
        self.model = freeze(TinyTransformer(width=16, layers=1))

    def test_splits_have_disjoint_program_bodies(self):
        groups = [{tuple(row[PREFIX:].tolist()) for row in rows} for rows, _ in self.data.values()]
        for i in range(len(groups)):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])

    def test_synthetic_labels_depend_on_opcode_only(self):
        for rows, labels in self.data.values():
            self.assertTrue(torch.equal(rows[:, PREFIX + 1] - 9, labels))

    def test_dataset_is_deterministic(self):
        second = dataset(Config(train_size=24, fit_size=12, test_size=12))
        self.assertTrue(torch.equal(self.data['test'][0], second['test'][0]))

    def test_trigger_preserves_body_and_semantic_labels(self):
        attacked = apply_trigger(self.x, torch.tensor([1, 2, 3]))
        self.assertTrue(torch.equal(self.x[:, PREFIX:], attacked[:, PREFIX:]))
        self.assertTrue(torch.equal(attacked[:, PREFIX + 1] - 9, self.y))
        self.assertFalse(torch.equal(self.x[:, :PREFIX], attacked[:, :PREFIX]))

    def test_causal_attention_has_no_future_mass(self):
        out = self.model(self.x)
        mask = torch.ones(LENGTH, LENGTH, dtype=torch.bool).triu(1)
        self.assertEqual(float(out['attention'][:, :, mask].abs().max()), 0.0)
        self.assertTrue(torch.allclose(out['attention'].sum(-1), torch.ones_like(out['attention'].sum(-1))))

    def test_hook_captures_residual_shape_and_is_removed(self):
        self.assertEqual(tuple(self.model(self.x)['residual'].shape), (12, LENGTH, 16))
        self.assertEqual(len(self.model.blocks[-1]._forward_hooks), 0)

    def test_input_gradient_is_nonzero_without_weight_gradients(self):
        embedded = self.model.embedding(self.x).detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(embeddings=embedded)['logits'], self.y)
        gradient, = torch.autograd.grad(loss, embedded)
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))

    def test_custom_gradient_agrees_with_finite_difference(self):
        embedded = self.model.embedding(self.x[:1]).detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(embeddings=embedded)['logits'], self.y[:1])
        gradient, = torch.autograd.grad(loss, embedded)
        direction = torch.zeros_like(embedded)
        direction[0, -1, 0] = 0.001
        high = F.cross_entropy(self.model(embeddings=embedded.detach() + direction)['logits'], self.y[:1])
        low = F.cross_entropy(self.model(embeddings=embedded.detach() - direction)['logits'], self.y[:1])
        numerical = float((high - low) / 0.002)
        self.assertAlmostEqual(float(gradient[0, -1, 0]), numerical, delta=0.003)

    def test_zero_budget_fgsm_is_identity(self):
        delta = embedding_attack(self.model, self.x, self.y, 0, 1)
        self.assertEqual(float(delta.abs().sum()), 0.0)

    def test_fgsm_stays_in_prefix_and_budget(self):
        delta = embedding_attack(self.model, self.x, self.y, 0.2, 1)
        self.assertLessEqual(float(delta.abs().max()), 0.200001)
        self.assertEqual(float(delta[:, PREFIX:].abs().sum()), 0.0)
        self.assertGreater(float(delta[:, :PREFIX].abs().sum()), 0.0)

    def test_pgd_stays_in_prefix_and_budget(self):
        delta = embedding_attack(self.model, self.x, self.y, 0.2, 4)
        self.assertLessEqual(float(delta.abs().max()), 0.200001)
        self.assertEqual(float(delta[:, PREFIX:].abs().sum()), 0.0)

    def test_attack_does_not_modify_model_parameters(self):
        before = [p.clone() for p in self.model.parameters()]
        embedding_attack(self.model, self.x, self.y, 0.2, 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.model.parameters())))

    def test_state_and_combined_objectives_are_differentiable(self):
        clean = detach_output(self.model(self.x))
        for kind in ('state', 'combined'):
            embedded = self.model.embedding(self.x).detach().requires_grad_(True)
            loss = objective(self.model(embeddings=embedded), clean, self.y, kind)
            gradient, = torch.autograd.grad(loss, embedded)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_random_baseline_matches_attack_l2_norm(self):
        delta = embedding_attack(self.model, self.x, self.y, 0.2, 4)
        noise = random_matched(delta, 25)
        self.assertTrue(torch.allclose(delta.flatten(1).norm(dim=-1), noise.flatten(1).norm(dim=-1), atol=1e-6))
        self.assertLessEqual(float(noise.abs().max()), 0.200001)

    def test_projection_rejects_negative_budget(self):
        with self.assertRaises(ValueError):
            project(torch.ones(1, LENGTH, 16), -1)

    def test_js_identity_and_symmetry(self):
        p, q = torch.randn(5, 3), torch.randn(5, 3)
        self.assertTrue(torch.allclose(js_divergence(p, p), torch.zeros(5), atol=1e-7))
        self.assertTrue(torch.allclose(js_divergence(p, q), js_divergence(q, p), atol=1e-7))
        self.assertTrue((js_divergence(p, q) >= -1e-7).all())

    def test_uniform_attention_entropy(self):
        self.assertAlmostEqual(float(entropy(torch.ones(4) / 4)), 1.38629436, places=6)

    def test_identical_output_has_zero_drift(self):
        out = self.model(self.x)
        result = metrics(out, out, self.y)
        self.assertEqual(float(result['residual_relative_drift'].abs().max()), 0.0)
        self.assertLess(float(result['logit_js_nats'].abs().max()), 1e-7)

    def test_spectrum_of_constant_residual_has_zero_effective_rank(self):
        result = spectrum(torch.ones(5, LENGTH, 16))
        self.assertEqual(result['effective_rank'], 0.0)

    def test_gcg_uses_allowed_tokens_and_monotonic_acceptance(self):
        rows, labels = self.data['attack_fit']
        result = trigger_search(self.model, rows, labels, steps=2, shortlist=2)
        self.assertTrue(all(0 <= t < 8 for t in result['trigger']))
        history = result['objective_history']
        self.assertTrue(all(b >= a for a, b in zip(history, history[1:])))

    def test_output_only_search_counts_example_queries(self):
        rows, labels = self.data['attack_fit']
        result = trigger_search(self.model, rows, labels, steps=1, kind='ce', gradient_guided=False)
        self.assertEqual(result['example_logit_queries'], result['batch_forward_queries'] * len(rows))
        self.assertFalse(result['gradient_guided'])

    def test_final_residual_patch_restores_classifier_logits(self):
        clean = self.model(self.x)
        attacked = apply_trigger(self.x, torch.tensor([4, 5, 6]))
        patched = patched_forward(self.model, attacked, replacement=clean['residual'][:, -1])
        self.assertTrue(torch.allclose(clean['logits'], patched['logits'], atol=1e-7))
        self.assertEqual(len(self.model.blocks[-1]._forward_hooks), 0)

    def test_bootstrap_is_paired_and_deterministic(self):
        zero = bootstrap_mean(torch.zeros(12), 200, 13)
        self.assertEqual(zero, {'mean': 0.0, 'ci95': [0.0, 0.0]})
        self.assertEqual(bootstrap_mean(torch.tensor([-1., 0., 1.]), 200, 13),
                         bootstrap_mean(torch.tensor([-1., 0., 1.]), 200, 13))

    def test_same_seed_reproduces_pgd(self):
        a = embedding_attack(self.model, self.x, self.y, 0.2, 4, seed=12)
        b = embedding_attack(self.model, self.x, self.y, 0.2, 4, seed=12)
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main(verbosity=2)
