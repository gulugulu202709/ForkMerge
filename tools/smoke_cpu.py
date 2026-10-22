"""Synthetic tensor interface check of the ordinary updater and both losses. No video model."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def smoke(output):
    if output.exists():
        raise ValueError("output already exists; choose a new directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from pipeline.config import LR, METHODS, TOKENS, WIDTH
    from pipeline.model import OrdinaryUpdater, paired_objective
    torch.set_num_threads(2)
    torch.manual_seed(20261022)
    source = torch.randn(3, 4, TOKENS, WIDTH, device="cpu")
    actions = torch.tensor([[0, 1, 2, 6], [0, 1, 2, 6], [0, 1, 2, 6]], device="cpu")
    target = torch.randn(3, 4, 2, 4, 4, device="cpu")
    masks = torch.ones(3, 2, 4, 4, device="cpu")
    masks[0, :, :, 2:] = 0
    masks[1, :, :, :2] = 0
    prototype = OrdinaryUpdater().cpu()
    initial = copy.deepcopy(prototype.state_dict())
    digest = hashlib.sha256()
    for key, value in sorted(initial.items()):
        digest.update(key.encode()); digest.update(value.numpy().tobytes())
    output.mkdir(parents=True)
    arms = {}
    for arm in METHODS:
        model = OrdinaryUpdater().cpu()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        before, stages = model(source, actions, trace=True)
        # Deliberately artificial readout to exercise the flow-loss tensor API.
        # This slice is not a renderer, generator, VAE, or semantic state probe.
        predictions = before[:, :4, :32].reshape(3, 4, 2, 4, 4)
        loss, terms = paired_objective(predictions, target, masks, arm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        with torch.no_grad():
            after, _ = model(source, actions, trace=True)
        delta = sum(float((value - initial[key]).square().sum()) for key, value in model.state_dict().items()) ** .5
        np.savez_compressed(output / (arm + "_memory.npz"), before=before.detach().numpy(),
                            after=after.numpy(), history_stages=torch.stack(stages).detach().numpy())
        arms[arm] = {"initial_sha256": digest.hexdigest(), "cpu_optimizer_steps": 1,
                     "loss_terms_interface_diagnostics_only": terms, "parameter_delta_l2": delta}
    result = {"status": "SYNTHETIC_CPU_INTERFACE_CHECK", "seed": 20261022,
              "inputs": "Seeded random tensors and fixed action IDs; no recorded videos or observed labels.",
              "readout": "Artificial tensor slice solely for API checking; not the frozen generator.",
              "memory_shape": list(before.shape), "arms": arms, "cuda_initialized": torch.cuda.is_initialized(),
              "video_generated": False, "method_efficacy_result": False, "gpu_device_seconds": 0,
              "interpretation": "Saved tensors are actual CPU outputs. Their changes do not establish semantic erasure/protection or superiority."}
    (output / "SMOKE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/cpu-smoke")
    args = parser.parse_args()
    try:
        print(json.dumps(smoke(args.output.resolve()), indent=2))
    except ValueError as exc:
        parser.exit(2, "CPU smoke refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
