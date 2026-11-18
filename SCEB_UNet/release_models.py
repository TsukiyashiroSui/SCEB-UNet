"""Model construction for the manuscript's matched-supervision controls."""

import torch

from model_v2 import ACC_UNet


class PlainConcat(torch.nn.Module):
    def __init__(self, fp32_output):
        super().__init__()
        self.fp32_output = fp32_output

    def forward(self, decoder, skip, coarse_logits, edge_logits):
        fused = torch.cat((decoder, skip), dim=1)
        if self.fp32_output:
            fused = fused.float()
        return fused, torch.zeros_like(skip[:, :1])


def build_model(variant="", **kwargs):
    model = ACC_UNet(**kwargs)
    if variant in {"concat_matched_aux", "backbone_aux"}:
        # Keep the auxiliary heads and their initialization. Replace the gates
        # after constructing the parent, as in the formal ClinicDB controls.
        fp32_output = variant == "backbone_aux"
        model.ugbf_112 = PlainConcat(fp32_output)
        model.ugbf_224 = PlainConcat(fp32_output)
    return model
