# Dynamics-Aligned Latent Imagination (DALI)

#### [[OpenReview]](https://openreview.net/forum?id=41bIzD5sit) [[arXiv]](https://arxiv.org/abs/2508.20294) [[Code]](https://github.com/frankroeder/DALI)

[Frank Röder*](https://frankroeder.github.io/), [Jan Benad](https://scholar.google.com/citations?user=FWy1Ov0AAAAJ), [Manfred Eppe](https://scholar.google.de/citations?user=fG0VWroAAAAJ), [Pradeep Kr. Banerjee*](https://scholar.google.com/citations?user=cnSjMBwAAAAJ) (&#42; equal contribution)<br/>
All the listed authors are members of the [Institute for Data Science Foundations](https://www.tuhh.de/dsf/homepage).

---

This is the official implementation of the DALI approach using JAX.

<p align="center">
<img src="https://github.com/user-attachments/assets/0f2ae302-4b55-40ea-98f9-c99b556b3cff">
<img src="https://github.com/user-attachments/assets/2285bb58-6943-462d-ac26-4ceeaa3ccba1">
</p>


```
@article{Roder_DynamicsAlignedLatent_2025,
  title = {Dynamics-{{Aligned Latent Imagination}} in {{Contextual World Models}} for {{Zero-Shot Generalization}}},
  author = {R{\"o}der, Frank and Benad, Jan and Eppe, Manfred and Banerjee, Pradeep Kr},
  journal={arXiv preprint arXiv:2508.20294},
  year = {2025},
}
```

## Setup
Use `uv` to setup your python environment.

```bash
uv sync
uv pip install -e ./dreamerv3_compat
uv pip install -e ./
```

## Training

Run scripts in `./local_scripts/` to generate results for experts, random policies and DALI variants.

## Record Data for Anaylsis

```bash
uv run -m contextual_mbrl.dreamer.record_context --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337
```

### Log data for counterfactual dreams
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility --logdir logs/carl_dmc_ball_in_cup_double_box_enc_img_dec_img_ctxencoder_transformer_grssm_normalized/1337 --jax.platform cpu
```

### Log data for imagined counterfactual obs trajectories
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility_obs --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337 --jax.platform cpu
```

### Record dataset for counterfactual obs analysis
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility_obs_dataset --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337 --jax.platform cpu
```

## Plots

To generate the plots, run the scripts in the `./analysis` directory.
