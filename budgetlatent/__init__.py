"""BudgetLatent: text-to-image latent diffusion at any token budget, on FLUX.2 [klein] 4B Base ("flux2"), its
step-distilled release FLUX.2 [klein] 4B ("flux2_distilled"), FLUX 3 Image Klein Base ("flux3") and its step-distilled
release FLUX 3 Image Klein ("flux3_distilled", a flow map); with reference images (editing, multi-reference
generation) at any budget too, the references kept dense.

The package layout, one subpackage per role:

    tokenization/  layout                     the schedules, detail pyramids, threshold and exact-budget layouts, the
                                              training mixture
    models/        flux2_trunk, flux3_trunk   the two released trunks (FLUX.2 [klein] MMDiT, FLUX 3 Image Klein DiT)
                   added_modules, budgetpix   the modules BudgetPix adds and the adaptive-token model on either trunk
                   families, weights          the family registry; snapshots, checkpoints, loading and saving
                   convert                    key maps between weight formats
                   refs                       reference images (editing, multi-reference): files, latents, positions
    encoders/      text, vae                  the frozen text encoders (Qwen3, Qwen3-VL) and the VAE
    inference/     sampling, fast, validate   the sampler (50 guided steps; FLUX.2 [klein] 4B: 4 unguided steps; FLUX 3
                                              Image Klein: 16 unguided flow-map steps), the speed settings, the fixed
                                              validation suite
    training/      data, stage1, trainlib,    the chunked data, a stage-1 step's preparation, what the trainers share,
                   dmd                        the optional DMD2 + GAN stage (FLUX.2)

The grid (32 x 32 latent cells for 512 x 512 px, 64 x 64 for 1024 x 1024) is read from the environment variable
``BP_GRID`` when ``budgetlatent.tokenization.layout`` is imported.
"""
__version__ = "2.0.0"
