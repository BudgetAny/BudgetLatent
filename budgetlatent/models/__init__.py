"""The networks and their weights.

  flux2_trunk     the FLUX.2 [klein] MMDiT, and the rotary / attention helpers both trunks share
  flux3_trunk     the FLUX 3 Image Klein DiT
  added_modules   the modules BudgetPix adds: cascade mixers and refiners, the 2-D integrated position encoding
  budgetpix       the adaptive-token model on either trunk (BudgetPixFlux2, BudgetPixFlux3)
  families        the four families (FLUX.2 [klein] 4B Base, FLUX.2 [klein] 4B, FLUX 3 Image Klein Base, FLUX 3 Image
                  Klein): released repositories, classes, sampling defaults
  weights         snapshots, checkpoints, load_model / load_teacher / save_weights
  convert         key maps between weight formats
"""
