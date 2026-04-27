# DWNP (Device-Wise Federated Network Pruning)

This module integrates DWNP into the distributed FedPruning workflow.

Core ideas implemented:

- A server-side subnet mask and a device-wise subnet mask are generated from a server code and a client id embedding.
- Local training uses the joint mask, i.e. the intersection of the server and device masks.
- Device-wise masks are generated inside server-active weights, which keeps the device subnet from selecting channels outside the server subnet.
- HN updates use a FLOPs-aware objective with task loss, `lambda_reg`, and an optional smoothness term controlled by `hn_smooth_coeff`.
- `server_flops_retention` and `device_flops_retention` define the target retained FLOPs for the server and device subnetworks.
- `hn_min_retention` and `hn_max_retention` bound the predicted retention ratios.
- `hn_stochastic_gate` can switch CNN-mask application to a softer training path, while `dwnp_warmup_rounds` keeps device-wise extra pruning disabled early on.
- `r_w`, `r_hn`, and `r_theta` control model synchronization, HN local updates, and HN aggregation respectively.
- Client-server messaging keeps the current model, HN, and embedding parameters aligned across rounds.
- After the main communication rounds finish, the server can optionally run one final pruning pass controlled by `dwnp_enable_final_prune`, then broadcast the pruned model for the last fine-tuning round.
