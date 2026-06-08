import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    from main import CausalVLA, vla_sequential_forward

    return CausalVLA, TSNE, optim, plt, torch, vla_sequential_forward


@app.cell
def _(torch):
    # 1. Temporal Data Generation
    # We generate sequences where tokens are autoregressive based on a latent context (Cluster 0 or 1)
    def generate_temporal_data(batch_size=32, seq_len=20, d_model=16):
        # Determine the 'latent context' for each sequence (0 or 1)
        context = torch.randint(0, 2, (batch_size,)).float() # [B]

        # Create temporal sequences: x_t = context + noise
        # The 'temporal' aspect: context remains fixed for the entire sequence, 
        # forcing the VLA to maintain a stable memory state for that sequence.
        x = torch.zeros(batch_size, seq_len, d_model)
        for b in range(batch_size):
            x[b] = (context[b] * 2 - 1) + torch.randn(seq_len, d_model) * 0.1

        targets = context.view(-1, 1).repeat(1, seq_len).long()
        return x, targets


    return (generate_temporal_data,)


@app.cell
def _(generate_temporal_data, plt):
    # Generate a small batch for visualization
    # x: [Batch, Time, Features]
    x_example, targets = generate_temporal_data(batch_size=8, seq_len=20, d_model=16)

    # Average across features to project the temporal trajectory to 1D
    x_mean = x_example.mean(dim=-1) 

    plt.figure(figsize=(10, 6))

    for i in range(x_example.shape[0]):
        # Blue for context 0, Red for context 1
        color = 'blue' if targets[i, 0] == 0 else 'red'
        label = 'Context 0' if targets[i, 0] == 0 else 'Context 1'

        # Plotting each sequence trajectory over time
        plt.plot(x_mean[i].numpy(), color=color, alpha=0.6, marker='o', markersize=4, label=label)

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Visualization of Input Temporal Sequences")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Mean Feature Value (Projected)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    return


@app.cell
def _(CausalVLA, optim):
    # 2. Training Setup
    ## vla
    model_vla = CausalVLA(d_model=16, d_hidden=8, cr=False)
    optimizer_vla = optim.Adam(model_vla.parameters(), lr=1e-3)
    ## vla-cr
    model_vla_cr = CausalVLA(d_model=16, d_hidden=8, cr=True)
    optimizer_vla_cr = optim.Adam(model_vla_cr.parameters(), lr=1e-3)
    return model_vla, model_vla_cr, optimizer_vla, optimizer_vla_cr


@app.cell
def _(
    generate_temporal_data,
    model_vla,
    model_vla_cr,
    optimizer_vla,
    optimizer_vla_cr,
):
    # 3. Unified Training Loop
    model_vla.train()
    model_vla_cr.train()

    for epoch in range(1000):
        x, y = generate_temporal_data()

        # Train VLA
        optimizer_vla.zero_grad()
        _, loss_vla = model_vla(x, y)
        loss_vla.backward()
        optimizer_vla.step()

        # Train VLA-CR
        optimizer_vla_cr.zero_grad()
        # Ensure model_vla_cr uses cr=True in vla_sequential_forward
        _, loss_vla_cr = model_vla_cr(x, y) 
        loss_vla_cr.backward()
        optimizer_vla_cr.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | VLA Loss: {loss_vla.item():.4f} | VLA-CR Loss: {loss_vla_cr.item():.4f}")
    return x, y


@app.cell
def _(TSNE, model_vla, model_vla_cr, plt, torch, vla_sequential_forward, x, y):
    # 4. Comparative Temporal Cluster Visualization (t-SNE)
    model_vla.eval()
    model_vla_cr.eval()

    with torch.no_grad():
        # Retrieve memory history for both models
        _, _, hist_base = vla_sequential_forward(
            x, 
            model_vla.w_q.weight, model_vla.w_k.weight, 
            model_vla.w_v.weight, model_vla.w_u.weight, 
            cr=False
        )

        _, _, hist_cr = vla_sequential_forward(
            x, 
            model_vla_cr.w_q.weight, model_vla_cr.w_k.weight, 
            model_vla_cr.w_v.weight, model_vla_cr.w_u.weight, 
            cr=True
        )

        # Flatten and prepare t-SNE
        def get_tsne(hist):
            flat = hist.view(-1, hist.size(-2) * hist.size(-1)).cpu().numpy()
            return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(flat)

        emb_base = get_tsne(hist_base)
        emb_cr = get_tsne(hist_cr)
        labels_tsne = y.view(-1).cpu().numpy()

        # Plot side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.scatter(emb_base[:, 0], emb_base[:, 1], c=labels_tsne, cmap='coolwarm', alpha=0.3)
        ax1.set_title("VLA (Standard) - No CR")

        ax2.scatter(emb_cr[:, 0], emb_cr[:, 1], c=labels_tsne, cmap='coolwarm', alpha=0.3)
        ax2.set_title("VLA-CR (Contrastive Regularization)")

        for ax in [ax1, ax2]:
            ax.set_xlabel("t-SNE Dim 1")
            ax.set_ylabel("t-SNE Dim 2")

        plt.tight_layout()
        plt.show()
    return


if __name__ == "__main__":
    app.run()
