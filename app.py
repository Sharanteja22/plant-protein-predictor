import streamlit as st
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from esm_utils import load_esm_model, load_projection, extract_embeddings_and_attention
from graph_builder import build_graph
from inference import load_gat_model, predict
from llm_utils import generate_explanation
from graph_visualization import create_3d_residue_graph

MODEL_DIR = "models"

device = "cpu"

st.title("🌱 Plant Protein Function Predictor")


sequence = st.text_area("Enter Amino Acid Sequence")

if st.button("Predict"):

    if not sequence:
        st.warning("Please enter a sequence.")
    else:
        with st.spinner("Loading models..."):

            esm_model, alphabet = load_esm_model(device)
            projection = load_projection(os.path.join(MODEL_DIR, "projection.pt"), device)
            gat_model = load_gat_model(os.path.join(MODEL_DIR, "gat_mf.pt"), device)

            idx_to_go = torch.load(os.path.join(MODEL_DIR, "mf_idx_to_go.pt"))

        with st.spinner("Processing sequence..."):

            embeddings, attention = extract_embeddings_and_attention(
                esm_model,
                alphabet,
                sequence,
                projection,
                device
            )

            graph = build_graph(embeddings, attention)

            results, edge_index, attn_weights = predict(
                gat_model,
                graph,
                idx_to_go,
                device="cpu",
                top_k=5
            )

            from src.inference import compute_residue_importance

            residue_importance = compute_residue_importance(
                edge_index,
                attn_weights,
                graph.x.shape[0]
            )


        st.success("Prediction Complete")

        st.subheader("Top-5 Predicted GO Terms")
        for go, score in results:
            st.write(f"{go} — Confidence: {score:.4f}")
        
        st.subheader("Biological Explanation")

        go_list = [go for go, _ in results]
        explanation = generate_explanation(go_list)

        st.write(explanation)

        # st.subheader("Residue Importance Heatmap")

        # fig, ax = plt.subplots(figsize=(12, 2))

        # heatmap = residue_importance.reshape(1, -1)

        # im = ax.imshow(
        #     heatmap,
        #     aspect="auto",
        #     cmap="hot"
        # )

        # ax.set_yticks([])
        # ax.set_xlabel("Residue Position")
        # fig.colorbar(im, orientation="vertical")

        # st.pyplot(fig)

        st.subheader("3D Residue Interaction Network")
        st.info("Interactive 3D visualization showing top residues and their connections. Rotate, zoom, and hover for details!")

        fig_3d = create_3d_residue_graph(
            edge_index=edge_index,
            residue_importance=residue_importance,
            num_residues_to_show=20,
            sequence_length=len(sequence)
        )

        st.plotly_chart(fig_3d, use_container_width=True)


