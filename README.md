# PDV: Prompt Directional Vectors for Zero-shot Composed Image Retrieval

## Overview

PDV introduces **Prompt Directional Vectors** for zero-shot composed image retrieval (ZS-CIR). The core idea is to compute a directional vector in the embedding space that captures the semantic change described by a text prompt, and apply it to compose query representations without any task-specific training.

Given a reference image and a text prompt describing desired modifications, PDV:

1. Computes a **Prompt Directional Vector** as the difference between text embeddings with and without the prompt
2. Applies the directional vector at adjustable scales to steer retrieval in the embedding space
3. Fuses composed text and composed image embeddings for improved retrieval

<p align="center">
  <img src="assets/main_diagram.png" width="100%" alt="PDV method overview"/>
</p>

## Links

- **Paper:** [arXiv 2502.07215](https://arxiv.org/abs/2502.07215)
- **Video:** [YouTube](https://youtu.be/07YbMBqB-gk)

## PDV Implementation

```python
def calculate_pdv_features(feature_text, feature_text_composed, feature_image,
                            alpha_i=1, alpha_t=1, beta=1):
    """
    Calculates enhanced multimodal features using Prompt Difference Vector (PDV) approach.

    Parameters:
    - feature_text: Features extracted from the text branch of the VLM,
                    representing the text-only encoding (e.g., from text inversion or captioning)
    - feature_text_composed: Features of text with compositional prompt, representing
                             text encoding with additional prompt information
    - feature_image: Features extracted from the visual branch of the VLM,
                     representing the visual-only encoding
    - alpha_i: Scaling factor for applying PDV to image features (default=1)
    - alpha_t: Scaling factor for applying PDV to text features (default=1)
    - beta: Weighting factor for combining PDV-enhanced features (default=1)

    Returns:
    - Normalized combined feature vector enhanced with PDV
    """
    # Normalize all input features to unit length
    feature_text = normalize(feature_text, dim=-1)
    feature_text_composed = normalize(feature_text_composed, dim=-1)
    feature_image = normalize(feature_image, dim=-1)

    # Calculate the Prompt Difference Vector (PDV)
    # This captures the semantic difference added by the compositional prompt
    pdv = feature_text_composed - feature_text

    # Apply PDV to image features with scaling factor alpha_i
    # This enhances the image representation with prompt-related information
    feature_PDVI = feature_image + alpha_i * pdv

    # Apply PDV to text features with scaling factor alpha_t
    # This enhances the text representation with additional prompt influence
    feature_PDVT = feature_text + alpha_t * pdv

    # Combine the PDV-enhanced features with weighting factor beta
    # Higher beta values emphasize text features; lower values emphasize image features
    feature_PDVF = (1 - beta) * feature_PDVI + beta * feature_PDVT

    # Normalize and return the final feature vector
    return normalize(feature_PDVF, dim=-1)
```

## Code Release

The code will be released upon publication of the paper. Stay tuned!

## Citation

```bibtex
@inproceedings{tursun2026pdv,
  title={PDV: Prompt Directional Vectors for Zero-shot Composed Image Retrieval},
  author={Tursun, Osman and Kalkan, Sinan and Denman, Simon and Fookes, Clinton},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## Questions

If you have any questions, please feel free to [open an issue](../../issues).
