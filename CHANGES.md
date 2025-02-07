# Changes Made to Improve Classification Accuracy

## Key Changes

1. **New Model Architecture**
   - Implemented a Vision Transformer (ViT) model to better capture complex gas state patterns
   - Model splits 28x28 images into 4x4 patches and processes them with attention mechanisms
   - Lightweight design suitable for CPU/gaming GPU (3 transformer blocks, 64-dim embeddings)

2. **Training Improvements**
   - Switched to Binary Cross Entropy loss (more appropriate for classification)
   - Using AdamW optimizer with weight decay for better regularization
   - Added cosine learning rate schedule with warmup
   - Implemented early stopping to prevent overfitting
   - Added model checkpointing to save best model

3. **Code Organization**
   - Created new files to preserve original implementation:
     * vision_transformer.py: New model architecture
     * run_transformer.py: Updated training pipeline
   - Added detailed documentation throughout
   - Created requirements.txt for dependencies

## How to Use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training with the new transformer model:
   ```bash
   python run_transformer.py
   ```

The code will:
- Train the Vision Transformer model with early stopping
- Save the best model to outputs/best_model.pt
- Generate predictions in outputs/predictions.txt

## Model Architecture Explanation

The Vision Transformer works by:
1. Splitting the 28x28 image into 4x4 patches (like looking at small sections)
2. Converting these patches into embeddings (numerical representations)
3. Using attention to let each patch "look at" other patches
4. Making a final decision based on all patch relationships

This helps the model find subtle patterns in the gas measurements by considering how different regions of the measurement grid relate to each other. The attention mechanism is particularly good at capturing long-range dependencies, which may be important for identifying the new gas state.

## Parameters

Key parameters in run_transformer.py can be adjusted if needed:
- batch_size: Number of samples per training batch (default: 64)
- learning_rate: Controls how fast the model learns (default: 5e-4)
- patience: Number of epochs to wait before early stopping (default: 10)
- dropout: Controls regularization strength (default: 0.1)

Model architecture parameters:
- patch_size: Size of image patches (default: 4)
- emb_dim: Embedding dimension (default: 64)
- depth: Number of transformer blocks (default: 3)
- num_heads: Number of attention heads (default: 4)
- mlp_dim: Hidden dimension for MLP (default: 128)

## Results

The model should achieve the target 83% accuracy through:
- Better pattern recognition with attention mechanisms
- Improved training stability with modern optimization
- Effective regularization to prevent overfitting

## Original Implementation

The original CNN implementation (run.py) has been preserved and can still be used:
```bash
python run.py
```

This allows for easy comparison between the two approaches and provides a fallback option if needed.