# Engineering Prompt: Simplify Vision Transformer for Improved Generalization

## Objective
The current training behavior shows rapid overfitting: the model reaches 100% training accuracy almost immediately while validation accuracy remains stagnant. To address this, we need to simplify the Vision Transformer model and adjust training parameters to reduce its capacity. The task for the vision transformer is very simple, so we want to reduce model complexity (including in features) and increase regularization.

---

## Instructions

### 1. Update the Model Architecture (`vision_transformer.py`)

**a. Reduce Model Capacity**

- **Lower the embedding dimension (`emb_dim`):**  
  Change the default from `64` to `32`.  
  *Explanation:* A smaller embedding dimension reduces the overall number of parameters, which can help avoid memorizing the training data.

- **Reduce the MLP hidden dimension (`mlp_dim`):**  
  Change the default from `128` to `64`.  
  *Explanation:* This further lowers the capacity of the feed-forward network in each transformer block.

- **Decrease the number of transformer blocks (`depth`):**  
  Change the default from `3` to `1`.  
  *Explanation:* Fewer transformer blocks simplify the overall model architecture, which is beneficial for a simple classification task.

- **Reduce the number of attention heads (`num_heads`):**  
  Change the default from `4` to `2` (or even `1` if necessary).  
  *Explanation:* This reduction will decrease the complexity of the multi-head attention mechanism.

**b. Increase Regularization**

- **Increase the dropout probability:**  
  Change the default dropout value from `0.1` to `0.2` or `0.3` in all relevant modules (patch embedding, multi-head attention, MLP blocks).  
  *Explanation:* Higher dropout rates can help mitigate overfitting by randomly deactivating a portion of the neurons during training.

*Action:* Update the constructor defaults and any instantiation of these components in the `VisionTransformer` and its submodules.

---

### 2. Modify the Training Pipeline (`run_transformer.py`)

**a. Update Model Instantiation**

- **Pass the simplified parameters when initializing the Vision Transformer:**  
  For example, set:
  - `emb_dim=32`
  - `depth=1`
  - `num_heads=2`
  - `mlp_dim=64`
  - `dropout=0.2` (or `0.3` as needed)

  *Explanation:* This ensures that the training pipeline uses the simplified version of the model, reducing the chance of overfitting.

**b. (Optional) Review Training Hyperparameters**

- Although the primary focus is reducing model complexity, briefly inspect learning rate, weight decay, and batch size.  
  *Explanation:* The current settings are mostly fine, but minor adjustments here can further support the simplified model if needed.

*Action:* Update the code snippet in `run_transformer.py` where the `VisionTransformer` is instantiated.

---

### 3. Verification and Testing

- **Monitor training and validation metrics:**  
  After making the changes, ensure that:
  - The training accuracy does not hit 100% too quickly.
  - The validation loss and accuracy show more meaningful improvements over epochs.

  *Explanation:* This step confirms that the simplified model is generalizing better and that the training dynamics are improved.

---

## Deliverables

1. **Updated `vision_transformer.py` Code Snippet:**  
   Provide the modified sections with inline comments for each change.

2. **Updated `run_transformer.py` Code Snippet:**  
   Provide the revised model initialization section with inline comments explaining the parameter adjustments.

Ensure that the changes are minimal, focused on reducing the model’s complexity, and maintain the overall functionality for the binary classification task.
