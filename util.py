# ==========================================
# Define Function to Replace Sigmoid in Existing Models
# ==========================================
def replace_sigmoid_with_modified(model, scale=1.1, shift=0.05):
    """
    Recursively traverse all sub-modules of an existing model, 
    replacing standard nn.Sigmoid with ScaledSigmoid.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Sigmoid):
            setattr(model, name, ScaledSigmoid(scale=scale, shift=shift))
        else:
            # Recursively search sub-modules
            replace_sigmoid_with_modified(module, scale, shift)
    return model