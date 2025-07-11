tokenization_pipeline = None
conv1_patch_size = None
convb_patch_size = None

def add_transformer_args(pTokenization_pipeline, pConv1_patch_size, pConvb_patch_size):
    global tokenization_pipeline, conv1_patch_size, convb_patch_size
    tokenization_pipeline = pTokenization_pipeline
    conv1_patch_size = pConv1_patch_size
    convb_patch_size = pConvb_patch_size