# pip install llmcompressor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

MODEL_ID = "/home/ubuntu/olmocr/olmOCR-7B-0225-preview"

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the simple PTQ quantization
# recipe = QuantizationModifier(
#   targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Configure pre-defined qwen2vl recipe
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head", "re:visual.*"],
)

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic-Recipe"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
